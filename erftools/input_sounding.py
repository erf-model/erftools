#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from .constants import R_d, Cp_d, Gamma, CONST_GRAV, p_0
from .wrf.constants import rvovrd, cvpm
from .EOS import getRhogivenThetaPress
from .HSE import Newton_Raphson_hse


class InputSounding(object):

    def __init__(self,fpath=None,
                 p_surf=1e5,
                 th_surf=300.,
                 qv_surf=0.,
                 z_profile=None,
                 th_profile=None,
                 qv_profile=None,
                 qc_profile=None,
                 qr_profile=None,
                 u_profile=None,
                 v_profile=None):
        """File format is described in the WRF User's Guide

        > "The input_sounding file (already in appropriate case
          directories) can be any set of levels that goes at least up to
          the model top height in the namelist. The first line includes
          the surface pressure (hPa), potential temperature (K) and
          moisture mixing ratio (g/kg). Each subsequent line has five
          input values: height (meters above sea-level), dry potential
          temperature (K), vapor mixing ratio (g/kg), x-direction wind
          component (m/s), and y-direction wind component (m/s)."
        """
        assert ((fpath is not None) or
                ((z_profile is not None) and (th_profile is not None))), \
                ('Need to specify the path to an input_sounding file '
                 'or specify profiles of z, th(, qv=0, u=0, v=0)')
        if fpath is not None:
            self._load_wrf_ideal_sounding(fpath)
        elif (z_profile is not None) and (th_profile is not None):
            if qv_profile is None:
                qv_profile = np.zeros_like(z_profile)
            if qc_profile is None:
                qc_profile = np.zeros_like(z_profile)
            if qr_profile is None:
                qr_profile = np.zeros_like(z_profile)
            if u_profile is None:
                u_profile = np.zeros_like(z_profile)
            if v_profile is None:
                v_profile = np.zeros_like(z_profile)
            if z_profile[0] > 0:
                # prepend surface values
                z_profile  = np.concatenate(([0.0],    z_profile ))
                th_profile = np.concatenate(([th_surf],th_profile))
                qv_profile = np.concatenate(([qv_surf],qv_profile))
                qc_profile = np.concatenate(([0.0],    qc_profile))
                qr_profile = np.concatenate(([0.0],    qr_profile))
                u_profile  = np.concatenate(([0.0],    u_profile ))
                v_profile  = np.concatenate(([0.0],    v_profile ))
            assert len(z_profile) \
                    == len(th_profile) == len(qv_profile) \
                    == len(qc_profile) == len(qr_profile) \
                    == len(u_profile) == len(v_profile)
            self.z = z_profile
            self.th = th_profile
            self.qv = qv_profile
            self.qc = qc_profile
            self.qr = qr_profile
            self.qt = self.qv + self.qc + self.qr
            self.u = u_profile
            self.v = v_profile
            self.p_surf = p_surf
            self.th_surf = th_surf
            self.qv_surf = qv_surf


    def _load_wrf_ideal_sounding(self,fpath):
        with open(fpath,'r') as f:
            # The first line includes the surface pressure (hPa), potential
            # temperature (K) and moisture mixing ratio (g/kg).
            p_surf, T_surf, qv_surf = [float(val) for val in f.readline().split()]

            # Each subsequent line has five input values: height (meters above
            # sea-level), dry potential temperature (K), vapor mixing ratio
            # (g/kg), x-direction wind component (m/s), and y-direction wind
            # component (m/s)
            init = np.loadtxt(f)

        self.p_surf = p_surf * 100 # [Pa]
        self.th_surf = T_surf # [K]
        self.qv_surf = qv_surf * 0.001 # [g/g]

        self.z  = init[:,0]
        self.th = init[:,1]
        self.qv = init[:,2] * 0.001
        self.u  = init[:,3]
        self.v  = init[:,4]


    def interp_levels(self,z):
        """Interpolate profile to specified levels"""
        z = sorted(z)
        assert (z[0] >= self.z[0]) and (z[-1] <= self.z[-1])
        self.th = np.interp(z, self.z, self.th)
        self.qv = np.interp(z, self.z, self.qv)
        self.u  = np.interp(z, self.z, self.u )
        self.v  = np.interp(z, self.z, self.v )
        self.z  = z


    def integrate_column_wrf(self,verbose=False,Niter=10):
        """Follow legacy get_sounding subroutine
        from WRF dyn_em/module_initialize_ideal.F

        Notes:
        - `qvf` is not strictly correct as implemented in WRF ideal.exe
          which is consistent with WRF's "moist theta" which is not
          exactly virtual potential temperature. See, e.g.,
          https://github.com/NCAR/WRFV3/issues/4
        - `qv_surf` is ignored
        - moist integration has factor of (1+qv), implying that the
          density is dry; but the integration starts with rho_surf that
          is moist... (calculated from total p, moist theta on the
          surface)

        DEPRECATED
        """
        qvf = 1. + rvovrd*self.qv[0]
        rho_surf = 1. / ((R_d/p_0)*self.th_surf*qvf*((self.p_surf/p_0)**cvpm))
        pi_surf = (self.p_surf/p_0)**(R_d/Cp_d)
        if verbose:
            print('surface density, pi =',rho_surf,pi_surf)

        # integrate moist sounding hydrostatically, starting from the specified
        # surface pressure
        N = len(self.th)
        self.rho = np.zeros(N)
        self.thm = np.zeros(N)
        self.pm = np.zeros(N)
        self.p = np.zeros(N)

        # 1. Integrate from surface to lowest level
        qvf = 1. + rvovrd*self.qv[0]
        qvf1 = 1.0 + self.qv[0]
        self.rho[0] = rho_surf
        dz = self.z[0]
        for i in range(Niter):
            self.pm[0] = self.p_surf \
                       - 0.5*dz*(rho_surf + self.rho[0])*CONST_GRAV*qvf1
            self.rho[0] = 1./((R_d/p_0)*self.th[0]*qvf*((self.pm[0]/p_0)**cvpm))
        self.thm[0] = self.th[0] * qvf

        # 2. Integrate up the column
        for k in range(1,N):
            self.rho[k] = self.rho[k-1]
            dz = self.z[k] - self.z[k-1]
            qvf1 = 0.5*(2. + (self.qv[k-1] + self.qv[k]))
            qvf = 1. + rvovrd*self.qv[k]
            for i in range(Niter):
                self.pm[k] = self.pm[k-1] \
                        - 0.5*dz*(self.rho[k] + self.rho[k-1])*CONST_GRAV*qvf1
                assert self.pm[k] > 0, 'too cold for chosen height'
                self.rho[k] = 1./((R_d/p_0)*self.th[k]*qvf*((self.pm[k]/p_0)**cvpm))
            self.thm[k] = self.th[k] * qvf
        # we have the moist sounding at this point...

        # 3. Compute the dry sounding using p at the highest level from the
        #    moist sounding and integrating down
        self.p[N-1] = self.pm[N-1]
        for k in range(N-2,-1,-1):
            dz = self.z[k+1] - self.z[k]
            self.p[k] = self.p[k+1] \
                      + 0.5*dz*(self.rho[k]+self.rho[k+1])*CONST_GRAV

        # Note: WRF does not calculate dry density
        self.rhod = np.zeros_like(self.rho)
        self.rhod[:] = np.nan

        if verbose:
            ptmp = p_0 * (R_d * self.rho * self.thm / p_0)**Gamma
            print('error (moist)', np.max(np.abs(self.pm - ptmp)))
            ptmp = p_0 * (R_d * self.rho * self.th / p_0)**Gamma
            print('error (dry)', np.max(np.abs(self.p - ptmp)))


    def _iter_rho_p(self,dz,rlo,plo,th,qv=0,maxiter=100,tol=1e-15,
                    verbose=False):
        qvf = 1. + (rvovrd-1)*qv
        r = rlo # guess for rho
        for it in range(maxiter):
            rlast = r
            p = plo - 0.5*dz*(rlo + r)*CONST_GRAV
            assert p > 0, 'too cold for chosen height' \
                    f' (p={p:g}' \
                    f' rho_lo={rlo:g}' \
                    f' rho={r:g})'
            r = 1. / ((R_d/p_0)*th*qvf*((p/p_0)**cvpm))
            diff = np.abs(r - rlast)
            if verbose:
                print(it,p,r,diff)
            if diff < tol:
                #print(f'Converged after {it+1} iterations')
                break
        assert np.abs(r - rlast) < tol
        return p, r, th*qvf


    def integrate_column(self,verbose=False,**kwargs):
        """Based on dyn_em/module_initialize_ideal.F

        Here, the "moist" theta is virtual potential temperature:
          thm = th * qvf
        where
          qvf = 1 + (R_v/R_d - 1)*qv
        is the moist air correction factor.

        This returns profiles of total density, moist theta as defined
        above, water vapor mixing ratio, and horizontal wind components.

        DEPRECATED
        """
        #qvf = 1. + rvovrd*self.qv[0] # WRF
        qvf = 1. + (rvovrd-1)*self.qv_surf
        rho_surf = 1. / ((R_d/p_0)*self.th_surf*qvf*((self.p_surf/p_0)**cvpm))
        pi_surf = (self.p_surf/p_0)**(R_d/Cp_d)
        if verbose:
            ptmp = p_0 * (R_d * rho_surf * self.th_surf*qvf / p_0)**Gamma
            err = self.p_surf - ptmp
            print('surface density, pi, error =',rho_surf,pi_surf,err)

        # integrate moist sounding hydrostatically, starting from the specified
        # surface pressure
        N = len(self.th)
        self.rho = np.zeros(N)
        self.rhod = np.zeros(N)
        self.thm = np.zeros(N)
        self.pm = np.zeros(N)
        self.p = np.zeros(N)

        # 1. Integrate from surface to lowest level
        self.pm[0],self.rho[0],self.thm[0] = self._iter_rho_p(
            self.z[0], # surface to first cell center
            rho_surf, # guess
            self.p_surf,
            self.th[0],
            self.qv[0],
        )

        if verbose:
            ptmp = p_0 * (R_d * self.rho[0] * self.thm[0] / p_0)**Gamma
            err = self.pm[0] - ptmp
            print('MOIST column')
            print(self.z[0], self.pm[0], self.rho[0], self.thm[0], err)

        # 2. Integrate up the column
        for k in range(1,N):
            self.pm[k],self.rho[k],self.thm[k] = self._iter_rho_p(
                self.z[k] - self.z[k-1],
                self.rho[k-1], # guess
                self.pm[k-1],
                self.th[k],
                self.qv[k],
            )

            if verbose:
                ptmp = p_0 * (R_d * self.rho[k] * self.thm[k] / p_0)**Gamma
                err = self.pm[k] - ptmp
                print(self.z[k], self.pm[k], self.rho[k], self.thm[k], err)

        # we have the moist sounding at this point...

        # 3. Compute the dry sounding using p at the highest level from the
        #    moist sounding and integrating down
        self.p[N-1] = self.pm[N-1] # assume no moisture at top of column
        self.rhod[N-1] = 1./((R_d/p_0)*self.th[N-1]*((self.p[N-1]/p_0)**cvpm))
        for k in range(N-2,-1,-1):
            self.p[k],self.rhod[k],_ = self._iter_rho_p(
                self.z[k] - self.z[k+1],
                self.rhod[k+1], # guess
                self.p[k+1],
                self.th[k],
            )

        if verbose:
            ptmp = p_0 * (R_d * self.rho * self.thm / p_0)**Gamma
            print('error (moist)', np.max(np.abs(self.pm - ptmp)))
            ptmp = p_0 * (R_d * self.rhod * self.th / p_0)**Gamma
            print('error (dry)', np.max(np.abs(self.p - ptmp)))
            print('')
            print('DRY column')
            for k in range(N):
                ptmp = p_0 * (R_d * self.rhod[k] * self.th[k] / p_0)**Gamma
                err = self.p[k] - ptmp
                print(self.z[k], self.p[k], self.rhod[k], self.th[k], err)


    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(
            {
                'p': self.pm,
                'rho': self.rho,
                'theta_m': self.thm,
                'p_d': self.p,
                'rho_d': self.rhod,
                'theta': self.th,
                'u': self.u,
                'v': self.v,
            },
            index=pd.Index(self.z,name='z'))


    def calc_rho_p(self,tol=1e-12,eps=1e-6,verbose=False):
        """Integrate moist soinding hydrostatically, starting from the
        specified surface pressure

        New implementation uses Newton-Raphson iteration
        """
        N = len(self.th)
        self.rhod = np.zeros(N)
        self.pm = np.zeros(N)

        self.pm[0] = self.p_surf
        self.rhod[0] = getRhogivenThetaPress(self.th_surf, self.p_surf,
                                             qv=self.qv_surf)
        if verbose:
            print('Surface pressure, dry density, total density :',
                  self.pm[0],self.rhod[0],
                  self.rhod[0]*(1+self.qt[0]))

        # integrate from surface to domain top
        for k in range(1,N):
            dz = self.z[k] - self.z[k-1]

            # Establish known constant
            rho_tot_lo = self.rhod[k-1] * (1. + self.qt[k-1])
            C = -self.pm[k-1] + 0.5*rho_tot_lo*CONST_GRAV*dz

            #if verbose: print(f'C = {-self.pm[k-1]} + 0.5*{rho_tot_lo}*g*{dz} = {C}')

            # Initial guess and residual
            self.pm[k] = self.pm[k-1]
            self.rhod[k] = getRhogivenThetaPress(self.th[k],
                                                 self.pm[k],
                                                 qv=self.qv[k])
            rho_tot_hi = self.rhod[k] * (1. + self.qt[k])
            F = self.pm[k] + 0.5*rho_tot_hi*CONST_GRAV*dz + C

            # Do iterations
            if np.abs(F) > tol:
                self.pm[k], self.rhod[k] = Newton_Raphson_hse(self.pm[k], # guess
                                                              F, # initial residual
                                                              dz, C, # constants
                                                              self.th[k],
                                                              self.qv[k],
                                                              qt=self.qt[k],
                                                              verbose=verbose)



    def plot(self):
        """Call before integrate_column() to plot just the inputs;
        after integrating through the moist air column, this will plot
        additional quantities: rho and p.
        """
        allplots = True if hasattr(self, 'rho') else False
        if allplots:
            fig,ax = plt.subplots(ncols=5,sharey=True,figsize=(15,6))
        else:
            fig,ax = plt.subplots(ncols=3,sharey=True,figsize=(9,6))

        # plot profiles
        ax[0].plot(self.th, self.z, label='dry')
        if allplots:
            ax[0].plot(self.thm, self.z, label='moist')
        ax[0].plot(self.th_surf, 0, 'ko', markerfacecolor='none')
        ax[1].plot(self.qv, self.z)
        ax[1].plot(self.qv_surf, 0, 'ko', markerfacecolor='none')
        ax[1].set_xlim((0,None))
        ax[2].plot(self.u, self.z, label='u')
        ax[2].plot(self.v, self.z, label='v')
        if allplots:
            try:
                ax[3].plot(self.rhod, self.z, label='dry')
            except AttributeError:
                # not calculated by integrate_column_wrf
                pass
            ax[3].plot(self.rho, self.z, label='moist')
            ax[3].set_xlim((0,None))
            ax[4].plot(self.p, self.z, label='dry')
            ax[4].plot(self.pm, self.z, label='moist')
            ax[4].plot(self.p_surf, 0, 'ko', markerfacecolor='none')
            ax[4].set_xlim((0,None))

        # labels and legends
        ax[0].set_ylabel('altitude AGL [m]')
        if allplots:
            ax[0].set_xlabel('potential temperature [K]')
            ax[0].legend(loc='best')
        else:
            ax[0].set_xlabel('dry potential temperature [K]')
        ax[1].set_xlabel('water vapor mixing ratio [g/g]')
        ax[2].set_xlabel('horizontal wind component [m/s]')
        ax[2].legend(loc='best')
        if allplots:
            ax[3].set_xlabel('density [kg/m^3]')
            ax[3].legend(loc='best')
            ax[4].set_xlabel('pressure [Pa]')
            ax[4].legend(loc='best')

        for axi in ax:
            axi.grid()

        return fig,ax


#==============================================================================
if __name__ == '__main__':

    inp = InputSounding()
    fig,ax = inp.plot()
    fig.savefig('sounding.png',bbox_inches='tight')

    inp.integrate_column_wrf(verbose=True)
    fig,ax = inp.plot()
    fig.savefig('sounding_hse_wrf.png',bbox_inches='tight')

    inp.integrate_column(verbose=True)
    fig,ax = inp.plot()
    fig.savefig('sounding_hse.png',bbox_inches='tight')

