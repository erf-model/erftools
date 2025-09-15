import numpy as np
import xarray as xr

from scipy.optimize import root_scalar

from ..constants import R_d, Cp_d, Gamma, CONST_GRAV, p_0
from ..utils import get_lo_faces, get_hi_faces


class RealInit(object):
    """Initialize some quantities like WRF's real.exe
    """

    def __init__(self,
                 zsurf=None,
                 eta_half_levels=None,eta_levels=None,p_d=None,
                 ptop=10e3,
                 T0=290.0,A=50.,Tmin=200.,Tlp_strat=-11.,p_strat=0.,
                 etac=0.2,
                 dtype=np.float64):
        """Generate the base state, which is dictated by the surface
        elevation and a set of atmospheric constants
        -- see Section 5.2.2 in the WRF tech note
        
        Vertical levels are dictated by eta_levels, eta_half_levels, or
        p_d.

        Parameters
        ----------
        zsurf: xarray Dataset or DataArray
            Surface elevation map with west_east, south_north dims
        eta_levels or eta_half_levels: array-like
            Array of full/half eta levels (i.e., staggered/unstaggered,
            respectively), with eta being 1 at the surface and 0 at ptop
        p_d: xarray Dataset or DataArray
            Dry reference pressure in an air column at half levels
            (bottom_top dim); specify eta_half_levels or p_d
        T0: float, optional
            reference sea level temperature
        A: float, optional
            Temperature difference between p0 and p0/e, where p0 is the
            reference sea level pressure, a model constant
        Tmin: float, optional
            Minimum temperature permitted
        Tlp_strat: float, optional
            Stndard stratosphere lapse rate
        p_strat: float, optional
            Pressure at which stratospheric warming begins
        etac: float, optional
            Altitude above which eta surfaces are isobaric
        dtype: numpy numeric type, optional
            Change precision (e.g., to np.float32) to facilitate
            comparisons with real.exe outputs (wrfinput, wrfbdy)
        """
        if zsurf is None:
            zsurf = xr.DataArray([[0]],dims=('west_east','south_north'),name='HGT')
        else:
            assert isinstance(zsurf, (xr.Dataset, xr.DataArray)), \
                    'Only xarray data supported'
            assert ('west_east' in zsurf.dims) and \
                   ('south_north' in zsurf.dims), \
                   'WRF dimensions expected'
        self.dtype = dtype
        self.z_surf = zsurf.astype(dtype) # WRF "HGT"
        self.g = dtype(CONST_GRAV)
        self.R_d = dtype(R_d)
        self.Cp_d = dtype(Cp_d)
        self.p_0 = dtype(p_0)
        self.p_top = dtype(ptop)
        self.T0 = dtype(T0) # surface reference temperature
        self.A = dtype(A) # temperature lapse rate
        self.Tmin = dtype(Tmin) # minimum temperature permitted
        self.Tlpstrat = dtype(Tlp_strat) # standard stratosphere lapse rate
        self.pstrat = dtype(p_strat) # pressure at which stratospheric warming begins

        # base-state surface pressure
        TbyA = self.T0 / self.A
        self.pb_surf = self.p_0 * np.exp(
                -TbyA + np.sqrt(TbyA**2 - 2.*self.g*self.z_surf/self.A/self.R_d))

        # base-state dry air mass in column
        self.mub = self.pb_surf - self.p_top

        self.etac = dtype(etac)
        self._setup_hybrid_consts()

        if ((eta_levels is not None) or
            (eta_half_levels is not None) or
            (p_d is not None)
        ):
            # finish initialization
            self.init_base_state(eta_levels=eta_levels,
                                 eta_half_levels=eta_half_levels,
                                 p_d=p_d,
                                 dtype=dtype)
        else:
            print('Note: Base state initialization incomplete; '
                  'none of eta_levels, eta_half_levels, p_d was specified')

    def init_base_state(self, eta_levels=None, eta_half_levels=None, p_d=None,
                        dtype=np.float64):
        # calculate hybrid coordinate if not provided
        if eta_levels is not None:
            if isinstance(eta_levels,list):
                eta_levels = np.array(eta_levels)
            eta_levels = eta_levels.astype(dtype)
            if isinstance(eta_levels, xr.DataArray):
                eta_h = 0.5*(  eta_levels.isel(bottom_top_stag=slice(1,None)).values
                             + eta_levels.isel(bottom_top_stag=slice(0,  -1)).values )
                self.eta_h = xr.DataArray(eta_h, dims='bottom_top', name='eta')
                self.eta_f = eta_levels
            else:
                eta_h = 0.5*(eta_levels[1:] + eta_levels[:-1])
                self.eta_h = xr.DataArray(eta_h, dims='bottom_top', name='eta')
                self.eta_f = xr.DataArray(eta_levels, dims='bottom_top_stag', name='eta')
        else:
            # calculate or set self.eta_h
            if eta_half_levels is None:
                self._calc_eta(p_d)
            else:
                if isinstance(eta_half_levels,list):
                    eta_half_levels = np.array(eta_half_levels)
                self.eta_h = eta_half_levels.astype(dtype)
            # save eta at half levels
            if isinstance(self.eta_h, xr.DataArray):
                eta_h = self.eta_h.values
            else:
                eta_h = self.eta_h
                self.eta_h = xr.DataArray(eta_h, dims='bottom_top', name='eta')
            # calculate eta at full levels
            eta_levels = np.zeros(len(eta_h)+1, dtype=dtype)
            eta_levels[0] = 1.0
            eta_levels[1:-1] = 0.5*(eta_h[1:] + eta_h[:-1])
            self.eta_f = xr.DataArray(eta_levels, dims='bottom_top_stag', name='eta')

        self.rdnw = 1./self.eta_f.diff('bottom_top_stag').rename(bottom_top_stag='bottom_top')
        self.eta_levels = self.eta_f # alias

        # calculate column functions
        self._calc_column_funcs()

        # finish initializing the base state
        self._calc_base_state()
    
    def _calc_eta(self,p_d):
        """Calc WRF hybrid coordinate based on dry hydrostatic pressure

        Some base state quantities are initialized here...
        """
        assert p_d is not None, 'Need to call get_zlevels_auto to get eta levels'
        
        # calculate eta from known dry pressure
        print('Computing eta from',p_d.values)
        assert isinstance(p_d, (xr.Dataset, xr.DataArray)), \
                'Only xarray data supported for p_d for now'
        assert len(p_d.dims) == 1, 'Expected column of pressures'
        assert ('bottom_top_stag' in p_d.dims) or \
               ('bottom_top' in p_d.dims), \
               'Missing vertical dimension'
        assert 'bottom_top' in p_d.dims, 'Only handle "half-levels" for now'
        p_d = p_d.astype(self.dtype)
        eta = np.zeros_like(p_d)
        mub = self.p_0 - self.p_top # corresponding to pb_surf for z=0
        for k, pk in enumerate(p_d):
            def eqn5p4(η):
                B = blending_func(η, self.etac)
                return B*mub + (η - B)*(self.p_0 - self.p_top) + self.p_top - pk
            soln = root_scalar(eqn5p4, bracket=(0,1))
            eta[k] = soln.root
        self.eta_h = xr.DataArray(eta, dims='bottom_top')

    def _setup_hybrid_consts(self):
        one   = self.dtype(1)
        two   = self.dtype(2)
        three = self.dtype(3)
        four  = self.dtype(4)
        etac  = self.etac

        self.B1 = two * etac*etac * ( one - etac )
        self.B2 = -etac * ( four - three * etac - etac*etac*etac )
        self.B3 = two * ( one - etac*etac*etac )
        self.B4 = - ( one - etac*etac )
        self.B5 = np.power(one - etac, 4, dtype=self.dtype)
        #print(self.B1/self.B5,
        #      self.B2/self.B5,
        #      self.B3/self.B5,
        #      self.B4/self.B5,
        #      (self.B1+self.B2+self.B3+self.B4)/self.B5)

    def _calc_column_funcs(self):
        """For WRF hybrid coordinates ("HYBRID_OPT" == 2) with Klemp polynomial
        C3 = B(η)
        C4 = (η - B(η))(p_0 - p_top)

        η_c (`etac`) is the eta at which the hybrid coordinate becomes
        a pure pressure coordinate
        """
        half  = self.dtype(0.5)
        one   = self.dtype(1)
        B1    = self.B1
        B2    = self.B2
        B3    = self.B3
        B4    = self.B4
        B5    = self.B5

        # full levels (staggered)
        f = self.eta_f
        self.C3f = ( B1 + B2*f + B3*f*f + B4*f*f*f ) / B5
        self.C3f[0] = 1
        self.C3f[f < self.etac] = 0
        self.C4f = ( f - self.C3f ) * ( self.p_0 - self.p_top )
        self.C3f.name = 'C3F'
        self.C4f.name = 'C4F'

        # half levels
        h = self.eta_h
        self.C3h = half*(self.C3f[1:] + self.C3f[:-1]).rename(bottom_top_stag='bottom_top')
        self.C4h = ( h - self.C3h ) * ( self.p_0 - self.p_top )
        self.C3h.name = 'C3H'
        self.C4h.name = 'C4H'

        # c1 = dB/d(eta)
        self.C1f = 0.0*self.C3f
        dC3h = self.C3h.values[1:] - self.C3h.values[:-1]
        deta = self.eta_h.values[1:] - self.eta_h.values[:-1]
        self.C1f.loc[dict(bottom_top_stag=slice(1,-1))] = dC3h/deta
        self.C1f[0] = one
        self.C1f[-1] = 0.0
        self.C2f = (one - self.C1f) * (self.p_0 - self.p_top)
        self.C1f.name = 'C1F'
        self.C2f.name = 'C2F'

        self.C1h = half*(self.C1f[1:] + self.C1f[:-1]).rename(bottom_top_stag='bottom_top')
        self.C2h = (one - self.C1h) * (self.p_0 - self.p_top)
        self.C1h.name = 'C1H'
        self.C2h.name = 'C2H'

    def _calc_base_state(self):
        # dry hydrostatic base-state pressure (WRF Eqn. 5.4)
        self.pb = self.C3h * (self.pb_surf - self.p_top) + self.C4h + self.p_top
        self.pb.name = 'PB'
        
        # reference dry temperature
        self.Td = self.T0 + self.A * np.log(self.pb / self.p_0)
        self.Td = np.maximum(self.Tmin, self.Td)
        if self.pstrat > 0:
            strat = np.where(self.pb < self.pstrat)
            self.Td[strat] = self.Tmin \
                    + self.Tlpstrat*np.log(self.pb / self.pstrat)
        self.Td.name = 'Tdry'

        # reference dry potential temperature (WRF Eqn. 5.5)
        self.thd = self.Td * (self.p_0 / self.pb)**(self.R_d/self.Cp_d)
        self.thd.name = 'TH'

        # reciprocal reference density (WRF Eqn. 5.6)
        self.alb = (self.R_d*self.thd)/self.p_0 \
                 * np.power(self.pb / self.p_0, -1./Gamma, dtype=self.dtype)
        self.alb.name = 'ALB'

        self.rb = 1. / self.alb
        self.rb.name = 'RB'

        # base-state geopotential from hypsometric equation
        stag_dims = {'bottom_top_stag':len(self.eta_levels)}
        for dim,n in self.z_surf.sizes.items():
            stag_dims[dim] = n
        pfu = get_hi_faces(self.C3f)*self.mub + get_hi_faces(self.C4f) + self.p_top
        pfd = get_lo_faces(self.C3f)*self.mub + get_lo_faces(self.C4f) + self.p_top
        phm =              self.C3h *self.mub +              self.C4h  + self.p_top
        dphb = self.alb*phm * np.log(pfd/pfu)
        self.phb = xr.DataArray(np.zeros(tuple(stag_dims.values())),
                                dims=stag_dims, name='PHB')
        self.phb.loc[dict(bottom_top_stag=slice(1,None))] = dphb.cumsum('bottom_top').values
        self.phb += self.g * self.z_surf
        #DEBUG:
        self.dphb = dphb
        self.pfu = pfu
        self.pfd = pfd
        self.phm = phm


def blending_func(eta, etac=0.2):
    """Relative weighting function to blend between terrain-following sigma
    corodinate and pure pressure coordinate, B(η)

    see dyn_em/nest_init_utils.F
    """
    if eta < etac:
        return 0
    B1 = 2. * etac**2 * ( 1. - etac )
    B2 = -etac * ( 4. - 3. * etac - etac**3 )
    B3 = 2. * ( 1. - etac**3 )
    B4 = - ( 1. - etac**2 )
    B5 = (1.-etac)**4
    return ( B1 + B2*eta + B3*eta**2 + B4*eta**3 ) / B5


def get_zlevels_auto(nlev,
                     dzbot=50.,
                     dzmax=1000.,
                     dzstretch_s=1.3,
                     dzstretch_u=1.1,
                     ptop=5000.,
                     T0=290.,
                     geopotential_height=False,
                     verbose=False):
    """Following the description in the WRF User's Guide, vertical
    grid levels can be determined based on surface and maximum grid
    spacings (dz0, dzmax) and the surface and upper stretching factors
    (s0, s). nlev is the number of unstaggered ("half") levels.

    Assuming an isothermal atmosphere T0 that is hard-coded in WRF.

    Returns _staggered_ heights, pressure levels, and eta levels.

    See `levels` subroutine in dyn_em/module_initialize_real.F
    """
    zup = np.zeros(nlev+1) # Staggered grid heights (high side). In WRF, zup
    pup = np.zeros(nlev+1) #   and pup range from 1:nlev (inclusive) whereas
    eta = np.zeros(nlev+1) #   eta ranges from 0:nlev; here, we dimension
                           #   zup and pup to have size nlev+1 so that the
                           #   indices in both codes match for convenience
    zscale = R_d * T0 / CONST_GRAV
    ztop = zscale * np.log(p_0 / ptop)
    dz = dzbot
    zup[1] = dzbot
    pup[0] = p_0
    pup[1] = p_0 * np.exp(-zup[1]/zscale)
    eta[0] = 1.0
    eta[1] = (pup[1] - ptop) / (p_0 - ptop)
    if verbose:
        print(0,None,zup[0],eta[0]) # zup[0] doesn't exist in WRF
        print(1,dz,zup[1],eta[1])
    for i in range(1,nlev):
        a = dzstretch_u + (dzstretch_s - dzstretch_u) \
                        * max((0.5*dzmax - dz)/(0.5*dzmax), 0)
        dz = a * dz
        dztest = (ztop - zup[i]) / (nlev-i)
        if dztest < dz:
            if verbose:
                print('--- now, constant dz ---')
            break
        zup[i+1] = zup[i] + dz
        pup[i+1] = p_0 * np.exp(-zup[i+1]/zscale)
        eta[i+1] = (pup[i+1] - ptop) / (p_0 - ptop)
        if verbose:
            print(i+1,dz,zup[i+1],eta[i+1],'a=',a)
        assert i < nlev, 'Not enough eta levels to reach p_top, need to:'\
                         ' (1) add more eta levels,'\
                         ' (2) increase p_top to reduce domain height,'\
                         ' (3) increase min dz, or'\
                         ' (4) increase the stretching factor(s)'
    dz = (ztop - zup[i]) / (nlev - i)
    assert dz <= 1.5*dzmax, 'Upper levels may be too thick, need to:'\
                            ' (1) add more eta levels,'\
                            ' (2) increase p_top to reduce domain height,'\
                            ' (3) increase min dz'\
                            ' (4) increase the stretching factor(s), or'\
                            ' (5) increase max dz'
    ilast = i
    for i in range(ilast,nlev):
        zup[i+1] = zup[i] + dz
        pup[i+1] = p_0 * np.exp(-zup[i+1]/zscale)
        eta[i+1] = (pup[i+1] - ptop) / (p_0 - ptop)
        if verbose:
            print(i+1,dz,zup[i+1],eta[i+1])
    assert np.allclose(ztop,zup[-1])

    # calculate geopotential height based on a standard atmosphere
    if geopotential_height:
        real = RealInit(eta_levels=eta, ptop=ptop)
        phb = real.phb.values.squeeze()
        zup = phb / CONST_GRAV

        if verbose:
            dz = np.diff(zup)
            print(f'Full level index = {0:4d}   '
                  f'Height = {zup[0]:7.1f} m')
            for k in range(1,len(phb)):
                print(f'Full level index = {k:4d}   '
                      f'Height = {zup[k]:7.1f} m    '
                      f'Thickness = {dz[k-1]:6.1f} m')

    return zup,pup,eta
