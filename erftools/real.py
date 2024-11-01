import numpy as np
import xarray as xr

from scipy.optimize import root_scalar

from .constants import R_d, Cp_d, Gamma, CONST_GRAV, p_0

# staggered levels (Benjamin et al. 2016 MWR)
hrrr_eta = np.array([1.0000, 0.9980, 0.9940, 0.9870, 0.9750, 0.9590, 0.9390,
                     0.9160, 0.8920, 0.8650, 0.8350, 0.8020, 0.7660, 0.7270,
                     0.6850, 0.6400, 0.5920, 0.5420, 0.4970, 0.4565, 0.4205,
                     0.3877, 0.3582, 0.3317, 0.3078, 0.2863, 0.2670, 0.2496,
                     0.2329, 0.2188, 0.2047, 0.1906, 0.1765, 0.1624, 0.1483,
                     0.1342, 0.1201, 0.1060, 0.0919, 0.0778, 0.0657, 0.0568,
                     0.0486, 0.0409, 0.0337, 0.0271, 0.0209, 0.0151, 0.0097,
                     0.0047, 0.0000])
hrrr_eta = xr.DataArray(hrrr_eta, dims='bottom_top_stag', name='eta')

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


def get_hi_faces(darr,dim='bottom_top_stag'):
    assert dim.endswith('_stag')
    return darr.isel({dim:slice(1,None)}).rename({dim:dim[:-5]})

def get_lo_faces(darr,dim='bottom_top_stag'):
    assert dim.endswith('_stag')
    return darr.isel({dim:slice(0,-1)}).rename({dim:dim[:-5]})


class RealInit(object):
    """Initialize some quantities like WRF's real.exe
    """

    def __init__(self,
                 zsurf,
                 eta=None,eta_stag=None,p_d=None,
                 ptop=10e3,
                 T0=290.0,A=50.,Tmin=200.,Tlp_strat=-11.,p_strat=0.,
                 etac=0.2,
                 dtype=np.float64):
        """Start with just the base state, determined from surface
        elevation alone
        
        We can set constants to be 32-bit to enable closer comparisons
        with real.exe outputs (wrfinput, wrfbdy)
        """
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

        # calculate hybrid coordinate
        if eta_stag is not None:
            self.eta_stag = eta_stag.astype(dtype)
            self.eta = 0.5*(self.eta_stag[1:] + self.eta_stag[:-1])
            self.eta = self.eta.rename(bottom_top_stag='bottom_top')
        else:
            if eta is None:
                self.calc_eta(p_d)
            else:
                self.eta = eta.astype(dtype)
            self.eta_stag = np.zeros(len(self.eta)+1, dtype=dtype)
            self.eta_stag[0] = 1.0
            self.eta_stag[1:-1] = self.eta[1:] + self.eta[:-1]
            self.eta_stag = self.eta_stag.rename(bottom_top='bottom_top_stag')
        self.rdnw = 1./self.eta_stag.diff('bottom_top_stag').rename(bottom_top_stag='bottom_top')

        # calculate column functions
        self.calc_column_funcs(dtype(etac))

        # finish initializing the base state
        self.calc_base_state()
    
    def calc_eta(self,p_d):
        """Calc WRF hybrid coordinate based on dry hydrostatic pressure

        Some base state quantities are initialized here...
        """
        # automatic eta computed in dyn_em/module_initailize_real.F
        assert p_d is not None, 'WRF compute_eta not implemented, need to specify target p_d'
        
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
        self.eta = xr.DataArray(eta, dims='bottom_top')

    def calc_column_funcs(self,etac):
        """For WRF hybrid coordinates ("HYBRID_OPT" == 2) with Klemp polynomial
        C3 = B(η)
        C4 = (η - B(η))(p_0 - p_top)

        η_c (`etac`) is the eta at which the hybrid coordinate becomes
        a pure pressure coordinate
        """
        one   = self.dtype(1)
        two   = self.dtype(2)
        three = self.dtype(3)
        four  = self.dtype(4)
        half  = self.dtype(0.5)

        B1 = two * etac*etac * ( one - etac )
        B2 = -etac * ( four - three * etac - etac*etac*etac )
        B3 = two * ( one - etac*etac*etac )
        B4 = - ( one - etac*etac )
        B5 = np.power(one - etac, 4, dtype=self.dtype)
        #print(B1/B5,B2/B5,B3/B5,B4/B5, (B1+B2+B3+B4)/B5)

        # full levels (staggered)
        f = self.eta_stag
        self.C3f = ( B1 + B2*f + B3*f*f + B4*f*f*f ) / B5
        self.C3f[0] = 1
        self.C3f[f < etac] = 0
        self.C4f = ( f - self.C3f ) * ( self.p_0 - self.p_top )
        self.C3f.name = 'C3F'
        self.C4f.name = 'C4F'

        # half levels
        h = self.eta
        self.C3h = half*(self.C3f[1:] + self.C3f[:-1]).rename(bottom_top_stag='bottom_top')
        self.C4h = ( h - self.C3h ) * ( self.p_0 - self.p_top )
        self.C3h.name = 'C3H'
        self.C4h.name = 'C4H'

        # c1 = dB/d(eta)
        self.C1f = 0.0*self.C3f
        dC3h = self.C3h.values[1:] - self.C3h.values[:-1]
        deta = self.eta.values[1:] - self.eta.values[:-1]
        self.C1f.loc[dict(bottom_top_stag=slice(1,-1))] = dC3h/deta
        self.C1f[0] = 1
        self.C1f[-1] = 0
        self.C2f = (one - self.C1f) * (self.p_0 - self.p_top)
        self.C1f.name = 'C1F'
        self.C2f.name = 'C2F'

        self.C1h = half*(self.C1f[1:] + self.C1f[:-1]).rename(bottom_top_stag='bottom_top')
        self.C2h = (one - self.C1h) * (self.p_0 - self.p_top)
        self.C1h.name = 'C1H'
        self.C2h.name = 'C2H'

    def calc_base_state(self):
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

        # reference dry potential temperature (WRF Eqn. 5.5)
        self.thd = self.Td * (self.p_0 / self.pb)**(self.R_d/self.Cp_d)

        # reciprocal reference density (WRF Eqn. 5.6)
        self.alb = (self.R_d*self.thd)/self.p_0 \
                 * np.power(self.pb / self.p_0, -1./Gamma, dtype=self.dtype)
        self.rb = 1. / self.alb

        # base-state geopotential from hypsometric equation
        stag_dims = {'bottom_top_stag':len(self.eta_stag)}
        for dim,n in self.z_surf.sizes.items():
            stag_dims[dim] = n
        pfu = get_hi_faces(self.C3f)*self.mub + get_hi_faces(self.C4f) + self.p_top
        pfd = get_lo_faces(self.C3f)*self.mub + get_lo_faces(self.C4f) + self.p_top
        phm =              self.C3h *self.mub +              self.C4h  + self.p_top
        dphb = self.alb*phm * np.log(pfd/pfu)
        self.phb = xr.DataArray(np.zeros(tuple(stag_dims.values())), dims=stag_dims)
        self.phb.loc[dict(bottom_top_stag=slice(1,None))] = dphb.cumsum('bottom_top').values
        self.phb += self.g * self.z_surf
        #DEBUG:
        self.dphb = dphb
        self.pfu = pfu
        self.pfd = pfd
        self.phm = phm


