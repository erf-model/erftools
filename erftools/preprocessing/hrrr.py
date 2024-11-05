import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

import cartopy.crs as ccrs
from herbie import Herbie

from ..constants import R_d, R_v, Cp_d, Cp_v, CONST_GRAV, p_0
from ..EOS import getPgivenRTh, getThgivenRandT, getThgivenPandT
from .utils import get_hi_faces, get_lo_faces, get_w_from_omega
from .real import RealInit


hrrr_projection = ccrs.LambertConformal(
    central_longitude=-97.5,
    central_latitude=38.5,
    standard_parallels=[38.5],
    globe=ccrs.Globe(
        ellipse="sphere",
        semimajor_axis=6370000,
        semiminor_axis=6370000,
    ),
)

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

# GRIB2 variable names
varlist = [
    'HGT',  # geopotential height [m]
    'UGRD', # U-component of wind [m/s]
    'VGRD', # V-component of wind [m/s]
    'VVEL', # pressure vertical velocity [Pa/s]
    'PRES', # air pressure [Pa]
    'TMP',  # air temperature [K]
    'SPFH', # specific humidity [kg/kg]
    'CLMR', # cloud water mixing ratio [kg/kg]
    'RWMR', # rain water mixing ratio [kg/kg]
]


class NativeHRRR(object):
    """Get HRRR analysis on native levels and calculate fields
    consistent with WRF
    """

    def __init__(self,datetime,varlist=varlist):
        """Download data from native levels, see
        https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
        for data inventory
        """
        self.H = Herbie(datetime, model='hrrr', product='nat')
        self.H.download(verbose=True)
        self._combine_data(varlist)
        self._setup_grid()

    def _combine_data(self,varlist):
        varstr = '|'.join(varlist)
        ds = self.H.xarray(f':(?:{varstr}):\d+ hybrid') # get all levels
        if isinstance(ds, list):
            ds = xr.merge(ds)
        ds = ds.rename_vars({
            'u': 'U',
            'v': 'V',
            'clwmr': 'QCLOUD',
            'rwmr' : 'QRAIN',
        })

        surf = self.H.xarray(':surface:anl')
        ds['LANDMASK'] = surf['lsm']
        ds['SST']      = surf['t']
        ds['HGT']      = surf['orog']
        ds['PSFC']     = surf['sp']

        self.ds = ds

    def _setup_grid(self):
        lat = self.ds.coords['latitude']
        lon = self.ds.coords['longitude']
        self.xlim = {}
        self.ylim = {}
        # note: transform_points returns an (n,3) array
        self.xlim['sw'],self.ylim['sw'] = \
            hrrr_projection.transform_point(
                    lon.isel(x= 0,y= 0), lat.isel(x= 0,y= 0), ccrs.Geodetic())
        self.xlim['se'],self.ylim['se'] = \
            hrrr_projection.transform_point(
                    lon.isel(x=-1,y= 0), lat.isel(x=-1,y= 0), ccrs.Geodetic())
        self.xlim['ne'],self.ylim['ne'] = \
            hrrr_projection.transform_point(
                    lon.isel(x=-1,y=-1), lat.isel(x=-1,y=-1), ccrs.Geodetic())
        self.xlim['nw'],self.ylim['nw'] = \
            hrrr_projection.transform_point(
                    lon.isel(x= 0,y=-1), lat.isel(x= 0,y=-1), ccrs.Geodetic())

        # SANITY CHECK: our transformed grid is Cartesian
        assert np.allclose(self.xlim['sw'], self.xlim['nw'])
        assert np.allclose(self.xlim['se'], self.xlim['ne'])
        assert np.allclose(self.ylim['sw'], self.ylim['se'])
        assert np.allclose(self.ylim['nw'], self.ylim['ne'])

        # 1-D x array from y=0
        x1 = hrrr_projection.transform_points(ccrs.Geodetic(),
                                              lon.isel(y=0), lat.isel(y=0))
        assert np.allclose(x1[:,1], x1[0,1]) # y values are ~constant
        self.x1 = x1[:,0]

        # 1-D y array from x=0
        y1 = hrrr_projection.transform_points(ccrs.Geodetic(),
                                              lon.isel(x=0), lat.isel(x=0))
        assert np.allclose(y1[:,0], y1[0,0]) # x values are ~constant
        self.y1 = y1[:,1]

        # create dimension coordinates
        self.ds = self.ds.assign_coords(x=self.x1, y=self.y1)

    def __getitem__(self,key):
        return self.ds[key]

    def inventory(self):
        return self.H.inventory()

    def clip(self,xmin,xmax,ymin,ymax,inplace=False):
        """Clip the dataset based on x,y ranges in HRRR projected
        coordinates. If `inplace==False`, return a copy of the clipped
        dataset.
        """
        xlo = self.x1[self.x1 < xmin][-1]
        xhi = self.x1[self.x1 > xmax][0]
        ylo = self.y1[self.y1 < ymin][-1]
        yhi = self.y1[self.y1 > ymax][0]
        ds = self.ds.sel(x=slice(xlo,xhi), y=slice(ylo,yhi))
        ds = ds.rename_dims(x='west_east',
                            y='south_north',
                            hybrid='bottom_top')
        if inplace:
            self.ds = ds
        else:
            return ds

    def calculate(self,check=True):
        """Do all calculations to provide a consistent wrfinput dataset"""
        self.interpolate_na(inplace=True)
        self.derive_fields(check,inplace=True)
        self.calc_real(inplace=True)
        self.calc_perts(check,inplace=True)

    def interpolate_na(self,inplace=False):
        """Linearly interpolate between hybrid levels to remove any
        NaNs. If `inplace==False`, return a copy of the interpolated
        dataset.
        """
        if inplace:
            ds = self.ds
        else:
            ds = self.ds.copy()
        for varn in self.ds.variables:
            try:
                nnan = np.count_nonzero(~np.isfinite(ds[varn]))
            except TypeError:
                continue
            if nnan > 0:
                print(varn,nnan,'NaNs')
                ds[varn] = ds[varn].interpolate_na('bottom_top')
        if not inplace:
            return ds

    def derive_fields(self,check=True,inplace=False):
        """Calculate additional field quantities. If `inplace==False`,
        return a copy of the updated dataset.

        Calculated quantities include:
        - moist potential temperature (THM)
        - dry potential temperature (T)
        - water vapor mixing ratio (QVAPOR)
        - vertical velocity (W)
        - pressure at top of domain (PTOP)
        """
        # pull out working vars
        omega = self.ds['w']
        p_tot = self.ds['pres']
        Tair  = self.ds['t']
        q     = self.ds['q']
        gh    = self.ds['gh']

        if inplace:
            self.ds = self.ds.drop_vars(['w','pres','t','q','gh'])
            ds = self.ds
        else:
            ds = self.ds.copy()
            ds = ds.drop_vars(['w','pres','t','q','gh'])

        # water vapor mixing ratio, from definition of specified humidity
        qv = q / (1-q)
        ds['QVAPOR'] = qv

        # partial density of dry air (moisture reduces rho_d)
        rho_d = p_tot / (R_d * Tair) / (1 + R_v/R_d*qv)
        rho_m = rho_d * (1 + qv)

        # partial pressure of dry air
        p_dry = rho_d * R_d * Tair

        # perturbation _dry_ potential temperature [K]
        th_d = Tair * (p_0/p_tot)**(R_d/Cp_d)
        th_m = th_d * (1 + R_v/R_d*qv)
        ds['T'] = th_d - 300.0
        ds['THM'] = th_m - 300.0

        # total density of a parcel of air
        qt = ds['QVAPOR'] + ds['QCLOUD'] + ds['QRAIN']
        rho_t = rho_d * (1 + qt)

        # recover vertical velocity from hydrostatic equation
        ds['W'] = get_w_from_omega(omega, rho_m)

        # extrapolate pressure to top face
        p1 = p_tot.isel(bottom_top=-2).values
        p2 = p_tot.isel(bottom_top=-1).values
        ptop_faces = p2 + 0.5*(p2-p1)
        ptop = ptop_faces.max()
        ds['P_TOP'] = ptop

        # save for later
        self.p_dry = p_dry
        self.p_tot = p_tot
        self.rho_d = rho_d
        self.Tair = Tair
        self.gh = gh

        if check:
            assert np.allclose(getPgivenRTh(rho_d*th_m),
                               getPgivenRTh(rho_d*th_d,qv=qv))
            assert np.allclose(getPgivenRTh(rho_d*th_m),
                               p_tot)
            p_vap = rho_d*qv * R_v * Tair # vapor pressure
            assert np.allclose(p_tot, p_dry + p_vap)
            eps = R_d / R_v
            assert np.allclose(
                rho_m,
                p_tot/(R_d*Tair) * (1. - p_vap/p_tot*(1-eps)) # from sum of partial densities
            )

        if not inplace:
            return ds

    def calc_real(self,eta=hrrr_eta,inplace=False):
        """Calculate additional functions and constants like WRF
        real.exe. Hybrid coordinate functions `C1`, `C2`, `C3`, and `C4`
        -- at mass levels/cell centers ("half") and at staggered levels
        ("full") -- are all column functions that are known a priori and
        do not vary in time. This will initialize the base state like
        real.exe as well.
        """
        if inplace:
            ds = self.ds
        else:
            ds = self.ds.copy()

        real = RealInit(ds['HGT'], eta_stag=eta, ptop=ds['P_TOP'])
        self.real = real

        # hybrid coordinate functions
        ds['C1H'] = real.C1h
        ds['C2H'] = real.C2h
        ds['C3H'] = real.C3h
        ds['C4H'] = real.C4h
        ds['C1F'] = real.C1f
        ds['C2F'] = real.C2f
        ds['C3F'] = real.C3f
        ds['C4F'] = real.C4f

        # inverse difference in full eta levels
        ds['RDNW'] = real.rdnw

        if not inplace:
            return ds

    def calc_perts(self,check=True,inplace=False):
        """Calculate all perturbational (and remaining base state)
        quantities.
        """
        if inplace:
            ds = self.ds
        else:
            ds = self.ds.copy()

        ds['PB'] = self.real.pb
        ds['P'] = self.p_tot - ds['PB'] # perturbation

        ds['ALB'] = self.real.alb
        ds['AL'] = 1.0/self.rho_d - ds['ALB'] # perturbation

        # Set perturbation geopotential such that when destaggered we
        # recover the original geopotential heights. Note: ph[k=0] = 0
        ds['PHB'] = self.real.phb
        ds['PH'] = 0.0 * self.real.phb
        for k in range(1,self.ds.dims['bottom_top_stag']):
            ph_lo = ds['PH'].isel(bottom_top_stag=k-1)
            phb_lo = ds['PHB'].isel(bottom_top_stag=k-1)
            phb_hi = ds['PHB'].isel(bottom_top_stag=k)
            gh_avg = self.gh.isel(bottom_top=k-1)
            # (ph_lo+phb_lo + ph_hi+phb_hi) / (2*g) = gh_avg
            ph_hi = 2*CONST_GRAV*gh_avg - ph_lo - phb_hi - phb_lo
            ds['PH'].loc[dict(bottom_top_stag=k)] = ph_hi

        ds['MUB'] = self.real.mub
        ds['MU'] = xr.where(
            ds['C3H'] > 0,
            (self.p_dry - ds['C4H'] - ds['P_TOP']) / ds['C3H'] - ds['MUB'],
            ds['C4H'] + ds['P_TOP']
        )

        if check:
            zf = (ds['PH'] + ds['PHB']) / CONST_GRAV
            zh = 0.5*(get_hi_faces(zf) + get_lo_faces(zf))
            assert np.allclose(zh, self.gh)

        if not inplace:
            return ds

    def interp(self,name,xi,yi,dtype=float):
        """Linearly interpolate to points xi, yi"""
        da = self.ds[name].astype(dtype)
        xdim = [dim for dim in da.dims if dim.startswith('west_east')][0]
        ydim = [dim for dim in da.dims if dim.startswith('south_north')][0]
        try:
            zdim = [dim for dim in da.dims if dim.startswith('bottom_top')][0]
        except IndexError:
            zdim = None
        if zdim:
            dims = [xdim,ydim,zdim]
        else:
            dims = [xdim,ydim]

        print(f'Interpolating from {da.name} with dims {dims}')
        vals = da.transpose(*dims).values
        interpfun = RegularGridInterpolator((da.x,da.y),vals)
        interppts = np.stack([xi.ravel(), yi.ravel()], axis=-1)
        interpvals = interpfun(interppts)

        shape = list(xi.shape)
        if zdim:
             shape.append(da.sizes[zdim])
        interpvals = interpvals.reshape(shape)

        interpda = xr.DataArray(interpvals, dims=dims)
        return interpda.transpose(*dims[::-1]) # reverse dims to look like WRF

    def to_wrfinput(self,
                    start_date,
                    grid,
                    hrrr_xg, hrrr_yg,
                    hrrr_xg_u, hrrr_yg_u,
                    hrrr_xg_v, hrrr_yg_v,
                    dtype=float):
        """Create a new Dataset with HRRR fields interpolated to the
        input grid points
        """
        lat  , lon   = grid.calc_lat_lon()
        lat_u, lon_u = grid.calc_lat_lon('U')
        lat_v, lon_v = grid.calc_lat_lon('V')

        msf   = grid.calc_msf(lat)
        msf_u = grid.calc_msf(lat_u)
        msf_v = grid.calc_msf(lat_v)

        # create dataset with coordinates
        inp = xr.Dataset(
            coords={'XLAT' :(('south_north','west_east'),lat.astype(dtype)),
                    'XLONG':(('south_north','west_east'),lon.astype(dtype)),
                    'XLAT_U' :(('south_north','west_east_stag'),lat_u.astype(dtype)),
                    'XLONG_U':(('south_north','west_east_stag'),lon_u.astype(dtype)),
                    'XLAT_V' :(('south_north_stag','west_east'),lat_v.astype(dtype)),
                    'XLONG_V':(('south_north_stag','west_east'),lon_v.astype(dtype))}
        )
        inp['Times'] = bytes(start_date.strftime('%Y-%m-%d_%H:%M:%S'),'utf-8')

        # interpolate staggered velocity fields
        Ugrid = self.interp('U', hrrr_xg_u, hrrr_yg_u, dtype=dtype)
        Vgrid = self.interp('V', hrrr_xg_v, hrrr_yg_v, dtype=dtype)
        inp['U'] = Ugrid.rename(west_east='west_east_stag')
        inp['V'] = Vgrid.rename(south_north='south_north_stag')

        # interpolate fields that aren't staggered in x,y
        unstag_interp_vars = [
            'W',
            'ALB',
            'AL',
            'T',
            'PH',
            'PHB',
            'PB',
            'P',
            'SST',
            'LANDMASK',
            'MUB',
            'QVAPOR',
            'QCLOUD',
            'QRAIN',
        ]
        for varn in unstag_interp_vars:
            inp[varn] = self.interp(varn, hrrr_xg, hrrr_yg, dtype=dtype)

        # these are already on the output grid
        inp['MAPFAC_U'] = (('south_north', 'west_east_stag'), msf_u.astype(dtype))
        inp['MAPFAC_V'] = (('south_north_stag', 'west_east'), msf_v.astype(dtype))
        inp['MAPFAC_M'] = (('south_north', 'west_east'), msf.astype(dtype))

        # these only vary with height, no horizontal interp needed
        inp['C1H'] = self.ds['C1H'].astype(dtype)
        inp['C2H'] = self.ds['C2H'].astype(dtype)
        inp['RDNW'] = self.ds['RDNW'].astype(dtype)

        return inp
