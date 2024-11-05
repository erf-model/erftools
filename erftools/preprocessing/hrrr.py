import numpy as np
import xarray as xr

from herbie import Herbie
import cartopy.crs as ccrs

from ..constants import R_d, R_v, Cp_d, Cp_v, CONST_GRAV, p_0
from ..EOS import getPgivenRTh, getThgivenRandT, getThgivenPandT
from .utils import get_w_from_omega

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

    def __init__(self,datetime,get_surface=False,varlist=varlist):
        """Download data from native levels, see
        https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
        for data inventory
        """
        self.H = Herbie(datetime, model='hrrr', product='nat')
        self.H.download(verbose=True)
        self._combine_data(varlist,get_surface)
        self._setup_grid()

    def _combine_data(self,varlist,get_surface):
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
        if get_surface:
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

    def calculate(self,check=True,inplace=False):
        """Calculate additional field quantities to provide a consistent
        wrfinput_d01 dataset. If `inplace==False`, return a copy of the
        updated dataset.

        Calculated quantities include:
        - moist potential temperature (THM)
        - dry potential temperature (T)
        - water vapor mixing ratio (QVAPOR)
        - vertical velocity (W)
        """
        if inplace:
            ds = self.ds
        else:
            ds = self.ds.copy()

        # pull out working vars
        omega = ds['w'];    ds = ds.drop_vars('w')
        p_tot = ds['pres']; ds = ds.drop_vars('pres')
        Tair  = ds['t'];    ds = ds.drop_vars('t')
        q     = ds['q'];    ds = ds.drop_vars('q')

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

        if check:
            assert np.allclose(getPgivenRTh(rho_d*th_m),
                               getPgivenRTh(rho_d*th_d,qv=qv))
            assert np.allclose(getPgivenRTh(rho_d*th_m),
                               p_tot)
            e = rho_d*qv * R_v * Tair # vapor pressure
            assert np.allclose(p_tot, p_dry + e)
            eps = R_d / R_v
            assert np.allclose(
                rho_m,
                p_t/(R_d*Tair) * (1. - e/p_t*(1-eps)) # from sum of partial densities
            )

        if not inplace:
            return ds
