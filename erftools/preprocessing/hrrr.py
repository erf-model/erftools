import numpy as np
import xarray as xr

from herbie import Herbie
import cartopy.crs as ccrs


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
        self.ds = self.H.xarray(f':(?:{varstr}):\d+ hybrid') # get all levels
        if isinstance(self.ds, list):
            self.ds = xr.merge(self.ds)
        self.ds.rename_vars({
            'u': 'U',
            'v': 'V',
            'clwmr': 'QCLOUD',
            'rwmr' : 'QRAIN',
        })
        if get_surface:
            surf = self.H.xarray(':surface:anl')
            self.ds['LANDMASK'] = surf['lsm']
            self.ds['SST']      = surf['t']
            self.ds['HGT']      = surf['orog']
            self.ds['PSFC']     = surf['sp']

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

    def inventory(self):
        return self.H.inventory()
