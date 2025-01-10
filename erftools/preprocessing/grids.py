import numpy as np
import cartopy.crs as ccrs


class LambertConformalGrid(object):
    """Given WRF projection parameters, setup a projection and calculate
    map scale factors
    """
    def __init__(self,
                 ref_lat, ref_lon,
                 truelat1, truelat2=None,
                 stand_lon=None,
                 dx=None, dy=None,
                 nx=None, ny=None,
                 earth_radius=6370000.):
        """Initialize projection on a spherical datum with grid centered
        at (ref_lat, ref_lon).

        Parameters
        ----------
        ref_lat, ref_lon: float
            Central latitude and longitude in degrees
        truelat1, truelat2: float
            Standard parallel(s) at which the map scale is unity
        stand_lon: float, optional
            Central meridian
        dx, dy : float
            Grid spacing in west-east, south-north directions
        nx, ny : int
            Number of cells in the west-east, south-north directions
        earth_radius: float
            Radius of the earth approximated as a sphere
        """
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        if (truelat2 is None) or (truelat2==truelat1):
            truelat2 = None
            standard_parallels = [truelat1]
        else:
            standard_parallels = [truelat1,truelat2]
        self.truelat1 = truelat1
        self.truelat2 = truelat2
        if stand_lon is None:
            stand_lon = ref_lon
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.proj = ccrs.LambertConformal(
            central_longitude=stand_lon,
            central_latitude=ref_lat,
            standard_parallels=standard_parallels,
            globe=ccrs.Globe(
                ellipse="sphere",
                semimajor_axis=earth_radius,
                semiminor_axis=earth_radius,
            ),
        )
        if self.dx and self.nx and self.ny:
            self.setup_grid()

    def setup_grid(self):
        assert self.dx is not None
        if self.dy is None:
            self.dy = self.dx
        assert (self.nx is not None) and (self.ny is not None)

        self.x0, self.y0 = self.proj.transform_point(
                self.ref_lon, self.ref_lat, ccrs.Geodetic())

        xlo = self.x0 - (self.nx)/2*self.dx
        ylo = self.y0 - (self.ny)/2*self.dy
        self.x = np.arange(self.nx+1)*self.dx + xlo
        self.y = np.arange(self.ny+1)*self.dy + ylo
        self.x_destag = (np.arange(self.nx)+0.5)*self.dx + xlo
        self.y_destag = (np.arange(self.ny)+0.5)*self.dy + ylo

    def calc_lat_lon(self,stagger=None):
        if stagger is None and hasattr(self,'lat'):
            return self.lat, self.lon
        elif stagger=='U' and hasattr(self,'lat_u'):
            return self.lat_u, self.lon_u
        elif stagger=='V' and hasattr(self,'lat_v'):
            return self.lat_v, self.lon_v

        if not hasattr(self,'x'):
            self.setup_grid()

        if stagger=='U':
            print('Calculating lat-lon staggered in x')
            xx,yy = np.meshgrid(self.x, self.y_destag)
        elif stagger=='V':
            print('Calculating lat-lon staggered in y')
            xx,yy = np.meshgrid(self.x_destag, self.y)
        else:
            print('Calculating unstaggered lat-lon')
            xx,yy = np.meshgrid(self.x_destag, self.y_destag)
        lonlat = ccrs.Geodetic().transform_points(self.proj, xx.ravel(), yy.ravel())
        lon = lonlat[:,0].reshape(xx.shape)
        lat = lonlat[:,1].reshape(xx.shape)

        if stagger is None:
            self.lat = lat
            self.lon = lon
        elif stagger =='U':
            self.lat_u = lat
            self.lon_u = lon
        elif stagger =='V':
            self.lat_v = lat
            self.lon_v = lon
        return lat,lon

    def calc_msf(self,lat):
        """From WRF WPS process_tile_module.F"""
        if self.truelat2 is None:
            colat0 = np.radians(90.0 - self.truelat1)
            colat  = np.radians(90.0 - lat)
            return np.sin(colat0)/np.sin(colat) \
                    * (np.tan(colat/2.0)/np.tan(colat0/2.0))**np.cos(colat0)
        else:
            colat1 = np.radians(90.0 - self.truelat1)
            colat2 = np.radians(90.0 - self.truelat2)
            n = (np.log(np.sin(colat1))     - np.log(np.sin(colat2))) \
              / (np.log(np.tan(colat1/2.0)) - np.log(np.tan(colat2/2.0)))
            colat  = np.radians(90.0 - lat)
            return np.sin(colat2)/np.sin(colat) \
                    * (np.tan(colat/2.0)/np.tan(colat2/2.0))**n
