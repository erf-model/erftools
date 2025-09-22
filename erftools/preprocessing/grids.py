import numpy as np
import cartopy.crs as ccrs
import pyproj


class GridLevel(object):
    """Simple data class containing 1-D staggered and destaggered grid
    coordinates to simplify access
    """
    def __init__(self, x0, y0, dx, dy, nx, ny):
        xlo = x0 - nx/2 * dx
        ylo = y0 - ny/2 * dy
        self.x = np.arange(nx+1) * dx + xlo
        self.y = np.arange(ny+1) * dy + ylo
        self.x_destag = (np.arange(nx)+0.5) * dx + xlo
        self.y_destag = (np.arange(ny)+0.5) * dy + ylo


class NestedGrids(object):
    """Container for nested grids with projected coordinates with the
    same map projection
    """
    def __init__(self, projection, dx, dy, nx, ny):
        self.proj = projection
        self.level = []

        if hasattr(dx, '__iter__'):
            assert len(dx) == len(nx) == len(ny)
            if dy is not None:
                assert len(dx) == len(dy)

            self.nlev = len(dx)
            self.dx = dx
            self.dy = dy if dy is not None else dx
            self.nx = nx
            self.ny = ny
        else:
            self.nlev = 1
            self.dx = [dx]
            self.dy = [dy] if dy is not None else [dx]
            self.nx = [nx]
            self.ny = [ny]

        self._setup_grid()

    def _setup_grid(self):
        self.x0, self.y0 = self.proj.transform_point(
                self.ref_lon, self.ref_lat, ccrs.Geodetic())

        for ilev in range(self.nlev):
            self.level.append(
                GridLevel(
                    self.x0,
                    self.y0,
                    self.dx[ilev],
                    self.dy[ilev],
                    self.nx[ilev],
                    self.ny[ilev],
                )
            )

        if self.nlev==1:
            self.x = self.level[0].x
            self.y = self.level[0].y
            self.x_destag = self.level[0].x_destag
            self.y_destag = self.level[0].y_destag

    def latlon(self,level=0,stagger=None):
        assert level < self.nlev
        if stagger is None and hasattr(self,'lat'):
            return self.lat[level], self.lon[level]
        elif stagger=='U' and hasattr(self,'lat_u'):
            return self.lat_u[level], self.lon_u[level]
        elif stagger=='V' and hasattr(self,'lat_v'):
            return self.lat_v[level], self.lon_v[level]
        else:
            lat, lon = self.calc_lat_lon(stagger=stagger)
            return lat[0], lon[0]

    def calc_lat_lon(self,stagger=None):
        """Calculate latitude and longitude at cell centers or u/v
        staggered locations (i.e., staggered in x/y)
        """
        lat_levels = []
        lon_levels = []
        for ilev in range(self.nlev):
            if stagger=='U':
                print(f'Calculating lat-lon staggered in x (lev={ilev})')
                xx,yy = np.meshgrid(self.level[ilev].x,
                                    self.level[ilev].y_destag)
            elif stagger=='V':
                print(f'Calculating lat-lon staggered in y (lev={ilev})')
                xx,yy = np.meshgrid(self.level[ilev].x_destag,
                                    self.level[ilev].y)
            else:
                print(f'Calculating unstaggered lat-lon (lev={ilev})')
                xx,yy = np.meshgrid(self.level[ilev].x_destag,
                                    self.level[ilev].y_destag)

            transformer = pyproj.Transformer.from_proj(
                self.proj,
                "EPSG:4326",  # WGS84 geographic coordinates (equivalent to ccrs.Geodetic())
                always_xy=True
            )
            lon, lat = transformer.transform(xx, yy)

            lat_levels.append(lat)
            lon_levels.append(lon)

        if stagger is None:
            self.lat = lat_levels
            self.lon = lon_levels
        elif stagger =='U':
            self.lat_u = lat_levels
            self.lon_u = lon_levels
        elif stagger =='V':
            self.lat_v = lat_levels
            self.lon_v = lon_levels
        return lat_levels, lon_levels



class LambertConformalGrid(NestedGrids):
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

        proj = ccrs.LambertConformal(
            central_longitude=stand_lon,
            central_latitude=ref_lat,
            standard_parallels=standard_parallels,
            globe=ccrs.Globe(
                ellipse="sphere",
                semimajor_axis=earth_radius,
                semiminor_axis=earth_radius,
            ),
        )
        super().__init__(proj,dx,dy,nx,ny)

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
