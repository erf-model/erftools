import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj


class GridLevel(object):
    """Simple data class containing 1-D staggered and destaggered grid
    coordinates to simplify access
    """
    def __init__(self, dx, dy, nx, ny, center=None, ll_corner=None):
        assert (center is not None) or (ll_corner is not None)
        if center:
            x0, y0 = center
            xlo = x0 - nx/2 * dx
            ylo = y0 - ny/2 * dy
        else:
            xlo, ylo = ll_corner
        self.x = np.arange(nx+1) * dx + xlo
        self.y = np.arange(ny+1) * dy + ylo
        self.x_destag = (np.arange(nx)+0.5) * dx + xlo
        self.y_destag = (np.arange(ny)+0.5) * dy + ylo


class NestedGrids(object):
    """Container for nested grids with projected coordinates with the
    same map projection
    """
    def __init__(self, projection, dx, dy, nx, ny, ll_ij=None, ll_xy=None):
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
            self.ll_ij = ll_ij
            self.ll_xy = ll_xy
        else:
            self.nlev = 1
            self.dx = [dx]
            self.dy = [dy] if dy is not None else [dx]
            self.nx = [nx]
            self.ny = [ny]
            self.ll_ij = None
            self.ll_xy = None

        self._setup_grid()

    def _setup_grid(self):
        self.x0, self.y0 = self.proj.transform_point(
                self.ref_lon, self.ref_lat, ccrs.Geodetic())

        concentric = (self.ll_ij is None) and (self.ll_xy is None)
        if not concentric:
            if self.ll_xy:
                assert len(self.ll_xy) >= self.nlev-1
                assert all([len(xy)==2 for xy in self.ll_xy])
            elif self.ll_ij:
                assert len(self.ll_ij) >= self.nlev-1
                assert all([len(ij)==2 for ij in self.ll_ij])


        for ilev in range(self.nlev):
            if (ilev == 0) or concentric:
                anchor_pt = {'center': (self.x0, self.y0)}
            else:
                if self.ll_xy:
                    xll = self.ll_xy[ilev-1][0]
                    yll = self.ll_xy[ilev-1][0]
                elif self.ll_ij:
                    ioff,joff = self.ll_ij[ilev-1]
                    xll = self.level[ilev-1].x[0] + ioff * self.dx[ilev-1]
                    yll = self.level[ilev-1].y[0] + joff * self.dy[ilev-1]
                anchor_pt = {'ll_corner':(xll, yll)}

            self.level.append(
                GridLevel(
                    self.dx[ilev],
                    self.dy[ilev],
                    self.nx[ilev],
                    self.ny[ilev],
                    **anchor_pt
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

    def find_ij_from_latlon(self,lat,lon,level=None,stagger=None):
        """Find i,j indices (corresponding to x,y) for given lat,lon

        All levels are searched, from finest to coarsest, if `level` is
        not specified.
        """
        levels = np.arange(self.nlev) if level is None else [level]
        for ilev in levels[::-1]:
            if stagger=='U':
                x1 = self.level[ilev].x
                y1 = self.level[ilev].y_destag
            elif stagger=='V':
                x1 = self.level[ilev].x_destag
                y1 = self.level[ilev].y
            else:
                x1 = self.level[ilev].x_destag
                y1 = self.level[ilev].y_destag

            transformer = pyproj.Transformer.from_proj(
                "EPSG:4326",  # WGS84 geographic coordinates (equivalent to ccrs.Geodetic())
                self.proj,
                always_xy=True
            )
            x, y = transformer.transform(lon, lat)

            if ((x >= x1[0]) and (x <= x1[-1]) and
                (y >= y1[0]) and (y <= y1[-1])
            ):
                i = int((x - x1[0]) / self.dx[ilev])
                j = int((y - y1[0]) / self.dy[ilev])
                return ilev, i, j

    def plot_grids(self,projection=None,fig=None,ax=None,**kwargs):
        if ax is None:
            if projection is None:
                projection = self.proj
            fig,ax = plt.subplots(subplot_kw=dict(projection=projection))

        # plot map features
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle='-')
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        # plot all domain boundaries
        for ilev, lev in enumerate(self.level):
            label = f'level {ilev}'
            plot_boundaries(lev.x, lev.y, self.proj, label=label, **kwargs)

        # default plot extents
        xmin = self.level[0].x[0]
        xmax = self.level[0].x[-1]
        ymin = self.level[0].y[0]
        ymax = self.level[0].y[-1]
        Lx = xmax - xmin
        Ly = ymax - ymin
        ax.set_extent([xmin - 0.5*Lx,
                       xmax + 0.5*Lx,
                       ymin - 0.5*Ly,
                       ymax + 0.5*Ly],
                      crs=self.proj)

        ax.legend(loc='best', frameon=False)

        return fig, ax


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
                 ll_ij=None, ll_xy=None,
                 earth_radius=6370000.):
        """Initialize projection on a spherical datum with grid centered
        at (ref_lat, ref_lon).

        To specify nested domains, dx, dy, nx, and ny should be
        specified as lists with one value per domain level. In addition,
        either ll_ij or ll_xy are needed if the nests are not
        concentric.

        Parameters
        ----------
        ref_lat, ref_lon: float
            Central latitude and longitude in degrees
        truelat1, truelat2: float
            Standard parallel(s) at which the map scale is unity
        stand_lon: float, optional
            Central meridian
        dx, dy : float or array-like
            Grid spacing in west-east, south-north directions
        nx, ny : int or array-like
            Number of cells in the west-east, south-north directions
        ll_ij : list of pairs, optional
            Parent indices of the lower-left corner of each nested grid
            level; this takes precedence over ll_xy
        ll_xy : list of pairs, optional
            Coordinates of the lower-left corner of each nested grid
            level; may be specified instead of ll_ij
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
        super().__init__(proj,
                         dx,dy,nx,ny,
                         ll_ij, ll_xy)

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


def plot_boundaries(x,y, proj, label='', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    color = None # default cycle
    if 'color' in kwargs.keys():
        color = kwargs.pop('color')
    elif 'c' in kwargs.keys():
        color = kwargs.pop('c')

    line, = ax.plot(x, y[0]*np.ones_like(x),
            color=color,
            transform=proj, label=label, **kwargs)
    ax.plot(x, y[-1]*np.ones_like(x),
            color=line.get_color(),
            transform=proj, **kwargs)
    ax.plot(x[0]*np.ones_like(y), y,
            color=line.get_color(),
            transform=proj, **kwargs)
    ax.plot(x[-1]*np.ones_like(y), y,
            color=line.get_color(),
            transform=proj, **kwargs)
