import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

import cartopy.crs as ccrs
from herbie import Herbie

from ..constants import R_d, R_v, Cp_d, Cp_v, CONST_GRAV, p_0
from ..EOS import getPgivenRTh, getThgivenRandT, getThgivenPandT
from ..utils import (get_hi_faces, get_lo_faces,
                     get_w_from_omega, get_mass_weighted)
from ..wrf.real import RealInit


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
gribvars = [
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
surfgribvars = [
    'LAND',
    'HGT',
    'TMP',
    'PRES'
]


class NativeHRRR(object):
    """Get HRRR analysis on native levels and calculate fields
    consistent with WRF
    """

    def __init__(self,datetime,varlist=gribvars,verbose=False):
        """Download data from native levels, see
        https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
        for data inventory
        """
        self.datetime = datetime
        self.verbose = verbose
        self.H = Herbie(datetime, model='hrrr', product='nat')
        self.H.download(verbose=True)
        self._combine_data(varlist)
        self._setup_hrrr_grid()

    def _combine_data(self,varlist):
        """Combine data from hybrid levels with surface data; rename
        like WRF
        """
        # This may through a "WrongLengthError"...
        #varstr = '|'.join(varlist)
        #ds = self.H.xarray(f':(?:{varstr}):\d+ hybrid') # get all levels
        #if isinstance(ds, list):
        #    ds = xr.merge(ds)

        #print('Retrieving hybrid-level variables:',end='')
        #dslist = []
        #for varn in gribvars:
        #    print(f' {varn}',end='',flush=True)
        #    dslist.append(self.H.xarray(f':{varn}:\d+ hybrid'))
        #print('')
        #ds = xr.merge(dslist)
        gribpath = self.H.get_localFilePath()
        ds = xr.open_dataset(gribpath,engine='cfgrib',
                             filter_by_keys={'typeOfLevel':'hybrid'})
        ds = ds.rename_vars({
            'u': 'U',
            'v': 'V',
            'clwmr': 'QCLOUD',
            'rwmr' : 'QRAIN',
        })

        #print('Retrieving surface variables:',end='')
        #dslist = []
        #for varn in surfgribvars:
        #    print(f' {varn}',end='',flush=True)
        #    dslist.append(self.H.xarray(f':{varn}:surface:anl'))
        #print('')
        #surf = xr.merge(dslist)
        surf = xr.open_dataset(gribpath,engine='cfgrib',
                               filter_by_keys={'stepType':'instant',
                                               'typeOfLevel':'surface'})
        ds['LANDMASK'] = surf['lsm']
        ds['SST']      = surf['t']
        ds['HGT']      = surf['orog']
        ds['PSFC']     = surf['sp']

        self.ds = ds

    def _setup_hrrr_grid(self):
        """The resulting grid is Cartesian with coordinates in the
        Lambert Conformal CRS defined by x1 and y1.
        """
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

        self.dx = np.mean(np.diff(self.x1))
        self.dy = np.mean(np.diff(self.y1))

        # create dimension coordinates
        self.ds = self.ds.assign_coords(x=self.x1, y=self.y1)

        # initialize output grids
        self.xg = None
        self.yg = None
        self.xg_u = None
        self.yg_u = None
        self.xg_v = None
        self.yg_v = None

    def __getitem__(self,key):
        return self.ds[key]

    def inventory(self):
        return self.H.inventory()

    def clip(self,xmin,xmax,ymin,ymax,zmax=None,inplace=False):
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
        if zmax is not None:
            gh_avg = ds['gh'].mean(['south_north','west_east']).values
            kmax = np.where(gh_avg > zmax)[0][0]
            ds = ds.isel(bottom_top=slice(0,kmax))
            print('Setting nz=',kmax)
        if inplace:
            self.ds = ds
        else:
            return ds

    def _compare_arrays(self,arr1,arr2,checktype,desc=None):
        if isinstance(arr1, xr.DataArray):
            arr1 = arr1.values
        if isinstance(arr2, xr.DataArray):
            arr2 = arr2.values
        if checktype == 'assert':
            assert np.allclose(arr1, arr2)
        elif checktype == 'warn':
            if not np.allclose(arr1, arr2):
                abserr = np.abs(arr2 - arr1)
                relerr = np.abs((arr2 - arr1) / arr1)
                descstr = '' if desc is None else desc
                print(f'\x1b[31mWARNING:\x1b[0m {descstr} abserr={np.max(abserr):g} relerr={np.max(relerr)}')
        else:
            print('Skipping check, unknown type=',checktype)

    def calculate(self,check='assert',zeroW=False):
        """Do all calculations to provide a consistent wrf-like dataset

        `check` can be "warn" or "assert"
        """
        self.interpolate_na(inplace=True)

        self.derive_fields(check,inplace=True)

        # handle clipping
        eta = hrrr_eta  # this is staggered
        nz = self.ds.sizes['bottom_top'] # unstaggered
        nzmax = len(eta) - 1
        if nz < nzmax:
            clip = nzmax - nz
            eta = eta[:-clip]

        # calculate base state following WRF real.exe
        self.calc_real(inplace=True,eta=eta)

        self.calc_perts(check,inplace=True)

        rhow = self.rho_d.isel(bottom_top=-1,drop=True) \
             * self.ds['W'].isel(bottom_top_stag=-1,drop=True)
        zflux = rhow.sum(['south_north','west_east'])*self.dx*self.dy
        print(f'Note: mass flux through top faces is {zflux.item():g}')

        if zeroW:
            print('Setting vertical velocity to zero')
            self.ds['W'] *= 0.

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
                if self.verbose:
                    print(varn,nnan,'NaNs')
                ds[varn] = ds[varn].interpolate_na('bottom_top')
        if not inplace:
            return ds

    def derive_fields(self,check='assert',inplace=False):
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

        # water vapor mixing ratio, from definition of specific humidity
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

        # save for reference
        self.p_dry = p_dry
        self.p_tot = p_tot
        self.rho_d = rho_d
        self.rho_m = rho_m
        self.Tair = Tair
        self.gh = gh

        if check:
            # from definition of moist potential temperature
            self._compare_arrays(getPgivenRTh(rho_d*th_m),
                                 getPgivenRTh(rho_d*th_d,qv=qv),
                                 check,'moist EOS')

            # from EOS, partial pressures
            self._compare_arrays(getPgivenRTh(rho_d*th_m), p_tot,
                                 check, 'EOS p_tot')

            # from derivation of dry air density
            p_vap = rho_d*qv * R_v * Tair # vapor pressure
            self._compare_arrays(p_tot, p_dry + p_vap,
                                 check, 'total pressure')

            # from sum of partial densities
            eps = R_d / R_v
            self._compare_arrays(
                rho_m,
                p_tot/(R_d*Tair) * (1. - p_vap/p_tot*(1-eps)),
                check, 'sum of partial densities')

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

    def calc_perts(self,check='assert',inplace=False):
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

        self.zstag = (ds['PHB'] + ds['PH']) / CONST_GRAV
        zl = get_lo_faces(self.zstag)
        zh = get_hi_faces(self.zstag)
        self.zcc = 0.5*(zl + zh)
        self.dz = zh - zl
        if check:
            self._compare_arrays(self.zcc, self.gh, check, 'geopotential height')

        ds['MUB'] = self.real.mub
        ds['MU'] = (self.p_dry.isel(bottom_top=0)
                    - ds['C4F'].isel(bottom_top_stag=0)
                    - ds['P_TOP']) / ds['C3F'].isel(bottom_top_stag=0) - ds['MUB']

        if not inplace:
            return ds

    def interpxy(self,name,xi,yi,dtype=float):
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

        if self.verbose:
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

    def set_output_grid(self,grid):
        """Transform the output surface grid points from the source grid
        (a LambertConformalGrid object) to the HRRR projection
        """
        self.grid = grid

        xg,   yg   = np.meshgrid(grid.x_destag, grid.y_destag, indexing='ij')
        xg_u, yg_u = np.meshgrid(grid.x       , grid.y_destag, indexing='ij')
        xg_v, yg_v = np.meshgrid(grid.x_destag, grid.y       , indexing='ij')

        pts = hrrr_projection.transform_points(grid.proj, xg, yg)
        self.xg  = pts[:,:,0]
        self.yg  = pts[:,:,1]

        pts_u = hrrr_projection.transform_points(grid.proj, xg_u, yg_u)
        self.xg_u  = pts_u[:,:,0]
        self.yg_u  = pts_u[:,:,1]

        pts_v = hrrr_projection.transform_points(grid.proj, xg_v, yg_v)
        self.xg_v  = pts_v[:,:,0]
        self.yg_v  = pts_v[:,:,1]


    def to_wrfinput(self,dtype=float):
        """Create a new Dataset with HRRR fields interpolated to the
        input grid points
        """
        lat  , lon   = self.grid.calc_lat_lon()
        lat_u, lon_u = self.grid.calc_lat_lon('U')
        lat_v, lon_v = self.grid.calc_lat_lon('V')

        msf   = self.grid.calc_msf(lat)
        msf_u = self.grid.calc_msf(lat_u)
        msf_v = self.grid.calc_msf(lat_v)

        # create dataset with coordinates
        inp = xr.Dataset({
            'XLAT' :   (('south_north','west_east'), lat.astype(dtype)),
            'XLONG':   (('south_north','west_east'), lon.astype(dtype)),
            'XLAT_U' : (('south_north','west_east_stag'), lat_u.astype(dtype)),
            'XLONG_U': (('south_north','west_east_stag'), lon_u.astype(dtype)),
            'XLAT_V' : (('south_north_stag','west_east'), lat_v.astype(dtype)),
            'XLONG_V': (('south_north_stag','west_east'), lon_v.astype(dtype)),
        })
        inp['Times'] = bytes(self.datetime.strftime('%Y-%m-%d_%H:%M:%S'),'utf-8')

        # interpolate staggered velocity fields
        Ugrid = self.interpxy('U', self.xg_u, self.yg_u, dtype=dtype)
        Vgrid = self.interpxy('V', self.xg_v, self.yg_v, dtype=dtype)
        inp['U'] = Ugrid.rename(west_east='west_east_stag')
        inp['V'] = Vgrid.rename(south_north='south_north_stag')

        # interpolate fields that aren't staggered in x,y
        unstag_interp_vars = [
            'W',
            'ALB',
            'AL',
            'T',
            'THM',
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
            inp[varn] = self.interpxy(varn, self.xg, self.yg, dtype=dtype)

        # these are already on the output grid
        # note: MAPFAC_U == MAPFAC_UX == MAPFAC_UY, etc
        inp['MAPFAC_U'] = (('south_north', 'west_east_stag'), msf_u.astype(dtype))
        inp['MAPFAC_V'] = (('south_north_stag', 'west_east'), msf_v.astype(dtype))
        inp['MAPFAC_M'] = (('south_north', 'west_east'), msf.astype(dtype))

        # these only vary with height, no horizontal interp needed
        inp['C1H'] = self.ds['C1H'].astype(dtype)
        inp['C2H'] = self.ds['C2H'].astype(dtype)
        inp['RDNW'] = self.ds['RDNW'].astype(dtype)

        # add time dimension
        inp = inp.expand_dims('Time',axis=0)

        # make lat,lon data vars into coordinates, remove the rest
        inp = inp.drop_vars(inp.coords)
        inp = inp.assign_coords({'XLAT'   : inp['XLAT'],
                                 'XLONG'  : inp['XLONG'],
                                 'XLAT_U' : inp['XLAT_U'],
                                 'XLONG_U': inp['XLONG_U'],
                                 'XLAT_V' : inp['XLAT_V'],
                                 'XLONG_V': inp['XLONG_V']})

        return inp

    def to_wrfbdy(self,bdy_width,dtype=float):
        """Create a new Dataset with HRRR fields interpolated to the
        input grid points on the specified boundary

        This is a stripped down version of to_wrfinput but with mass-
        weighting for the field variables
        """
        ib = {
            'lo': slice(0,bdy_width),
            'hi': slice(-bdy_width,None),
        }
        bndry = {
            # dims: west_east*, south_north*
            'BXS': (ib['lo'], slice(None)),
            'BXE': (ib['hi'], slice(None)),
            'BYS': (slice(None), ib['lo']),
            'BYE': (slice(None), ib['hi']),
        }
        width_dim = {
            'BXS': 'west_east',
            'BXE': 'west_east',
            'BYS': 'south_north',
            'BYE': 'south_north',
        }

        lat_u, _ = self.grid.calc_lat_lon('U')
        lat_v, _ = self.grid.calc_lat_lon('V')

        # create data subsets for each boundary
        bdy = {}
        for bname,idxs in bndry.items():
            ds = xr.Dataset()

            # interpolate staggered velocity fields
            Ugrid = self.interpxy('U', self.xg_u[*idxs], self.yg_u[*idxs], dtype=dtype)
            Vgrid = self.interpxy('V', self.xg_v[*idxs], self.yg_v[*idxs], dtype=dtype)
            ds['U'] = Ugrid.rename(west_east='west_east_stag')
            ds['V'] = Vgrid.rename(south_north='south_north_stag')

            # interpolate fields (a subset of wrfinput) that aren't staggered
            # in x,y
            unstag_interp_vars = [
                'W',
                'PH',
                'T',
                'MU',
                'MUB', # needed to couple the other quantities
                'QVAPOR',
                'QCLOUD',
                'QRAIN',
            ]
            for varn in unstag_interp_vars:
                ds[varn] = self.interpxy(varn, self.xg[*idxs], self.yg[*idxs], dtype=dtype)

            # setup map scale factors
            sn_ew_idxs = idxs[::-1]
            msf_u = self.grid.calc_msf(lat_u[*sn_ew_idxs])
            msf_v = self.grid.calc_msf(lat_v[*sn_ew_idxs])
            ds['MAPFAC_U'] = (('south_north', 'west_east_stag'), msf_u.astype(dtype))
            ds['MAPFAC_V'] = (('south_north_stag', 'west_east'), msf_v.astype(dtype))

            # these column functions are needed to couple all quantities (except MU)
            ds['C1H'] = self.ds['C1H'].astype(dtype)
            ds['C2H'] = self.ds['C2H'].astype(dtype)
            ds['C1F'] = self.ds['C1F'].astype(dtype)
            ds['C2F'] = self.ds['C2F'].astype(dtype)

            # calculate coupled fields
            vars_to_couple = [varn for varn in ds.data_vars
                              if not varn.startswith('MU') and
                                 not varn.startswith('MAPFAC') and
                                 not varn.startswith('C1') and
                                 not varn.startswith('C2')]
            for varn in vars_to_couple:
                # this dimension may or may not be staggered
                bw_dim = [dim for dim in ds[varn].dims
                          if dim.startswith(width_dim[bname])][0]
                if self.verbose:
                    print(f'Coupling {varn} on {bname} (bdy_width dim: {bw_dim})')
                for w in range(bdy_width):
                    coupled = get_mass_weighted(varn, ds, **{bw_dim:w})
                    ds[varn].loc[dict({bw_dim:w})] = coupled

            bdy[bname] = ds

        # create combined dataset
        ds = xr.Dataset({
            'Times': ('Time',
                      [bytes(self.datetime.strftime('%Y-%m-%d_%H:%M:%S'),'utf-8')])
        })
        output_vars = ['U','V','PH','T','MU','QVAPOR','QCLOUD','QRAIN']
        for varn in output_vars:
            for bname,idxs in bndry.items():
                # get all the buffer region planes
                if varn=='MU':
                    lat_dim = we_dim if width_dim[bname]==sn_dim else sn_dim
                    mu = bdy[bname]['MU'].isel({'west_east': idxs[0],
                                                'south_north': idxs[1]})
                    mu = mu.rename({width_dim[bname]:'bdy_width'})
                    mu = mu.transpose('bdy_width',lat_dim)
                    ds[f'MU_{bname}'] = mu.expand_dims('Time',axis=0)
                else:
                    # these dimensions may or may not be staggered
                    we_dim = [dim for dim in bdy[bname][varn].dims
                              if dim.startswith('west_east')][0]
                    sn_dim = [dim for dim in bdy[bname][varn].dims
                              if dim.startswith('south_north')][0]
                    bt_dim = [dim for dim in bdy[bname][varn].dims
                              if dim.startswith('bottom_top')][0]
                    # boundary-normal dimension
                    bw_dim = [dim for dim in bdy[bname][varn].dims
                              if dim.startswith(width_dim[bname])][0]
                    # lateral dimension
                    lat_dim = we_dim if bw_dim==sn_dim else sn_dim
                    # subset the field
                    fld = bdy[bname][varn].isel({we_dim: idxs[0],
                                                 sn_dim: idxs[1]})
                    fld = fld.rename({bw_dim:'bdy_width'})
                    fld = fld.transpose('bdy_width',bt_dim,lat_dim)
                    ds[f'{varn}_{bname}'] = fld.expand_dims('Time',axis=0)
        return ds
