import numpy as np
import pandas as pd
import xarray as xr
import f90nml
import cartopy.crs as ccrs

from .wrf.namelist import (TimeControl, Domains, Physics, Dynamics,
                           BoundaryControl)
from .wrf.landuse import LandUseTable
from .inputs import ERFInputFile

class WRFInputDeck(object):
    """Class to parse inputs from WRF and convert to inputs for ERF
    WRF inputs include:
    * namelist.input
    * wrfinput_d01[, wrfinput_d02, ...]
    """

    def __init__(self,nmlpath):
        # scrape WRF namelists
        with open(nmlpath,'r') as f:
            self.nml = f90nml.read(f)
        self.time_control = TimeControl(self.nml['time_control'])
        self.domains = Domains(self.nml['domains'])
        self.physics = Physics(self.nml['physics'])
        self.dynamics = Dynamics(self.nml['dynamics'])
        self.bdy_control = BoundaryControl(self.nml['bdy_control'])
        # calculate ERF equivalents
        self.erf_input = ERFInputFile()
        self.generate_inputs()

    def __str__(self):
        s = str(self.time_control) + '\n'
        s+= str(self.domains) + '\n'
        s+= str(self.physics) + '\n'
        s+= str(self.dynamics) + '\n'
        return s

    def generate_inputs(self):
        """Scrape inputs for ERF from a WRF namelist.input file

        Note that the namelist does _not_ provide the following input
        information:
        * WRF geopotential height levels
        * surface temperature map
        * surface roughness map
        To get these values, call `process_initial_conditions()` to extract
        them from `wrfinput_d01` (output from real.exe).
        """
        tdelta = self.time_control.end_datetime - self.time_control.start_datetime
        self.erf_input['stop_time'] = tdelta.total_seconds()

        # note: starting index assumed == 1
        # note: ending index --> number of _staggered_ pts
        # TODO: add vertical stretching
        n_cell = np.array([self.domains.e_we[0], self.domains.e_sn[0], self.domains.e_vert[0]]) - 1
        self.erf_input['geometry.prob_extent'] = n_cell * np.array([self.domains.dx[0], self.domains.dy[0], np.nan])
        self.erf_input['geometry.is_periodic'] = [
                self.bdy_control.periodic_x,
                self.bdy_control.periodic_y,
                False]
        if self.domains.ztop is None:
            ztop = 287.0 * 300.0 / 9.81 * np.log(1e5/self.domains.p_top_requested)
            print('NOTE: Estimated domain ztop from domains.p_top_requested',
                  f'= {self.domains.p_top_requested:g}')
        else:
            ztop = self.domains.ztop
        self.erf_input['geometry.prob_extent'][2] = ztop
        self.erf_input['amr.n_cell'] = n_cell

        # TODO: verify that refined regions will take finer time steps
        dt = np.array(self.domains.parent_time_step_ratio) * self.domains.time_step
        self.erf_input['erf.fixed_dt'] = dt[0]

        # refinements
        self.erf_input['amr.max_level'] = self.domains.max_dom - 1 # zero-based indexing
        grid_ratio = self.domains.parent_grid_ratio[-1] # TODO: assume all nests have same ratio
        self.erf_input['amr.ref_ratio_vect'] = [grid_ratio, grid_ratio, 1]
        if self.domains.max_dom > 1:
            refine_names = ' '.join([f'box{idom:d}' for idom in range(1,self.domains.max_dom)])
            self.erf_input['amr.refinement_indicators'] = refine_names
            for idom in range(1,self.domains.max_dom):
                dx = self.domains.dx
                dy = self.domains.dy
                imax = self.domains.e_we
                jmax = self.domains.e_sn
                parent_ds  = np.array([  dx[idom-1],   dy[idom-1]], dtype=float)
                child_ds   = np.array([  dx[idom  ],   dy[idom  ]], dtype=float)
                parent_ext = np.array([imax[idom-1], jmax[idom-1]]) * parent_ds
                child_ext  = np.array([imax[idom  ], jmax[idom  ]]) * child_ds
                lo_idx = np.array([
                    self.domains.i_parent_start[idom] - 1,
                    self.domains.j_parent_start[idom] - 1])
                in_box_lo = lo_idx * parent_ds
                in_box_hi = in_box_lo + child_ext
                assert (in_box_hi[0] <= parent_ext[0])
                assert (in_box_hi[1] <= parent_ext[1])
                self.erf_input[f'amr.box{idom:d}.in_box_lo'] = in_box_lo
                self.erf_input[f'amr.box{idom:d}.in_box_hi'] = in_box_hi

        restart_period = self.time_control.restart_interval * 60.0 # [s]
        self.erf_input['amr.check_int'] = int(restart_period / dt[0])

        sfclayscheme = self.physics.sf_sfclay_physics[0]
        if sfclayscheme == 'none':
            self.erf_input['zlo.type'] = 'SlipWall'
        elif sfclayscheme == 'MOST':
            self.erf_input['zlo.type'] = 'MOST'
        else:
            print(f'NOTE: Surface layer scheme {sfclayscheme} not implemented in ERF')
            self.erf_input['zlo.type'] = sfclayscheme

        # TODO: specify PBL scheme per level
        self.erf_input['erf.pbl_type'] = self.physics.bl_pbl_physics[0]
        if self.physics.bl_pbl_physics[0] != 'none':
            assert (not any([diff_opt.startswith('3D') for diff_opt in self.dynamics.km_opt])), \
                    'Incompatible PBL scheme and diffusion options specified'

        if any([opt != 'constant' for opt in self.dynamics.km_opt]):
            self.erf_input['erf.les_type'] = self.dynamics.km_opt[0]
            self.erf_input['erf.molec_diff_type'] = 'Constant' # default
            self.erf_input['erf.rho0_trans'] = 1.0
            self.erf_input['erf.dynamicViscosity'] = 0.0
            self.erf_input['erf.alpha_T'] = 0.0
            self.erf_input['erf.alpha_C'] = 0.0
        else:
            self.erf_input['erf.les_type'] = 'None' # default
            if any([kh != kv for kh,kv in zip(self.dynamics.khdif, self.dynamics.kvdif)]):
                print('NOTE: horizontal and vertical diffusion coefficients assumed equal')
            self.erf_input['erf.molec_diff_type'] = 'ConstantAlpha'
            self.erf_input['erf.rho0_trans'] = 1.0
            self.erf_input['erf.dynamicViscosity'] = self.dynamics.khdif[0]
            self.erf_input['erf.alpha_T'] = self.dynamics.khdif[0]
            self.erf_input['erf.alpha_C'] = self.dynamics.khdif[0]
        if any([opt != 0 for opt in self.dynamics.diff_6th_opt]):
            print('NOTE: 6th-order horizontal hyperdiffusion not implemented in ERF')
        
        # TODO: turn on Rayleigh damping, set tau
        if self.dynamics.damp_opt != 'none':
            print(f'NOTE: Upper level damping specified ({self.dynamics.damp_opt}) but not implemented in ERF')

        return self.erf_input
        
    def process_initial_conditions(self,init_input='wrfinput_d01',landuse_table_path=None):
        # Note: This calculates the heights for the outer WRF domain only
        wrfinp = xr.open_dataset(init_input)
        ph = wrfinp['PH'] # perturbation geopotential
        phb = wrfinp['PHB'] # base-state geopotential
        hgt = wrfinp['HGT'] # terrain height
        self.terrain = hgt
        geo = ph + phb # geopotential, dims=(Time: 1, bottom_top_stag, south_north, west_east)
        geo = geo/9.81 - hgt
        geo = geo.isel(Time=0).mean(['south_north','west_east']).values
        self.heights = (geo[1:] + geo[:-1]) / 2 # destaggered
        self.erf_input['erf.z_levels'] = self.heights

        # Get Coriolis parameters
        self.erf_input['erf.latitude'] = wrfinp.attrs['CEN_LAT']
        period = 4*np.pi / wrfinp['F'] * np.sin(np.radians(wrfinp.coords['XLAT'])) # F: "Coriolis sine latitude term"
        self.erf_input['erf.rotational_time_period'] = float(period.mean())

        # Get surface temperature map
        Tsurf = wrfinp['TSK'].isel(Time=0) # "surface skin temperature"
        # TODO: need to convert to surface field for ERF
        # temporarily use a scalar value
        self.erf_input['erf.most.surf_temp'] = float(Tsurf.mean())

        # Get roughness map from land use information
        if landuse_table_path is None:
            print('Need to specify `landuse_table_path` from your WRF installation'\
                  'land-use indices to z0')
            return
        LUtype =  wrfinp.attrs['MMINLU']
        alltables = LandUseTable(landuse_table_path)
        tab = alltables[LUtype]
        if isinstance(tab.index, pd.MultiIndex):
            assert tab.index.levels[1].name == 'season'
            startdate = self.time_control.start_datetime
            dayofyear = startdate.timetuple().tm_yday
            is_summer = (dayofyear >= LandUseTable.summer_start_day) & (dayofyear < LandUseTable.winter_start_day)
            #print(startdate,'--> day',dayofyear,'is summer?',is_summer)
            if is_summer:
                tab = tab.xs('summer',level='season')
            else:
                tab = tab.xs('winter',level='season')
        z0dict = tab['roughness_length'].to_dict()
        def mapfun(idx):
            return z0dict[idx]
        LU = wrfinp['LU_INDEX'].isel(Time=0).astype(int)
        z0 = xr.apply_ufunc(np.vectorize(mapfun), LU)
        print('Distribution of roughness heights')
        print('z0\tcount')
        for roughval in np.unique(z0):
            print(f'{roughval:g}\t{np.count_nonzero(z0==roughval)}')
        # TODO: need to convert to surface field for ERF
        # temporarily use a scalar value
        z0mean = float(z0.mean())
        self.erf_input['erf.most.z0'] = z0mean
        

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
        if (truelat2 is None):
            truelat2 = truelat1
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
        if not hasattr(self,'x'):
            self.setup_grid()
        if stagger=='U':
            xx,yy = np.meshgrid(self.x, self.y_destag)
        elif stagger=='V':
            xx,yy = np.meshgrid(self.x_destag, self.y)
        else:
            xx,yy = np.meshgrid(self.x_destag, self.y_destag)
        lonlat = ccrs.Geodetic().transform_points(self.proj, xx.ravel(), yy.ravel())
        lon = lonlat[:,0].reshape(xx.shape)
        lat = lonlat[:,1].reshape(xx.shape)
        return lat,lon
