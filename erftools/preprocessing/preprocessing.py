import logging
import numpy as np
import pandas as pd
import xarray as xr
import f90nml
import calendar
import cartopy.crs as ccrs

from ..wrf.namelist import (TimeControl, Domains, Physics, Dynamics,
                           BoundaryControl)
from ..wrf.landuse import LandUseTable
from .real import RealInit
from ..inputs import ERFInputs

class WRFInputDeck(object):
    """Class to parse inputs from WRF and convert to inputs for ERF
    WRF inputs include:
    * namelist.input
    * wrfinput_d01[, wrfinput_d02, ...]
    """

    def __init__(self,nmlpath,verbosity=logging.DEBUG):
        # setup logger -- note this only has an effect the first time it
        # is called
        logging.basicConfig(level=verbosity,
                            format='%(levelname)s: %(message)s')
        # scrape WRF namelists
        with open(nmlpath,'r') as f:
            self.nml = f90nml.read(f)
        self.time_control = TimeControl(self.nml['time_control'])
        self.domains = Domains(self.nml['domains'])
        self.physics = Physics(self.nml['physics'])
        self.dynamics = Dynamics(self.nml['dynamics'])
        self.bdy_control = BoundaryControl(self.nml['bdy_control'])
        # calculate ERF equivalents
        self.set_defaults()
        self.generate_inputs()

    def __str__(self):
        s = str(self.time_control) + '\n'
        s+= str(self.domains) + '\n'
        s+= str(self.physics) + '\n'
        s+= str(self.dynamics) + '\n'
        return s

    def set_defaults(self):
        # WRF defaults
        self.input_dict = {
            'erf.use_gravity': True,
            'erf.use_coriolis': True,
            'zhi.type': 'SlipWall',
            'erf.use_terrain': True,
            'erf.init_type': 'real',
            'erf.use_real_bcs': True,
            'erf.nc_init_file_0': 'wrfinp_d01',
            'erf.nc_bdy_file': 'wrfbdy_d01',
            'erf.dycore_horiz_adv_type': 'Upwind_5th',
            'erf.dycore_vert_adv_type': 'Upwind_3rd',
            'erf.dryscal_horiz_adv_type': 'Upwind_5th',
            'erf.dryscal_vert_adv_type': 'Upwind_3rd',
            'erf.moistscal_horiz_adv_type': 'WENO5',
            'erf.moistscal_vert_adv_type': 'WENO5',
            'amr.v': 1, # verbosity in Amr.cpp
            'erf.v': 1, # verbosity in ERF.cpp
            'erf.sum_interval': 1, # timesteps between computing mass
            'erf.plot_vars_1': ['density','x_velocity','y_velocity','z_velocity',
                                'pressure','theta','KE',
                                'Kmh','Kmv','Khh','Khv','qv','qc'],
        }

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
        inp = self.input_dict

        logging.info('Assuming all domains have the same start/end datetime as level 0')
        startdate = self.time_control.start_datetimes[0]
        enddate = self.time_control.end_datetimes[0]
        tsim = (enddate - startdate).total_seconds()
        inp['start_time'] = calendar.timegm(startdate.timetuple())
        inp['stop_time'] = calendar.timegm(enddate.timetuple())
        logging.info(f'Total simulation time: {tsim}')
        logging.info(f"Start from {startdate.strftime('%Y-%m-%d %H:%M:%S')}"
                     f" ({inp['start_time']} seconds since epoch)")
        logging.info(f"Stop at {enddate.strftime('%Y-%m-%d %H:%M:%S')}"
                     f" ({inp['stop_time']} seconds since epoch)")
        assert tsim > 0, 'Start and end datetimes are equal'

        # note: starting index starts with 1, _not_ 0
        # note: ending index is the number of _staggered_ pts
        n_cell = [self.domains.e_we[0] - self.domains.s_we[0],
                  self.domains.e_sn[0] - self.domains.s_sn[0],
                  self.domains.e_vert[0] - self.domains.s_vert[0]]
        if self.domains.ztop is None:
            # get domain heights from base state geopotential
            # note: RealInit uses xarray to manage dims, kind of annoying here
            zsurf0 = xr.DataArray([[0]],dims=('west_east','south_north'))
            eta_levels = xr.DataArray(self.domains.eta_levels,
                                      dims='bottom_top_stag')
            ptop = self.domains.p_top_requested
            real = RealInit(zsurf0, eta_stag=eta_levels, ptop=ptop)
            z_levels = real.phb.squeeze().values / 9.81
            self.base_heights = z_levels
            ztop = z_levels[-1]
            inp['erf.terrain_z_levels'] = z_levels
            logging.info('Estimated domain ztop from domains.p_top_requested'
                         f'={ptop:g} : {ztop}')
        else:
            # this is only used by WRF for idealized cases
            ztop = self.domains.ztop
        logging.info('Domain SW corner is (0,0)')
        inp['geometry.prob_extent'] = [n_cell[0] * self.domains.dx[0],
                                       n_cell[1] * self.domains.dy[0],
                                       ztop]
        inp['amr.n_cell'] = n_cell
        inp['geometry.is_periodic'] = [
                self.bdy_control.periodic_x,
                self.bdy_control.periodic_y,
                0]

        assert self.domains.parent_time_step_ratio[0] == 1
        dt = np.array(self.domains.parent_time_step_ratio) * self.domains.time_step
        inp['erf.fixed_dt'] = dt[0]

        # refinements
        max_dom = self.domains.max_dom
        inp['amr.max_level'] = max_dom - 1 # zero-based indexing
        if max_dom > 1:
            logging.info('Assuming parent_time_step_ratio == parent_grid_ratio')

            refine_names = ' '.join([f'nest{idom:d}' for idom in range(1,max_dom)])
            inp['amr.refinement_indicators'] = refine_names

            dx = self.domains.dx
            dy = self.domains.dy
            imax = self.domains.e_we
            jmax = self.domains.e_sn
            ref_ratio_vect = []
            for idom in range(1,max_dom):
                grid_ratio = self.domains.parent_grid_ratio[idom]
                ref_ratio_vect += [grid_ratio, grid_ratio, 1]

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
                inp[f'amr.nest{idom:d}.in_box_lo'] = in_box_lo
                inp[f'amr.nest{idom:d}.in_box_hi'] = in_box_hi

            inp['amr.ref_ratio_vect'] = ref_ratio_vect

        restart_period = self.time_control.restart_interval * 60.0 # [s]
        inp['amr.check_int'] = int(restart_period / dt[0])

        sfclayscheme = self.physics.sf_sfclay_physics[0]
        if sfclayscheme == 'None':
            inp['zlo.type'] = 'SlipWall'
        elif sfclayscheme == 'MOST':
            inp['zlo.type'] = 'MOST'
        else:
            logging.warning(f'Surface layer scheme {sfclayscheme} not implemented in ERF')
            inp['zlo.type'] = sfclayscheme

        inp['erf.pbl_type'] = self.physics.bl_pbl_physics
        for idom in range(max_dom):
            if self.physics.bl_pbl_physics[idom] != 'None':
                km_opt = self.dynamics.km_opt[idom]
                if km_opt in ['Deardorff','Smagorinsky']:
                    logging.warning(f'erf.pbl_type[{idom}]={self.physics.bl_pbl_physics[idom]}'
                                    f' selected with 3D diffusion'
                                    f' (km_opt={km_opt})')

        if any([km_opt == 'constant' for km_opt in self.dynamics.km_opt]):
            if any([kh > 0 for kh in self.dynamics.khdif]):
                kdif = self.dynamics.khdif[0]
                if any([kv!=kh for kv,kh in zip(self.dynamics.khdif,
                                                self.dynamics.kvdif)]):
                    logging.info(f'Specifying khdif = kvdif = {kdif}')
                if len(set(self.dynamics.khdif)) > 1:
                    # more than one diffusion constant specified
                    logging.info(f'Specifying constant molecular diffusion on'
                                 f'all levels = {kdif}')
                inp['erf.molec_diff_type'] = 'ConstantAlpha'
                inp['erf.dynamic_viscosity'] = kdif
                inp['erf.alpha_T'] = kdif
                inp['erf.alpha_C'] = kdif
            else:
                logging.info('Requested km_opt=1 but nonzero diffusion'
                             ' constant has not be specified')

        if any([opt != 0 for opt in self.dynamics.diff_6th_opt]):
            if any([opt==1 for opt in self.dynamics.diff_6th_opt]):
                logging.warning('Simple 6th-order hyper diffusion is not recommended')
            num_diff_coeff = self.dynamics.diff_6th_factor[0]
            logging.warning(f'Applying numerical diffusion on all'
                            f' levels, with erf.num_diff_coeff'
                            f'={num_diff_coeff} -- this can have'
                            f' unexpected effects in ERF')
            inp['erf.num_diff_coeff'] = num_diff_coeff
        
        if any([opt != 'constant' for opt in self.dynamics.km_opt]):
            # in ERF, Smagorinsky == 2D Smagorinsky
            les_types = [turb if 'Smagorinsky' not in turb else 'Smagorinsky'
                         for turb in self.dynamics.km_opt]
            les_types = les_types[:max_dom]
            inp['erf.les_type'] = les_types
            smag_Cs = self.dynamics.c_s
            dear_Ck = self.dynamics.c_k
            inp['erf.Cs'] = smag_Cs[:max_dom]
            inp['erf.Ck'] = dear_Ck[:max_dom]

        if any([opt != 'constant' for opt in self.dynamics.km_opt]):
            # in ERF, Smagorinsky == 2D Smagorinsky
            pbl_types = self.physics.bl_pbl_physics[:max_dom]
            inp['erf.pbl_type'] = pbl_types

        if any([opt != 'None' for opt in self.physics.mp_physics]):
            moisture_model = self.physics.mp_physics[0]
            if len(set(self.physics.mp_physics)) > 1:
                logging.warning(f'Applying the {moisture_model} microphysics'
                                ' model on all levels')
            inp['erf.moisture_model'] = moisture_model

        if any([opt != 'None' for opt in self.physics.ra_physics]):
            rad_model = self.physics.ra_physics[0]
            if len(set(self.physics.ra_physics)) > 1:
                logging.warning(f'Applying the {rad_model} radiation scheme on'
                                ' all levels')
            inp['erf.radiation_model'] = rad_model

        if any([opt != 'None' for opt in self.physics.cu_physics]):
            logging.warning('ERF currently does not have any cumulus parameterizations')

        # TODO: turn on Rayleigh damping, set tau
        if self.dynamics.damp_opt != 'none':
            print(f'NOTE: Upper level damping specified ({self.dynamics.damp_opt}) but not implemented in ERF')

        self.input_dict = inp
        
    def process_initial_conditions(self,init_input='wrfinput_d01',landuse_table_path=None):
        wrfinp = xr.open_dataset(init_input)

        logging.info('Overwriting base-state geopotential heights with heights'
                     ' from {init_input}')
        ph = wrfinp['PH'] # perturbation geopotential
        phb = wrfinp['PHB'] # base-state geopotential
        hgt = wrfinp['HGT'] # terrain height
        self.terrain = hgt
        gh = ph + phb # geopotential, dims=(Time: 1, bottom_top_stag, south_north, west_east)
        gh = gh/9.81
        gh = gh.isel(Time=0).mean(['south_north','west_east']).values
        self.heights = (gh[1:] + gh[:-1]) / 2 # destaggered
        self.input_dict['erf.terrain_z_levels'] = self.heights

        # Get Coriolis parameters
        self.input_dict['erf.latitude'] = wrfinp.attrs['CEN_LAT']
        period = 4*np.pi / wrfinp['F'] * np.sin(np.radians(wrfinp.coords['XLAT'])) # F: "Coriolis sine latitude term"
        self.input_dict['erf.rotational_time_period'] = float(period.mean())

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
        self.input_dict['erf.most.z0'] = z0mean

    def write_inputfile(self,fpath):
        inp = ERFInputs(**self.input_dict)
        inp.write(fpath)
        

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
