import logging
import numpy as np
import pandas as pd
import xarray as xr
import f90nml
import calendar
from scipy.interpolate import RegularGridInterpolator

from ..wrf.namelist import (TimeControl, Domains, Physics, Dynamics,
                           BoundaryControl)
from ..wrf.landuse import LandUseTable
from .real import RealInit, get_zlevels_auto
from ..constants import CONST_GRAV
from ..inputs import ERFInputs

class WRFInputDeck(object):
    """Class to parse inputs from WRF and convert to inputs for ERF
    WRF inputs include:
    * namelist.input
    * wrfinput_d01[, wrfinput_d02, ...]
    """

    def __init__(self,nmlpath,verbosity=logging.DEBUG):
        # setup logger
        self.log = logging.getLogger(__name__)
        self.log.setLevel(verbosity)
        sh = logging.StreamHandler()
        sh.setLevel(verbosity)
        fmt = logging.Formatter('%(levelname)s: %(message)s')
        sh.setFormatter(fmt)
        self.log.addHandler(sh)

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
            'xlo.type': 'Outflow',
            'xhi.type': 'Outflow',
            'ylo.type': 'Outflow',
            'yhi.type': 'Outflow',
            'zhi.type': 'SlipWall',
            'erf.use_terrain': True,
            'erf.terrain_smoothing': 1,
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
            'erf.plot_file_1': 'plt',
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

        self.log.info('Assuming all domains have the same start/end datetime as level 0')
        startdate = self.time_control.start_datetimes[0]
        enddate = self.time_control.end_datetimes[0]
        tsim = (enddate - startdate).total_seconds()
        inp['start_date'] = startdate
        inp['stop_date'] = enddate
        inp['start_time'] = calendar.timegm(startdate.timetuple())
        inp['stop_time'] = calendar.timegm(enddate.timetuple())
        self.log.info(f'Total simulation time: {tsim}')
        self.log.info(f"Start from {startdate.strftime('%Y-%m-%d %H:%M:%S')}"
                      f" ({inp['start_time']} seconds since epoch)")
        self.log.info(f"Stop at {enddate.strftime('%Y-%m-%d %H:%M:%S')}"
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
            ptop = self.domains.p_top_requested
            if self.domains.eta_levels is not None:
                eta_levels = xr.DataArray(self.domains.eta_levels,
                                          dims='bottom_top_stag')
                real = RealInit(zsurf0, eta_stag=eta_levels, ptop=ptop)
                z_levels = real.phb.squeeze().values / CONST_GRAV
            else:
                z_levels,_,_ = get_zlevels_auto(n_cell[2],
                                                self.domains.dzbot,
                                                self.domains.max_dz,
                                                self.domains.dzstretch_s,
                                                self.domains.dzstretch_u,
                                                ptop=ptop)
            self.base_heights = z_levels
            inp['erf.terrain_z_levels'] = z_levels
            most_zref = 0.5*(z_levels[0] + z_levels[1])
            inp['erf.most.zref'] = most_zref  # need to specify for terrain
            ztop = z_levels[-1]
            self.log.info('Estimated domain ztop from domains.p_top_requested'
                          f'={ptop:g} : {ztop}')
        else:
            # this is only used by WRF for idealized cases
            ztop = self.domains.ztop
        self.log.info('Domain SW corner is (0,0)')
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
            self.log.info('Assuming parent_time_step_ratio == parent_grid_ratio')

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

        restart_interval = self.time_control.restart_interval * 60.0 # [s]
        inp['erf.check_int'] = int(restart_interval / dt[0])

        wrfout_interval = self.time_control.history_interval[0] * 60.0 # [s]
        inp['erf.plot_int_1'] = int(wrfout_interval / dt[0])

        sfclayscheme = self.physics.sf_sfclay_physics[0]
        if sfclayscheme == 'None':
            inp['zlo.type'] = 'SlipWall'
        elif sfclayscheme == 'MOST':
            inp['zlo.type'] = 'MOST'
        else:
            self.log.warning(f'Surface layer scheme {sfclayscheme} not implemented in ERF')
            inp['zlo.type'] = sfclayscheme

        inp['erf.pbl_type'] = self.physics.bl_pbl_physics
        for idom in range(max_dom):
            if self.physics.bl_pbl_physics[idom] != 'None':
                km_opt = self.dynamics.km_opt[idom]
                if km_opt in ['Deardorff','Smagorinsky']:
                    self.log.warning(f'erf.pbl_type[{idom}]={self.physics.bl_pbl_physics[idom]}'
                                     f' selected with 3D diffusion'
                                     f' (km_opt={km_opt})')

        if any([km_opt == 'constant' for km_opt in self.dynamics.km_opt]):
            if any([kh > 0 for kh in self.dynamics.khdif]):
                kdif = self.dynamics.khdif[0]
                if any([kv!=kh for kv,kh in zip(self.dynamics.khdif,
                                                self.dynamics.kvdif)]):
                    self.log.info(f'Specifying khdif = kvdif = {kdif}')
                if len(set(self.dynamics.khdif)) > 1:
                    # more than one diffusion constant specified
                    self.log.info(f'Specifying constant molecular diffusion on'
                                  f'all levels = {kdif}')
                inp['erf.molec_diff_type'] = 'ConstantAlpha'
                inp['erf.dynamic_viscosity'] = kdif
                inp['erf.alpha_T'] = kdif
                inp['erf.alpha_C'] = kdif
            else:
                self.log.info('Requested km_opt=1 but nonzero diffusion'
                              ' constant has not be specified')

        if any([opt != 0 for opt in self.dynamics.diff_6th_opt]):
            if any([opt==1 for opt in self.dynamics.diff_6th_opt]):
                self.log.warning('Simple 6th-order hyper diffusion is not recommended')
            num_diff_coeff = self.dynamics.diff_6th_factor[0]
            self.log.warning(f'Applying numerical diffusion on all'
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
                self.log.warning(f'Applying the {moisture_model} microphysics'
                                 ' model on all levels')
            inp['erf.moisture_model'] = moisture_model

        if any([opt != 'None' for opt in self.physics.ra_physics]):
            rad_model = self.physics.ra_physics[0]
            if len(set(self.physics.ra_physics)) > 1:
                self.log.warning(f'Applying the {rad_model} radiation scheme on all levels')
            inp['erf.radiation_model'] = rad_model

        if any([opt != 'None' for opt in self.physics.cu_physics]):
            self.log.warning('ERF currently does not have any cumulus parameterizations')

        if self.dynamics.damp_opt != 'none':
            if self.dynamics.damp_opt.startswith('Rayleigh'):
                self.log.info(f'Applying Rayleigh damping to w on all levels'
                              f' based on level 0 inputs')
                inp['erf.rayleigh_damp_W'] = True
                inp['erf.rayleigh_dampcoef'] = self.dynamics.dampcoef[0]
                inp['erf.rayleigh_zdamp'] = self.dynamics.zdamp[0]
            else:
                self.log.warning(f'Damping option {self.dynamics.damp_opt} not supported')

        self.input_dict = inp
        
    def process_initial_conditions(self,init_input='wrfinput_d01',
                                   calc_geopotential_heights=False,
                                   landuse_table_path=None,
                                   write_hgt=None,
                                   write_z0=None):
        wrfinp = xr.open_dataset(init_input)

        # Get Coriolis parameters
        period = 4*np.pi / wrfinp['F'] * np.sin(np.radians(wrfinp.coords['XLAT'])) # F: "Coriolis sine latitude term"
        mean_lat = np.mean(wrfinp.coords['XLAT'].values)
        self.log.info(f"Using mean XLAT={mean_lat}"
                      f" (projection CEN_LAT={wrfinp.attrs['CEN_LAT']})")
        mean_period = np.mean(period.values)
        self.log.info(f"Earth rotational period from Coriolis param :"
                      f" {mean_period/3600} h")
        self.input_dict['erf.latitude'] = mean_lat
        self.input_dict['erf.rotational_time_period'] = mean_period

        if calc_geopotential_heights:
            self.log.info(f'Overwriting base-state geopotential heights with heights'
                          f' from {init_input}')
            ph = wrfinp['PH'] # perturbation geopotential
            phb = wrfinp['PHB'] # base-state geopotential
            gh = ph + phb # geopotential, dims=(Time: 1, bottom_top_stag, south_north, west_east)
            gh = gh / CONST_GRAV
            self.heights = gh.isel(Time=0).mean(['south_north','west_east']).values
            self.input_dict['erf.terrain_z_levels'] = self.heights
            most_zref = 0.5*(self.heights[0] + self.heights[1])
            inp['erf.most.zref'] = most_zref  # need to specify for terrain

        # Grid data needed if hgt or z0 are written
        dx = self.domains.dx[0]
        dy = self.domains.dy[0]
        nx = wrfinp.sizes['west_east']
        ny = wrfinp.sizes['south_north']
        west_east   = np.arange(0.5,nx) * dx
        south_north = np.arange(0.5,ny) * dy
        west_east_stag   = np.arange(nx+1) * dx
        south_north_stag = np.arange(ny+1) * dy
        xg,yg = np.meshgrid(west_east_stag, south_north_stag, indexing='ij')

        # Write out terrain elevation map
        if write_hgt is not None:
            hgt = wrfinp['HGT'].isel(Time=0) # terrain height
            self.terrain = hgt # save orig terrain data
            # interpolate to nodes
            hgt = hgt.transpose('west_east','south_north')
            interpfun = RegularGridInterpolator(
                    (west_east,south_north), hgt.values,
                    bounds_error=False,
                    fill_value=None)
            hgt_nodes = interpfun((xg,yg))
            xyz = np.stack((xg.ravel(order='F'),
                            yg.ravel(order='F'),
                            hgt_nodes.ravel(order='F')),axis=-1)
            print('Writing out',write_hgt)
            np.savetxt(write_hgt, xyz, fmt='%.8g')
            self.input_dict['erf.terrain_file_name'] = write_hgt

        # Get roughness map from land use information
        if landuse_table_path is None:
            print('Need to specify `landuse_table_path` from your WRF installation'
                  'land-use indices to estimate z0')
            if write_z0:
                print('Surface roughness map not was not written')
        else:
            LUtype =  wrfinp.attrs['MMINLU']
            alltables = LandUseTable(landuse_table_path, verbose=False)
            tab = alltables[LUtype]
            if isinstance(tab.index, pd.MultiIndex):
                assert tab.index.levels[1].name == 'season'
                startdate = self.time_control.start_datetimes[0]
                dayofyear = startdate.timetuple().tm_yday
                is_summer = (dayofyear >= LandUseTable.summer_start_day) \
                          & (dayofyear < LandUseTable.winter_start_day)
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
            self.z0 = z0

            # Write out surface roughness map
            if write_z0:
                # interpolate to nodes
                z0 = z0.transpose('west_east','south_north')
                interpfun = RegularGridInterpolator(
                        (west_east,south_north), z0.values,
                        bounds_error=False,
                        fill_value=None)
                z0_nodes = interpfun((xg,yg))
                xyz0 = np.stack((xg.ravel(order='F'),
                                 yg.ravel(order='F'),
                                 z0_nodes.ravel(order='F')),axis=-1)
                print('Writing out',write_z0)
                np.savetxt(write_z0, xyz0, fmt='%.8g')
                self.input_dict['erf.most.roughness_file_name'] = write_z0
            else:
                self.log.info('Roughness map not written,'
                              ' using mean roughness for MOST')
                print('Distribution of roughness heights')
                print('z0\tcount')
                for roughval in np.unique(z0):
                    print(f'{roughval:g}\t{np.count_nonzero(z0==roughval)}')
                z0mean = float(z0.mean())
                self.input_dict['erf.most.z0'] = z0mean

    def write_inputfile(self,fpath):
        inp = ERFInputs(**self.input_dict)
        inp.write(fpath)
        print('Wrote',fpath)
