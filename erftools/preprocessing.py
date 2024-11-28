import numpy as np
import pandas as pd
import xarray as xr
import f90nml

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
        

class RealSounding(object):
    """Class to extract soundings from WPS output"""

    def __init__(self,met_em_path):
        self.load_met_em(met_em_path)

    def load_met_em(self,fpath):
        ds = xr.open_dataset(fpath)
        self.U = ds['UU'] # staggered in west_east dir [m/s]
        self.V = ds['VV'] # staggered in south_north dir [m/s]
        self.P = ds['PRES'] # unstaggered [Pa]
        self.T = ds['TT'] # unstaggered [K]
        self.RH = ds['RH'] # unstaggered [%]
        # see https://github.com/a2e-mmc/mmctools/blob/master/mmctools/helper_functions.py#L104
        #   for calculating vapor mixing ratio
        # see https://github.com/a2e-mmc/mmctools/blob/master/mmctools/helper_functions.py#L76
        #   for calculating virtual temperature (to get virtual potential temperature)

    def write(self,soundingfile):
        print('Wrote',soundingfile)

