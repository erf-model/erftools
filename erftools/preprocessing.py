import numpy as np
import pandas as pd
import xarray as xr
import f90nml

from .wrf_namelist_input import TimeControl, Domains, Physics
from .wrf_landuse import LandUseTable
from .inputs import ERFInputFile

class WRFInputDeck(object):
    """Class to parse inputs from WRF and convert to inputs for ERF
    WRF inputs include:
    * namelist.input
    * wrfinput_d01[, wrfinput_d02, ...]
    """

    def __init__(self,nmlpath):
        with open(nmlpath,'r') as f:
            self.nml = f90nml.read(f)
        self.time_control = TimeControl(self.nml['time_control'])
        self.domains = Domains(self.nml['domains'])
        self.physics = Physics(self.nml['physics'])
        self.erf_input = ERFInputFile()
        self.calculate_inputs()

    def __str__(self):
        s = str(self.time_control) + '\n'
        s+= str(self.domains)+ '\n'
        s+= str(self.physics)+ '\n'
        return s

    def calculate_inputs(self):
        """Scrape inputs for ERF from a WRF namelist.input file

        Note that the namelist does not provide the following input information
        * WRF geopotential height levels
        * surface temperature map
        * surface roughness map
        To get these values, call `process_initial_conditions()` to extract
        them from `wrfinput_d01` (output from real.exe).
        """
        tdelta = self.time_control.end_datetime - self.time_control.start_datetime
        self.erf_input['stop_time'] = tdelta.total_seconds()

        # note: number vert pts is staggered
        # TODO: add vertical stretching
        n_cell = np.array([self.domains.e_we[0], self.domains.e_sn[0], self.domains.e_vert[0]-1])
        ztop_est = 287.0 * 300.0 / 9.81 * np.log(1e5/self.domains.p_top_requested)
        self.erf_input['geometry.prob_extent'] = n_cell * np.array([self.domains.dx[0], self.domains.dy[0], np.nan])
        self.erf_input['geometry.prob_extent'][2] = ztop_est
        self.erf_input['amr.n_cell'] = n_cell

        # TODO: verify that refined regions will take finer time steps
        dt = np.array(self.domains.parent_time_step_ratio) * self.domains.time_step
        self.erf_input['erf.fixed_dt'] = dt[0]

        self.erf_input['amr.max_level'] = self.domains.max_dom - 1 # zero-based indexing
        grid_ratio = self.domains.parent_grid_ratio[1] # TODO: assume all nests have same ratio
        self.erf_input['amr.ref_ratio_vect'] = [grid_ratio, grid_ratio, 1]
        refine_names = ' '.join([f'box{idom:d}' for idom in range(1,self.domains.max_dom)])
        self.erf_input['amr.refinement_indicators'] = refine_names
        for idom in range(1,self.domains.max_dom):
            parent_ds  = np.array([  self.domains.dx[idom-1],   self.domains.dy[idom-1]], dtype=float)
            child_ds   = np.array([  self.domains.dx[idom  ],   self.domains.dy[idom  ]], dtype=float)
            parent_ext = np.array([self.domains.e_we[idom-1], self.domains.e_sn[idom-1]]) * parent_ds
            child_ext  = np.array([self.domains.e_we[idom  ], self.domains.e_sn[idom  ]]) * child_ds
            lo_idx = np.array([self.domains.i_parent_start[idom]-1, self.domains.j_parent_start[idom]-1])
            in_box_lo = lo_idx * parent_ds
            in_box_hi = in_box_lo + child_ext
            assert (in_box_hi[0] <= parent_ext[0])
            assert (in_box_hi[1] <= parent_ext[1])
            self.erf_input[f'amr.box{idom:d}.in_box_lo'] = in_box_lo
            self.erf_input[f'amr.box{idom:d}.in_box_hi'] = in_box_hi

        restart_period = self.time_control.restart_interval * 60.0 # [s]
        self.erf_input['amr.check_int'] = int(restart_period / dt[0])

        # TODO: specify PBL scheme per level
        self.erf_input['erf.pbl_type'] = self.physics.bl_pbl_physics[0]
        
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
            self.erf_input['erf.most.z0'] = 'UNDEFINED'
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
        

