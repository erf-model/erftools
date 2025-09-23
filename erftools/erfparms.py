"""
Data container for ParmParse data

see https://erf.readthedocs.io/en/latest/Inputs.html
"""
import warnings
import numpy as np
from dataclasses import field
from typing import List, Tuple, Union
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict


def check_unknown_params(data_dict, dataclass_type):
    known_params = set(dataclass_type.__dataclass_fields__.keys())
    provided_params = set(data_dict.keys())
    unknown_params = list(provided_params - known_params)

    # ignore MOST settings
    unknown_params = [param for param in unknown_params
                      if not param.startswith('most.')]

    # ignore refinement indicators
    if 'refinement_indicators' in data_dict:
        refine_names = data_dict['refinement_indicators']
        if isinstance(refine_names, str):
            refine_names = [refine_names]
        for box in refine_names:
            unknown_params = [param for param in unknown_params
                              if not param.startswith(f'{box}.')]

    # ignore nc_init_file_*
    unknown_params = [param for param in unknown_params
                      if not param.startswith('nc_init_file_')]

    if unknown_params:
        warnings.warn(f'Non-standard {dataclass_type.__name__} ignored: {unknown_params}')


@dataclass(config=ConfigDict(extra='allow'))
class AMRParms:
    """amr.* parameters"""
    n_cell: Tuple[int, int, int] = (0,0,0)
    max_level: int = 0
    ref_ratio: Union[int,List[int]] = 2
    ref_ratio_vect: List[int] = field(default_factory=list)

    v: int = 0  # verbosity

    def __post_init__(self):
        assert all([ival>0 for ival in self.n_cell]), \
                'Need to specify amr.n_cell'
        assert self.max_level >= 0
        if isinstance(self.ref_ratio, list):
            assert len(self.ref_ratio) == self.max_level, \
                    'Need to specify a constant amr.ref_ratio' \
                    ' or one value per level'
            assert all(ratio in [2,3,4] for ratio in self.ref_ratio), \
                    'Invalid refinement ratio(s)'
        else:
            assert self.ref_ratio in [2,3,4], 'Invalid amr.ref_ratio'
        assert len(self.ref_ratio_vect) % 3 == 0, \
                'Need to specify ref ratios for each direction'
        assert all([ival in [1,2,3,4] for ival in self.ref_ratio_vect]), \
                'Invalid directional refinement ratio(s)'

    
@dataclass(config=ConfigDict(extra='allow'))
class GeometryParms:
    """geometry.* parameters"""
    prob_lo: Tuple[float, float, float] = (0.,0.,0.)
    prob_hi: List[float] = field(default_factory=list)
    prob_extent: List[float] = field(default_factory=list)
    is_periodic: Tuple[int, int, int] = (0,0,0)

    def __post_init__(self):
        have_prob_hi = (len(self.prob_hi) == 3)
        have_prob_extent = (len(self.prob_extent) == 3)
        assert have_prob_hi or have_prob_extent
        if have_prob_extent:
            self.prob_hi = [self.prob_lo[i] + self.prob_extent[i]
                            for i in range(3)]
        else:
            self.prob_extent = [self.prob_hi[i] - self.prob_lo[i]
                                for i in range(3)]
        assert all([ival==0 or ival==1 for ival in self.is_periodic])


dycore_adv_schemes = [
    # from WRF
    'Centered_2nd',
    'Upwind_3rd',
    'Blended_3rd4th',
    'Centered_4th',
    'Upwind_5th',
    'Blended_5th6th',
    'Centered_6th',
    # WENO-JS
    'WENO3',
    'WENO5',
    'WENO7'
    # WENO-Z
    'WENOZ3',
    'WENOZ5',
    'WENOZ7'
]
extra_scalar_adv_schemes = [
    'Upwind_3rd_SL' # slope-limited
    'WENOMZQ3',
]

# corresponding to ERF InitType
init_types = [
    'none',
    'uniform',
    'input_sounding',
    'wrfinput',
    'metgrid',
    'ncfile',
]

# corresponding to ERF MoistureType
moisture_models = [
    'none',
    'satadj',
    'kessler', 'kessler_norain',
    'sam', 'sam_noice', 'sam_noprecip_noice',
    'morrison', 'morrison_noice',
 ]

@dataclass(config=ConfigDict(extra='allow'))
class ERFParms:
    """erf.* parameters"""

    # Governing Equations
    anelastic: bool = False  # solv anelastic eqns instead of compressible
    use_fft: bool = False  # use FFT rather than multigrid to solve Poisson eqns
    mg_v: int = 0  # multigrid solver verbosiy when solving Poisson

    # Refinement
    refinement_indicators: Union[str,List[str]] = field(default_factory=list)
    regrid_int: int = -1

    # Grid Stretching
    grid_stretching_ratio: float = 1.
    initial_dz: float = np.nan
    terrain_z_levels: List[float] = field(default_factory=list)

    # Time Step
    substepping_type: str = 'DEFAULT'
    cfl: float = 0.8
    substepping_cfl: float = 1.0
    fixed_dt: float = np.nan
    fixed_fast_dt: float = np.nan
    fixed_mri_dt_ratio: int = -1

    # Restart
    restart: str = ''
    check_file: str = 'chk'
    check_int: int = -1
    check_per: float = -1.

    # PlotFiles
    plotfile_type: str = 'amrex'
    plot_file_1: str = 'plt_1_'
    plot_file_2: str = 'plt_2_'
    plot_int_1: int = -1
    plot_int_2: int = -1
    plot_per_1: float = -1.
    plot_per_2: float = -1.
    plot_vars_1: List[str] = field(default_factory=list)
    plot_vars_2: List[str] = field(default_factory=list)

    # Screen Output
    v: int = 0  # verbosity
    sum_interval: int = -1

    # Diagnostic Ouptuts
    data_log: List[str] = field(default_factory=list)
    profile_int: int = -1
    destag_profiles: bool = True

    # Line Sampling
    do_line_sampling: bool = False
    line_sampling_vars: List[str] = field(default_factory=list)
    line_sampling_text_output: bool = False
    sampler_interval: int = -1
    sample_line_lo: List[float] = field(default_factory=list)
    sample_line_hi: List[float] = field(default_factory=list)
    sample_line_dir: List[int] = field(default_factory=list)

    # Advection Schemes
    dycore_horiz_adv_type: str = 'Upwind_3rd'
    dycore_vert_adv_type: str = 'Upwind_3rd'
    dryscal_horiz_adv_type: str = 'Upwind_3rd'
    dryscal_vert_adv_type: str = 'Upwind_3rd'
    moistscal_horiz_adv_type: str = 'Upwind_3rd'
    moistscal_vert_adv_type: str = 'Upwind_3rd'
    dycore_horiz_upw_frac: float = 1.
    dycore_vert_upw_frac: float = 1.
    dryscal_horiz_upw_frac: float = 1.
    dryscal_vert_upw_frac: float = 1.
    moistscal_horiz_upw_frac: float = 1.
    moistscal_vert_upw_frac: float = 1.
    use_efficient_advection: bool = False
    use_mono_advection: bool = False

    # Diffusive Physics
    molec_diff_type: str = 'None'
    dynamic_viscosity: float = 0.
    rho0_trans: float = 0.
    alpha_T: float = 0.
    alpha_C: float = 0.

    les_type: Union[str,List[str]] = 'None'
    Pr_t: float = 1.0
    Sc_t: float = 1.0

    # - Smagorinsky SGS model
    Cs: Union[float,List[float]] = 0.

    # - Deardorff SGS model
    Ck: Union[float,List[float]] = 0.1
    Ce: float = 0.93
    Ce_wall: float = -1.
    sigma_k: float = 0.5
    theta_ref: float = 300.

    # - numerical diffusion
    num_diff_coeff: float = 0.

    # PBL Scheme
    pbl_type: Union[str,List[str]] = 'None'

    # Forcing Terms
    use_gravity: bool = False
    use_coriolis: bool = False
    rotational_time_period: float = 86400.
    latitude: float = 90.
    coriolis_3d: bool= True

    abl_driver_type: str = 'None'
    abl_pressure_grad: Tuple[float,float,float] = (0.,0.,0.)
    abl_geo_wind: Tuple[float,float,float] = (0.,0.,0.)
    abl_geo_wind_table: str = ''

    nudging_from_input_sounding: bool = False
    input_sounding_file: Union[str,List[str]] = field(default_factory=list)
    input_sounding_time: List[float] = field(default_factory=list)

    rayleigh_damp_U: bool = False
    rayleigh_damp_V: bool = False
    rayleigh_damp_W: bool = False
    rayleigh_damp_T: bool = False
    rayleigh_dampcoef: float = 0.2
    rayleigh_zdamp: float = 500.

    # Initialization
    init_type: str = 'None'
    init_sounding_ideal: bool = False
    nc_init_file_0: Union[str,List[str]] = ''
    nc_bdy_file: str = ''
    project_initial_velocity: bool = False
    use_real_bcs: bool = False
    real_width: int = 5
    real_set_width: int = 1

    # Terrain
    terrain_type: str = 'None'
    terrain_smoothing: int = 0
    terrain_file_name: str = ''

    # Moisture
    moisture_model: str = 'None'

    # Radiation
    radiation_model: str = 'None'
    rad_freq_in_steps: int = 1
    rad_write_fluxes: bool = False
    #nswbands: int = 14
    #nlwbands: int = 16
    #nswgpts: int = 224
    #nlwgpts: int = 256
    co2vmr: float = 0.00036 # from DevTests/Radiation
    o3vmr: Union[float,List[float]] = 1.887e-7 # default
    n2ovmr: float = 3.2e-7 # from DevTests/Radiation
    covmr: float = 1.5e-7 # from DevTests/Radiation
    ch4vmr: float = 1.7e-6 # from DevTests/Radiation
    o2vmr: float = 0.209 # default
    n2vmr: float = 0.7906 # default
    rrtmgp_file_path: str = './'
    #rrtmgp_coeffs_sw: str = 'rrtmgp-data-sw-g224-2018-12-04.nc'
    #rrtmgp_coeffs_lw: str = 'rrtmgp-data-lw-g256-2018-12-04.nc'
    #rrtmgp_cloud_optics_sw: str = 'rrtmgp-cloud-optics-coeffs-sw.nc'
    #rrtmgp_cloud_optics_lw: str = 'rrtmgp-cloud-optics-coeffs-lw.nc'

    # Land Surface
    land_surface_model: str = 'None'

    def __post_init__(self):
        if self.anelastic:
            assert self.use_fft
            assert self.project_initial_velocity
            if self.substepping_type == 'DEFAULT':
                self.substepping_type = 'none'
        else:
            if self.substepping_type == 'DEFAULT':
                self.substepping_type = 'implicit'

        if isinstance(self.refinement_indicators, str):
            self.refinement_indicators = [self.refinement_indicators]

        assert self.cfl > 0 and self.cfl <= 1, 'erf.cfl out of range'
        assert self.substepping_cfl > 0 and self.substepping_cfl <= 1, \
                'erf.substepping_cfl out of range'
        if self.fixed_mri_dt_ratio > 0:
            assert self.fixed_mri_dt_ratio % 2 == 0, \
                    'erf.fixed_mri_dt_ratio should be even'
        if (self.fixed_dt is not np.nan) and (self.fixed_fast_dt is not np.nan):
            self.fixed_mri_dt_ratio = int(self.fixed_dt / self.fixed_fast_dt)
            assert self.fixed_mri_dt_ratio % 2 == 0, \
                    'erf.fixed_dt/erf.fixed_fast_dt should be even'

        assert len(self.data_log) <= 4, 'Unexpected number of data_log files'

        if len(self.sample_line_lo) > 0:
            nlines = len(self.sample_line_lo) // 3
            assert len(self.sample_line_hi) == len(self.sample_line_lo)
            assert len(self.sample_line_lo) == 3*nlines, \
                    'Unexpected number of ijk values in sampling indices'
            if len(self.sample_line_dir) > 0:
                assert len(self.sample_line_dir) == nlines
            else:
                self.sample_line_dir = nlines*[2]
        assert len(self.sample_line_lo) == len(self.sample_line_hi)

        for vartype in ['dycore','dryscal','moistscal']:
            for advdir in ['horiz','vert']:
                advinp = f'{vartype}_{advdir}_adv_type'
                advscheme = getattr(self, advinp)
                if vartype == 'dycore':
                    assert advscheme in dycore_adv_schemes, \
                            f'Unexpected erf.{advinp}={advscheme}'
                else:
                    assert advscheme in (dycore_adv_schemes
                                         +extra_scalar_adv_schemes), \
                            f'Unexpected erf.{advinp}={advscheme}'
                if advscheme.startswith('Blended'):
                    upwinding = getattr(self, f'{vartype}_{advdir}_upw_frac')
                    assert (upwinding >= 0) and (upwinding <= 1)

        assert self.molec_diff_type in ['None','Constant','ConstantAlpha'], \
                f'Unexpected erf.molec_diff_type={erf.molec_diff_type}'

        les_types = self.les_type if isinstance(self.les_type,list) else [self.les_type]
        for turbmodel in les_types:
            assert turbmodel in ['None','Smagorinsky','Smagorinsky2D','Deardorff'], \
                    f'Unexpected erf.les_type={turbmodel}'
        if any([turbmodel == 'Smagorinsky' for turbmodel in les_types]):
            smag_Cs = self.Cs if isinstance(self.Cs,list) else len(les_types)*[self.Cs]
            assert all([Cs > 0 for Cs in smag_Cs]), 'Need to specify valid Smagorinsky Cs'

        pbl_types = self.pbl_type if isinstance(self.pbl_type,list) else [self.pbl_type]
        for pblscheme in pbl_types:
            assert pblscheme in ['None','MYJ','MYNN25','MRF','YSU'], \
                    f'Unexpected erf.pbl_type={pblscheme}'

        assert self.abl_driver_type in \
                ['None','PressureGradient','GeostrophicWind'], \
                f'Unexpected erf.abl_driver_type={self.abl_driver_type}'

        if self.nudging_from_input_sounding \
                and (len(self.input_sounding_file) > 1):
            assert len(self.input_sounding_file) == len(self.input_sounding_time), \
                    'Need to specify corresponding erf.input_sounding_time'
        elif isinstance(self.input_sounding_file, list) \
                and (len(self.input_sounding_file) > 0):
            self.input_sounding_file = self.input_sounding_file[0]

        assert self.init_type.lower() in init_types, \
                f'Invalid erf.init_type={self.init_type}'
        if self.init_type.lower() == 'real':
            assert isinstance(self.nc_init_file_0, str), \
                    'should only have one nc_init_file_0'
        elif self.init_type.lower() == 'metgrid' \
                and isinstance(self.nc_init_file_0, str):
            self.nc_init_file_0 = [self.nc_init_file_0]

        if self.have_terrain():
            assert self.terrain_type.lower() in ['none','staticfittedmesh',
                                                 'movingfittedmesh',
                                                 'immersedforcing', 'eb'], \
                f'Invalid erf.terrain_type {self.terrain_type}'
        assert (self.terrain_smoothing >= 0) and (self.terrain_smoothing <= 2),\
                'Invalid erf.terrain_smoothing option'

        assert self.moisture_model.lower() in moisture_models, \
                f'Unexpected erf.moisture_model={self.moisture_model}'

    def have_terrain(self):
        return self.terrain_type.lower() != 'none'
