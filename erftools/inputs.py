import sys
import contextlib
import numpy as np

# parameters parsed by AMReX's ParmParse
from .parms import AMRParms, GeometryParms, ERFParms


def parmparse(prefix, ppdata):
    return {key[len(prefix)+1:]: val
            for key,val in ppdata.items()
            if key.startswith(prefix+'.')}

# https://stackoverflow.com/questions/17602878/how-to-handle-both-with-open-and-sys-stdout-nicely
@contextlib.contextmanager
def open_file_or_stdout(fpath=None):
    if (fpath is not None):
        f = open(fpath, 'w')
    else:
        f = sys.stdout
    try:
        yield f
    finally:
        if f is not sys.stdout:
            f.close()

def bool_to_str(bval):
    return str(bval).lower()

def list_to_str(mylist,dtype=None):
    if dtype is None:
        return ' '.join([f'{val:.8g}' for val in mylist])
    else:
        return ' '.join([f'{dtype(val):.8g}' for val in mylist])

def strs_to_str(mylist):
    return ' '.join([s.strip('"').strip("'") for s in mylist])


class ERFInputs(object):
    """Input data container with validation and output"""

    def __init__(self,inpfile=None,**ppdata):
        if inpfile:
            ppdata = self.parse_input(inpfile)
        self.inpdict = ppdata

        # top level inputs
        self.max_step = int(ppdata.get('max_step',-1))
        self.start_time = float(ppdata.get('start_time',0.))
        self.stop_time = float(ppdata.get('stop_time',1e34))
        self.start_date = ppdata.get('start_date',None)
        self.stop_date = ppdata.get('stop_date',None)

        # read amr, geometry, and erf inputs
        self.amr = AMRParms(**parmparse('amr',ppdata))
        self.geometry = GeometryParms(**parmparse('geometry',ppdata))
        self.erf = ERFParms(**parmparse('erf',ppdata))

        self.read_bcs(ppdata)
        self.read_refinement(ppdata)

        # read problem-specific inputs
        self.prob = parmparse('prob',ppdata)

        self.validate()

    def parse_input(self,fpath):
        pp = {}
        with open(fpath,'r') as f:
            prevline = ''
            for line in f:
                line = line.strip()

                # strip comments
                try:
                    trailingcomment = line.index('#')
                except ValueError:
                    pass
                else:
                    line = line[:trailingcomment].rstrip()
                if line == '':
                    continue

                # allow multiline
                if line.endswith('\\'):
                    prevline += ' ' + line
                    continue
                elif prevline != '':
                    line = prevline + ' ' + line
                    prevline = ''

                # finally parse
                split = line.split('=')
                key = split[0].rstrip()
                val = split[1].lstrip()
                vals = val.split()
                if len(vals) == 1:
                    val = vals[0]
                    val = val.strip('"').strip("'")
                    pp[key] = val
                else:
                    pp[key] = vals
        return pp

    def read_bcs(self,ppdata):
        if not self.geometry.is_periodic[0]:
            assert 'xlo.type' in ppdata.keys()
            assert 'xhi.type' in ppdata.keys()
            self.xlo = parmparse('xlo',ppdata)
            self.xhi = parmparse('xhi',ppdata)
        if not self.geometry.is_periodic[1]:
            assert 'ylo.type' in ppdata.keys()
            assert 'yhi.type' in ppdata.keys()
            self.ylo = parmparse('ylo',ppdata)
            self.yhi = parmparse('yhi',ppdata)
        if not self.geometry.is_periodic[2]:
            assert 'zlo.type' in ppdata.keys()
            assert 'zhi.type' in ppdata.keys()
            self.zlo = parmparse('zlo',ppdata)
            self.zhi = parmparse('zhi',ppdata)
            if self.zlo['type'] == 'MOST':
                self.most = parmparse('erf.most',ppdata)

    def read_refinement(self,ppdata):
        self.refine = {}
        for box in self.erf.refinement_indicators:
            self.refine[box] = parmparse(f'erf.{box}',ppdata)
            assert 'max_level' in self.refine[box]
            self.refine[box]['max_level'] = int(self.refine[box]['max_level'])
            assert self.refine[box]['max_level'] > 0
            if 'in_box_lo' in self.refine[box]:
                # static or dynamic refinement
                assert 'in_box_hi' in self.refine[box]
            else:
                # dynamic refinement
                assert 'field_name' in self.refine[box]
            if 'field_name' in self.refine[box]:
                tests = [parm for parm in self.refine[box]
                         if parm not in ['max_level','field_name',
                                         'in_box_lo','in_box_hi']]
                assert len(tests) >= 1

    def validate(self):
        # additional validation that depends on different parmparse types
        if self.erf.terrain_type.lower() != 'none':
            if self.erf.terrain_z_levels:
                nz = self.amr.n_cell[2]
                assert len(self.erf.terrain_z_levels) == nz+1

        for box,refineparams in self.refine.items():
            max_level = refineparams['max_level']
            if max_level > self.amr.max_level:
                print(f'Note: {box} refinement (max_level={max_level})'
                      f' will be inactive with amr.max_level={self.amr.max_level}')

    def write(self,fpath=None):
        with open_file_or_stdout(fpath) as f:
            f.write('# ------------------------------- INPUTS TO ERF -------------------------------\n')
            f.write('# written by erftools.inputs (https://github.com/erf-model/erftools)\n')
            if self.start_date and self.stop_date:
                f.write(f"""
max_step       = {self.max_step}
start_datetime = {self.start_date}  # epoch time: {self.start_time} s
stop_datetime  = {self.stop_date}  # epoch time: {self.stop_time} s
""")
            else:
                f.write(f"""
max_step    = {self.max_step}
start_time  = {self.start_time}
stop_time   = {self.stop_time}
""")
            if self.erf.restart:
                f.write(f"""
erf.restart = {self.erf.restart}
""")

            if self.erf.anelastic:
                f.write(f"""
erf.anelastic = {bool_to_str(self.erf.anelastic)}
erf.use_fft   = {bool_to_str(self.erf.use_fft)}
erf.mg_v      = {bool_to_str(self.erf.mg_v)}
""")

            ########################################
            f.write('\n# PROBLEM SIZE & GEOMETRY')
            if self.geometry.prob_lo == (0,0,0):
                f.write(f"""
geometry.prob_extent = {list_to_str(self.geometry.prob_extent)}
amr.n_cell           = {list_to_str(self.amr.n_cell)}

geometry.is_periodic = {list_to_str(self.geometry.is_periodic)}
""")
            else:
                f.write(f"""
geometry.prob_lo = {list_to_str(self.geometry.prob_lo)}
geometry.prob_hi = {list_to_str(self.geometry.prob_hi)}
amr.n_cell       = {list_to_str(self.amr.n_cell)}

geometry.is_periodic = {list_to_str(self.geometry.is_periodic)}
""")
            if hasattr(self,'xlo'):
                f.write('\n')
                write_bc(f,'xlo',self.xlo)
                write_bc(f,'xhi',self.xhi)
            if hasattr(self,'ylo'):
                f.write('\n')
                write_bc(f,'ylo',self.ylo)
                write_bc(f,'yhi',self.yhi)
            if hasattr(self,'zlo'):
                f.write('\n')
                write_bc(f,'zlo',self.zlo)
                if self.zlo['type'] == 'MOST':
                    for key,val in self.most.items():
                        f.write(f'erf.most.{key} = {val}\n')
                write_bc(f,'zhi',self.zhi)

            ########################################
            f.write('\n# TIME STEP CONTROL\n')
            if self.erf.fixed_dt > 0:
                f.write(f'erf.fixed_dt        = {self.erf.fixed_dt}\n')
            else:
                f.write(f'erf.cfl             = {self.erf.cfl}\n')
            if self.erf.no_substepping == 1:
                f.write('erf.no_substepping  = 1\n')
            else:
                if self.erf.fixed_fast_dt > 0:
                    f.write(f'erf.fixed_fast_dt   = {self.erf.fixed_fast_dt}\n')
                elif self.erf.fixed_mri_dt_ratio > 0:
                    f.write('erf.fixed_mri_dt_ratio = '
                            f'{self.erf.fixed_mri_dt_ratio}\n')
                else:
                    f.write('erf.substepping_cfl = '
                            f'{self.erf.substepping_cfl}\n')

            ########################################
            f.write(f"""\n# DIAGNOSTICS & VERBOSITY
amr.v            = {self.amr.v}  # verbosity in Amr.cpp
erf.v            = {self.erf.v}  # verbosity in ERF.cpp
erf.sum_interval = {self.erf.sum_interval}  # timesteps between computing mass
""")

            ########################################
            f.write('\n# REFINEMENT / REGRIDDING\n')
            f.write(f'amr.max_level = {self.amr.max_level}\n')
            if len(self.erf.refinement_indicators) > 0:
                if len(self.amr.ref_ratio_vect) > 0:
                    f.write('amr.ref_ratio_vect = '
                            f'{list_to_str(self.amr.ref_ratio_vect)}\n')
                elif isinstance(self.amr.ref_ratio,list):
                    f.write('amr.ref_ratio = '
                            f'{list_to_str(self.amr.ref_ratio)}\n')
                else:
                    f.write(f'amr.ref_ratio = {self.amr.ref_ratio}\n')
                f.write('\nerf.refinement_indicators = '
                        f'{strs_to_str(self.erf.refinement_indicators)}\n')
            for box in self.erf.refinement_indicators:
                boxdict = self.refine[box]
                f.write('\n')
                if 'max_level' in boxdict.keys():
                    f.write(f"erf.{box}.max_level = {boxdict['max_level']}\n")
                f.write(f'erf.{box}.in_box_lo = '
                        f"{list_to_str(boxdict['in_box_lo'],float)}\n")
                f.write(f'erf.{box}.in_box_hi = '
                        f"{list_to_str(boxdict['in_box_hi'],float)}\n")

            ########################################
            if self.erf.grid_stretching_ratio > 1 or self.erf.have_terrain():
                f.write('\n# GRID STRETCHING / TERRAIN')
            if self.erf.grid_stretching_ratio > 1:
                f.write(f"""
erf.grid_stretching_ratio = {self.erf.grid_stretching_ratio}
erf.initial_dz            = {self.erf.initial_dz}
""")
            if self.erf.have_terrain():
                f.write(f"""
erf.terrain_type      = {self.erf.terrain_type}
erf.terrain_smoothing = {self.erf.terrain_smoothing}
""")
                if self.erf.terrain_z_levels:
                    f.write(f'erf.terrain_z_levels = {list_to_str(self.erf.terrain_z_levels)}\n')
                if len(self.erf.terrain_file_name) > 0:
                    f.write('erf.terrain_file_name = '
                            f'{self.erf.terrain_file_name}\n')

            ########################################
            f.write('\n# CHECKPOINT FILES\n')
            f.write(f'erf.check_file = {self.erf.check_file}\n')
            if self.erf.check_per > 0:
                f.write(f'erf.check_per  = {self.erf.check_per}\n')
            else:
                f.write(f'erf.check_int  = {self.erf.check_int}\n')

            ########################################
            f.write('\n# PLOTFILES\n')
            if self.erf.plotfile_type != 'amrex':
                f.write(f'erf.plotfile_type = {self.erf.plotfile_type}\n')
            if (self.erf.plot_int_1 > 0) or (self.erf.plot_per_1 > 0):
                f.write(f'erf.plot_file_1 = {self.erf.plot_file_1}\n')
                if self.erf.plot_per_1 > 0:
                    f.write(f'erf.plot_per_1  = {self.erf.plot_per_1}\n')
                else:
                    f.write(f'erf.plot_int_1  = {self.erf.plot_int_1}\n')
                f.write('erf.plot_vars_1 = '
                        f"{strs_to_str(self.erf.plot_vars_1)}\n")
            if (self.erf.plot_int_2 > 0) or (self.erf.plot_per_2 > 0):
                f.write(f'erf.plot_file_2 = {self.erf.plot_file_2}\n')
                if self.erf.plot_per_2 > 0:
                    f.write(f'erf.plot_per_2  = {self.erf.plot_per_2}\n')
                else:
                    f.write(f'erf.plot_int_2  = {self.erf.plot_int_2}\n')
                f.write('erf.plot_vars_2 = '
                        f"{strs_to_str(self.erf.plot_vars_2)}\n")

            ########################################
            if len(self.erf.data_log) > 0:
                f.write('\n# DATA COLLECTION\n')
                f.write(f"erf.data_log        = {strs_to_str(self.erf.data_log)}\n")
                f.write(f'erf.profile_int     = {self.erf.profile_int}\n')
                f.write(f'erf.destag_profiles = {bool_to_str(self.erf.destag_profiles)}\n')

            # tslist-like sampling
            if len(self.erf.sample_line_lo) > 0:
                nlines = len(self.erf.sample_line_lo) // 3
                assert len(self.erf.sample_line_lo) == 3*nlines, \
                        'Unexpected number of values sampling indices'
                f.write(f"""
erf.do_line_sampling = true
erf.line_sampling_vars = x_velocity y_velocity z_velocity theta qv pressure
erf.line_sampling_text_output = true
erf.sampler_interval = 1
erf.sample_line_lo   = {list_to_str(self.erf.sample_line_lo)}
erf.sample_line_hi   = {list_to_str(self.erf.sample_line_hi)}
erf.sample_line_dir  = {list_to_str(self.erf.sample_line_dir)}
""")

            ########################################
            f.write('\n# ADVECTION SCHEMES\n')
            for vartype in ['dycore','dryscal','moistscal']:
                for advdir in ['horiz','vert']:
                    advinp = f'{vartype}_{advdir}_adv_type'
                    advscheme = getattr(self.erf, advinp)
                    f.write(f'erf.{advinp} = {advscheme}\n')
                    if advscheme.startswith('Blended'):
                        upwinp = f'{vartype}_{advdir}_upw_frac'
                        upwinding = getattr(self.erf, upwinp)
                        f.write(f'erf.{upwinp} = {upwinding}\n')
            if self.erf.use_efficient_advection:
                f.write('erf.use_efficient_advection = true\n')
            if self.erf.use_mono_advection:
                f.write('erf.use_mono_advection = true\n')

            ########################################
            f.write('\n# DIFFUSIVE PHYSICS')
            if self.erf.molec_diff_type.startswith('Constant'):
                f.write(f"""
erf.molec_diff_type   = {self.erf.molec_diff_type}
erf.dynamic_viscosity = {self.erf.dynamic_viscosity}
erf.alpha_T           = {self.erf.alpha_T}
erf.alpha_C           = {self.erf.alpha_C}
""")
            lestypes = self.erf.les_type \
                    if isinstance(self.erf.les_type,list) \
                    else [self.erf.les_type]
            have_smag = ('Smagorinsky' in lestypes)
            have_dear = ('Deardorff' in lestypes)
            if have_smag or have_dear:
                f.write(f'\nerf.les_type  = {strs_to_str(lestypes)}\n')
            if have_smag:
                Cs_list = self.erf.Cs if isinstance(self.erf.Cs,list) else [self.erf.Cs]
                f.write(f'erf.Cs        = {list_to_str(Cs_list)}\n')
            if have_dear:
                Ck_list = self.erf.Ck if isinstance(self.erf.Ck,list) else [self.erf.Ck]
                f.write(f'erf.Ck        = {list_to_str(Ck_list)}\n')
                f.write(f'erf.Ce        = {self.erf.Ce}\n')
                f.write(f'erf.Ce_wall   = {self.erf.Ce_wall}\n')
                f.write(f'erf.sigma_k   = {self.erf.sigma_k}\n')
                f.write(f'erf.theta_ref = {self.erf.theta_ref}\n')
            if have_smag or have_dear:
                f.write(f'erf.Pr_t      = {self.erf.Pr_t}\n')
                f.write(f'erf.Sc_t      = {self.erf.Sc_t}\n')

            pbltypes = self.erf.pbl_type \
                    if isinstance(self.erf.pbl_type,list) \
                    else [self.erf.pbl_type]
            if any([pblscheme != 'None' for pblscheme in pbltypes]):
                f.write(f'\nerf.pbl_type = {strs_to_str(pbltypes)}\n')
            if 'MYNN25' in pbltypes:
                for key,val in self.inpdict.items():
                    if key.startswith('erf.pbl_mynn') or ('QKE' in key):
                        f.write(f'{key} = {float(val):g}\n')
            if 'YSU' in pbltypes:
                for key,val in self.inpdict.items():
                    if key.startswith('erf.pbl_ysu'):
                        f.write(f'{key} = {float(val):g}\n')

            ########################################
            if self.erf.moisture_model != 'None':
                f.write(f"""\n# MOISTURE
erf.moisture_model = {self.erf.moisture_model}
""")

            ########################################
            if self.erf.radiation_model != 'None':
                f.write(f"""\n# RADIATION
erf.radiation_model = {self.erf.radiation_model}
""")

            ########################################
            f.write(f"""\n# FORCING TERMS
erf.use_gravity = {bool_to_str(self.erf.use_gravity)}
erf.use_coriolis = {bool_to_str(self.erf.use_coriolis)}
""")
            if self.erf.use_coriolis:
                f.write(f'erf.coriolis_3d = {bool_to_str(self.erf.coriolis_3d)}\n')
                if self.erf.latitude != 90.:
                    f.write(f'erf.latitude = {self.erf.latitude}\n')
                if self.erf.rotational_time_period != 86400.:
                    f.write(f'erf.rotational_time_period = {self.erf.rotational_time_period}\n')

            if (self.erf.abl_driver_type == 'GeostrophicWind') \
                    and (len(self.erf.abl_geo_wind_table) > 0):
                f.write(f"""
erf.abl_driver_type    = GeostrophicWind
erf.abl_geo_wind_table = {self.erf.abl_geo_wind_table}
""")
            elif (self.erf.abl_driver_type == 'GeostrophicWind') \
                    and (self.erf.abl_geo_wind != (0,0,0)):
                f.write(f"""
erf.abl_driver_type = GeostrophicWind
erf.abl_geo_wind    = {list_to_str(self.erf.abl_geo_wind)}
""")
                if len(self.erf.abl_geo_wind_table) > 0:
                    f.write(f'erf.abl_geo_wind_table = {self.erf.latitude}\n')
            elif (self.erf.abl_driver_type == 'PressureGradient') \
                    and (self.erf.abl_pressure_grad != (0,0,0)):
                f.write(f"""
erf.abl_driver_type   = PressureGradient
erf.abl_pressure_grad = {list_to_str(self.erf.abl_pressure_grad)}
""")

            have_rayleigh = (  self.erf.rayleigh_damp_U \
                            or self.erf.rayleigh_damp_V \
                            or self.erf.rayleigh_damp_W \
                            or self.erf.rayleigh_damp_T)
            if have_rayleigh:
                f.write('\n')
            if self.erf.rayleigh_damp_U:
                f.write(f'erf.rayleigh_damp_U   = {bool_to_str(self.erf.rayleigh_damp_U)}\n')
            if self.erf.rayleigh_damp_V:
                f.write(f'erf.rayleigh_damp_V   = {bool_to_str(self.erf.rayleigh_damp_V)}\n')
            if self.erf.rayleigh_damp_W:
                f.write(f'erf.rayleigh_damp_W   = {bool_to_str(self.erf.rayleigh_damp_W)}\n')
            if self.erf.rayleigh_damp_T:
                f.write(f'erf.rayleigh_damp_T   = {bool_to_str(self.erf.rayleigh_damp_T)}\n')
            if have_rayleigh:
                f.write(f'erf.rayleigh_zdamp    = {self.erf.rayleigh_zdamp}\n')
                f.write(f'erf.rayleigh_dampcoef = {self.erf.rayleigh_dampcoef}\n')

            if self.erf.nudging_from_input_sounding:
                f.write(f"""
erf.nudging_from_input_sounding = true
erf.input_sounding_file = {strs_to_str(self.erf.input_sounding_file)}
erf.input_sounding_time = {list_to_str(self.erf.input_sounding_time)}
""")
            elif isinstance(self.erf.input_sounding_file, str):
                f.write(f'\nerf.input_sounding_file = {self.erf.input_sounding_file}\n')

            ########################################
            f.write('\n# INITIALIZATION')
            if self.erf.init_type.lower() == 'input_sounding':
                f.write(f"""
erf.init_type           = input_sounding
erf.init_sounding_ideal = {bool_to_str(self.erf.init_sounding_ideal)}
""")
            elif self.erf.init_type.lower() == 'wrfinput':
                f.write(f"""
erf.init_type      = wrfinput
erf.use_real_bcs   = {bool_to_str(self.erf.use_real_bcs)}
erf.nc_init_file_0 = {self.erf.nc_init_file_0}""")
                if self.erf.use_real_bcs:
                    f.write(f"""
erf.nc_bdy_file    = {self.erf.nc_bdy_file}
erf.real_width     = {self.erf.real_width}
erf.real_set_width = {self.erf.real_set_width}
""")
                else:
                    f.write('\n')
            elif self.erf.init_type.lower() == 'metgrid':
                f.write(f"""
erf.init_type      = metgrid
erf.nc_init_file_0 = {strs_to_str(self.erf.nc_init_file_0)}
""")
                for key,val in self.inpdict.items():
                    if key.startswith('erf.metgrid'):
                        try:
                            float(val)
                        except ValueError:
                            f.write(f'{key} = {bool_to_str(val)}\n')
                        else:
                            f.write(f'{key} = {float(val):g}\n')
            else:
                f.write(f'\nerf.init_type = {self.erf.init_type}\n')


def write_bc(f, bcname, bcdict):
    f.write(f"{bcname}.type = {bcdict['type']}\n")
    for key,val in bcdict.items():
        if key=='type':
            continue
        if isinstance(val, str):
            f.write(f"{bcname}.{key} = {val}\n")
        elif isinstance(val, float):
            f.write(f"{bcname}.{key} = {float(val):g}\n")
        else:
            assert isinstance(val, list)
            f.write(f"{bcname}.{key} = {list_to_str(val,float)}\n")
