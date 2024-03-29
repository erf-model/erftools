"""
Processing for each namelist within a WRF namelist.input file
"""
from datetime import datetime

class WRFNamelist(object):
    def __init__(self,nmldict):
        self.nml = nmldict

    def getvar(self,varname,optional=False,default=None):
        if default is not None:
            optional = True
        try:
            val = self.nml[varname]
        except KeyError:
            if optional:
                return default
            else:
                raise KeyError(varname)
        assert not hasattr(val,'__iter__')
        assert isinstance(val, (int, float, bool))
        return val

    def getarrayvar(self,varname,optional=False,default=None):
        if default is not None:
            optional = True
        try:
            arr = self.nml[varname]
        except KeyError:
            if optional:
                arr = default
            else:
                raise KeyError(varname)
        if not hasattr(arr,'__iter__'):
            arr = [arr]
        for i,val in enumerate(arr):
            assert isinstance(val, (int, float, bool)), \
                    f'{varname} value {i} is invalid'
        return arr

class TimeControl(WRFNamelist):
    """&time_control namelist"""

    def __init__(self,nmldict):
        super().__init__(nmldict)
        self.restart_interval = self.getvar('restart_interval') # [min]
        self.parse_datetime_range()

    def __str__(self):
        return f"""WRF `time_control` parameters
  sim range: {self.start_datetime} to {self.end_datetime}"""

    def parse_datetime_range(self):
        # assume all domains have the same start/end datetime for now; TODO: allow for variable start times per domain
        idom = 0
        start_year  = self.getarrayvar('start_year')[idom]
        start_month = self.getarrayvar('start_month')[idom]
        start_day   = self.getarrayvar('start_day')[idom]
        start_hour  = self.getarrayvar('start_hour')[idom]
        end_year    = self.getarrayvar('end_year')[idom]
        end_month   = self.getarrayvar('end_month')[idom]
        end_day     = self.getarrayvar('end_day')[idom]
        end_hour    = self.getarrayvar('end_hour')[idom]
        self.start_datetime = datetime(start_year, start_month, start_day, start_hour)
        self.end_datetime = datetime(end_year, end_month, end_day, end_hour)
        
    
class Domains(WRFNamelist):
    """&domains namelist"""

    def __init__(self,nmldict):
        super().__init__(nmldict)
        self.parse_time_integration()
        self.parse_grid()

    def __str__(self):
        s = 'WRF `domains` parameters\n'
        for dom in range(self.max_dom):
            s += f'  d{dom+1:02d}:' \
                 f' [(1,{self.e_we[dom]:d}),(1,{self.e_sn[dom]:d}),(1,{self.e_vert[dom]:d})]' \
                 f' ds=[{self.dx[dom]},{self.dy[dom]}]\n'
        return s.rstrip()

    def parse_time_integration(self):
        self.time_step = self.getvar('time_step') # seconds
        self.parent_time_step_ratio = self.getarrayvar(
                'parent_time_step_ratio', default=1)

    def parse_grid(self):
        self.max_dom = self.getvar('max_dom')
        if self.parent_time_step_ratio is None:
            assert self.max_dom == 1
        # - assume s_we == s_sn == s_vert == [1, 1, ...]
        self.e_we = self.getarrayvar('e_we')[:self.max_dom] # west--east, unstaggered
        self.e_sn = self.getarrayvar('e_sn')[:self.max_dom] # south--north, unstaggered
        self.e_vert = self.getarrayvar('e_vert')[:self.max_dom] # bottom--top, STAGGERED
        self.dx = self.getarrayvar('dx')[:self.max_dom]
        self.dy = self.getarrayvar('dy')[:self.max_dom]
        self.ztop = self.getvar('ztop', optional=True)
        self.p_top_requested = self.getvar('p_top_requested', default=5000.)
        self.i_parent_start = self.getarrayvar('i_parent_start', default=0)
        self.j_parent_start = self.getarrayvar('j_parent_start', default=0)
        self.parent_grid_ratio = self.getarrayvar(
                'parent_grid_ratio', default=1)
        if self.max_dom > 1:
            self.i_parent_start = self.i_parent_start[:self.max_dom]
            self.j_parent_start = self.j_parent_start[:self.max_dom]
            self.parent_grid_ratio = self.parent_grid_ratio[:self.max_dom]
            assert self.i_parent_start[0] == 1
            assert self.j_parent_start[0] == 1
        for dom in range(1,self.max_dom):
            assert (self.dx[dom-1]/self.dx[dom] == self.parent_grid_ratio[dom])
            assert (self.dy[dom-1]/self.dy[dom] == self.parent_grid_ratio[dom])


pbl_mapping = {
    0:  'None',
    1:  'YSU',
    2:  'MYJ',
    3:  'GFS',
    4:  'QNSE',
    5:  'MYNN2.5',
    6:  'MYNN3',
    7:  'ACM2',
    8:  'BouLac',
    9:  'UW',
    10: 'TEMF',
    11: 'Shin-Hong',
    12: 'GBM',
    99: 'MRF',
}
sfclay_mapping = {
    0: 'none',
    1: 'MOST', # w/ Carslon-Boland viscous sub-layer and standard similarity function lookup
    2: 'Eta', # Janjic Eta similarity, w/ Zilitinkevich thermal roughness length, standard similarity function lookup
    5: 'MYNN',
}
valid_sfclay = {
    # for each PBL scheme
    0:  list(range(100)),
    1:  [1],
    2:  [2],
    5:  [1,2,5,91],
    6:  [1,2,5,91],
    7:  [1,7,91],
    8:  [1,2,91],
    9:  [1,2,91],
    11: [1,91],
    12: [1,91],
    91: [1,91],
}


class Physics(WRFNamelist):
    """&physics namelist"""

    def __init__(self,nmldict):
        super().__init__(nmldict)
        self.parse_all()

    def __str__(self):
        s = 'WRF `physics` parameters\n'
        for dom in range(len(self.bl_pbl_physics)):
            s+=f'  d0{dom+1:d}: {self.bl_pbl_physics[dom]} PBL, {self.sf_sfclay_physics[dom]} surface layer\n'
        return s.rstrip()

    def parse_all(self):
        pbl_idx_list = self.getarrayvar('bl_pbl_physics')
        sfclay_idx_list = self.getarrayvar('sf_sfclay_physics')
        for pbl_idx,sfclay_idx in zip(pbl_idx_list, sfclay_idx_list):
            if sfclay_idx not in valid_sfclay[pbl_idx]:
                print(f'WARNING: unexpected pairing of bl_pbl_physics={pbl_idx} with sf_sfclay_idx={sfclay_idx}')
        self.bl_pbl_physics = [pbl_mapping.get(idx,'UNKNOWN') for idx in pbl_idx_list]
        self.sf_sfclay_physics = [sfclay_mapping.get(idx,'UNKNOWN') for idx in sfclay_idx_list]
        self.num_land_cat = self.getvar('num_land_cat', optional=True)


diff_opt_mapping = {
    1: 'simple',
    2: 'full',
}
km_opt_mapping = {
    1: 'constant',
    2: 'Deardorff', # 3D TKE
    3: 'Smagorinsky', # 3D deformation
    4: '2D deformation', # horizontal diffusion diagnosed from horizontal
                         #   deformation, vertical diffusion from PBL scheme
}
damp_opt_mapping = {
    0: 'none',
    1: 'increased diffusion',
    2: 'Rayleigh relaxation',
    3: 'Rayleigh implicit gravity-wave damping',
}

class Dynamics(WRFNamelist):
    """&dynamics namelist"""

    def __init__(self,nmldict):
        super().__init__(nmldict)
        self.parse_diffusion()
        self.parse_damping()

    def __str__(self):
        s = 'WRF `dynamics` parameters\n'
        for dom in range(len(self.diff_opt)):
            s+=f'  d0{dom+1:d}: diffusion={self.diff_opt[dom]}, K option={self.km_opt[dom]}\n'
        s+=f'  upper damping: {self.damp_opt}\n'
        s+=f'  vertical damping: {self.w_damping}\n'
        return s.rstrip()

    def parse_diffusion(self):
        diff_opt_list = self.getarrayvar('diff_opt')
        km_opt_list = self.getarrayvar('km_opt')
        self.diff_opt = [diff_opt_mapping.get(diff_opt_list[dom], 'UNKNOWN')
                         for dom in range(len(diff_opt_list))]
        self.km_opt = [km_opt_mapping.get(km_opt_list[dom], 'UNKNOWN')
                       for dom in range(len(diff_opt_list))]
        self.khdif = self.getvar('khdif', optional=True)
        self.kvdif = self.getvar('kvdif', optional=True)
        self.diff_6th_opt = self.getarrayvar(
                'diff_6th_opt', default=0)
        self.diff_6th_factor = self.getarrayvar(
                'diff_6th_factor', default=0.12)

    def parse_damping(self):
        self.damp_opt = damp_opt_mapping[self.getvar('damp_opt')]
        self.w_damping = bool(self.getvar('w_damping', default=0))
        self.zdamp = self.getvar('zdamp')
        self.dampcoef = self.getvar('dampcoef')

class BoundaryControl(WRFNamelist):
    """&bdy_control namelist"""

    def __init__(self,nmldict):
        super().__init__(nmldict)
        self.parse()

    def parse(self):
        self.periodic_x = bool(self.getvar('periodic_x', default=False))
        self.periodic_y = bool(self.getvar('periodic_y', default=False))

