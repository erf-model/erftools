"""
Processing for each namelist within a WRF namelist.input file
"""
import warnings
from datetime import datetime, timedelta

from .mappings import *

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
        if arr is None:
            return None
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

        self.restart_interval = self.getvar('restart_interval',default=1440) # [min]
        restart_interval_s = self.getvar('restart_interval_s',optional=True)
        if restart_interval_s is not None:
            self.restart_interval = restart_interval_s / 60.

        self.history_interval = self.getarrayvar('history_interval',default=[60]) # [min]
        history_interval_s = self.getarrayvar('history_interval_s',optional=True)
        if history_interval_s is not None:
            self.history_interval = [interval / 60. for interval in history_interval_s]

        self.auxhist2_interval = self.getarrayvar('auxhist2_interval',default=[0]) # [min]

        self.parse_datetime_range()

    def __str__(self):
        return f"""WRF `time_control` parameters
  sim range: {self.start_datetime} to {self.end_datetime}"""

    def parse_datetime_range(self):
        start_year   = self.getarrayvar('start_year')
        start_month  = self.getarrayvar('start_month')
        start_day    = self.getarrayvar('start_day')
        start_hour   = self.getarrayvar('start_hour')
        start_minute = self.getarrayvar('start_minute', optional=True)
        start_second = self.getarrayvar('start_second', optional=True)
        end_year     = self.getarrayvar('end_year')
        end_month    = self.getarrayvar('end_month')
        end_day      = self.getarrayvar('end_day')
        end_hour     = self.getarrayvar('end_hour')
        end_minute   = self.getarrayvar('end_minute', optional=True)
        end_second   = self.getarrayvar('end_second', optional=True)
        if start_minute is None:
            start_minute = len(start_year) * [0]
        if start_second is None:
            start_second = len(start_year) * [0]
        if end_minute is None:
            end_minute = len(start_year) * [0]
        if end_second is None:
            end_second = len(start_year) * [0]

        self.start_datetimes = []
        self.end_datetimes = []
        for idom in range(len(start_year)):
            try:
                start_date = datetime(start_year[idom],
                                      start_month[idom],
                                      start_day[idom],
                                      start_hour[idom],
                                      start_minute[idom],
                                      start_second[idom])
            except ValueError:
                print(f'Problem parsing start date on level {idom}')
                break
            try:
                end_date = datetime(end_year[idom],
                                    end_month[idom],
                                    end_day[idom],
                                    end_hour[idom],
                                    end_minute[idom],
                                    end_second[idom])
            except ValueError:
                print(f'Problem parsing end date on level {idom}')
                break
            self.start_datetimes.append(start_date)
            self.end_datetimes.append(end_date)
        
        run_days = self.getvar('run_days', default=0)
        run_hours = self.getvar('run_hours', default=0)
        run_minutes = self.getvar('run_minutes', default=0)
        run_seconds = self.getvar('run_seconds', default=0)
        spec_run_time = run_days    * 86400 \
                      + run_hours   * 3600 \
                      + run_minutes * 60 \
                      + run_seconds
        if spec_run_time > 0:
            simtime = self.end_datetimes[0] - self.start_datetimes[0]
            if spec_run_time != simtime.total_seconds():
                new_end_datetime = self.start_datetimes[0] \
                                 + timedelta(seconds=spec_run_time)
                # From WRF user guide: run_days/run_hours takes precedence
                # over the end times
                print(f'Specified run time {spec_run_time} s'
                      f' differs from {simtime.total_seconds()} s' \
                      f' (from {self.start_datetimes[0]}'
                      f' to {self.end_datetimes[0]})'
                      f' -- updated end_datetimes[0]={new_end_datetime}')
                self.end_datetimes[0] = new_end_datetime


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
        dt_num = self.getvar('time_step_fract_num',optional=True)
        if dt_num is not None and dt_num > 0:
            dt_den = self.getvar('time_step_fract_den')
            self.time_step = dt_num / dt_den
        self.parent_time_step_ratio = self.getarrayvar(
                'parent_time_step_ratio', default=1)

    def parse_grid(self):
        self.max_dom = self.getvar('max_dom')
        if self.parent_time_step_ratio is None:
            assert self.max_dom == 1
        ones = self.max_dom*[1]
        self.s_we = self.getarrayvar('s_we',default=ones)[:self.max_dom]
        self.s_sn = self.getarrayvar('s_sn',default=ones)[:self.max_dom]
        self.s_vert = self.getarrayvar('s_vert',default=ones)[:self.max_dom]
        self.e_we = self.getarrayvar('e_we')[:self.max_dom]
        self.e_sn = self.getarrayvar('e_sn')[:self.max_dom]
        self.e_vert = self.getarrayvar('e_vert')[:self.max_dom]
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
            if len(self.dx) >= self.max_dom:
                assert (self.dx[dom-1]/self.dx[dom] == self.parent_grid_ratio[dom])
            else:
                print(f'Note: dx on d{dom+1:02d} not found,'
                      ' setting from parent_grid_ratio')
                self.dx.append(self.dx[dom-1] / self.parent_grid_ratio[dom])
            if len(self.dy) >= self.max_dom:
                assert (self.dy[dom-1]/self.dy[dom] == self.parent_grid_ratio[dom])
            else:
                print(f'Note: dy on d{dom+1:02d} not found'
                      ' setting from parent_grid_ratio')
                self.dy.append(self.dy[dom-1] / self.parent_grid_ratio[dom])
        self.eta_levels = self.getarrayvar('eta_levels',optional=True)
        self.etac = self.getvar('etac',default=0.2)
        self.auto_levels_opt = self.getvar('auto_levels_opt',default=2)
        # for auto_levels_opt==2
        self.dzbot = self.getvar('dzbot',default=50.)
        self.dzstretch_s = self.getvar('dzstretch_s',default=1.3)
        self.dzstretch_u = self.getvar('dzstretch_s',default=1.1)
        self.max_dz = self.getvar('max_dz',default=1000.)


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
        pbl_mynn_closure = self.getvar('bl_mynn_closure', default=2.6)
        sfclay_idx_list = self.getarrayvar('sf_sfclay_physics')
        for pbl_idx,sfclay_idx in zip(pbl_idx_list, sfclay_idx_list):
            if sfclay_idx not in valid_sfclay[pbl_idx]:
                warnings.warn(f'Unexpected pairing of bl_pbl_physics={pbl_idx}'
                              f' with sf_sfclay_idx={sfclay_idx}')
        self.bl_pbl_physics = [pbl_mapping.get(idx,'UNKNOWN') for idx in pbl_idx_list]
        for i in range(len(self.bl_pbl_physics)):
            if self.bl_pbl_physics[i] == 'MYNN':
                if pbl_mynn_closure > 2.5:
                    pbl_mynn_closure = 2.5
                    warnings.warn(f'MYNN level {pbl_mynn_closure} selected, but'
                                  ' it is currently not available in ERF'
                                  ' -- reverting to MYNN level 2.5.'
                                  ' Note: A port of MYNN-EDMF is also available'
                                  ' but it has not been fully verified.')
                lvlstr = str(pbl_mynn_closure).replace('.','')
                self.bl_pbl_physics[i] += lvlstr
        self.sf_sfclay_physics = [sfclay_mapping.get(idx,'UNKNOWN') for idx in sfclay_idx_list]
        self.num_land_cat = self.getvar('num_land_cat', optional=True)

        mp_idx_list = self.getarrayvar('mp_physics')
        self.mp_physics = [mp_physics_mapping.get(idx,'UNKNOWN') for idx in mp_idx_list]

        ra_lw_idx_list = self.getarrayvar('ra_lw_physics')
        ra_sw_idx_list = self.getarrayvar('ra_sw_physics')
        assert all([lw==sw for lw,sw in zip(ra_lw_idx_list,ra_sw_idx_list)]), \
                'Different longwave/shortwave radiation schemes not handled'
        self.ra_physics = [ra_physics_mapping.get(idx,'UNKNOWN') for idx in ra_lw_idx_list]

        self.radt = self.getarrayvar('radt', default=-1)
        assert all([t==self.radt[0] for t in self.radt]), \
                'Different radiation call intervals not handled'

        lsm_idx_list = self.getarrayvar('sf_surface_physics')
        self.surface_physics = [surface_physics_mapping.get(idx,'UNKNOWN') for idx in lsm_idx_list]

        cu_idx_list = self.getarrayvar('cu_physics')
        self.cu_physics = [cu_physics_mapping.get(idx,'UNKNOWN') for idx in cu_idx_list]


class Dynamics(WRFNamelist):
    """&dynamics namelist"""

    def __init__(self,nmldict):
        super().__init__(nmldict)
        self.parse_advection()
        self.parse_diffusion()
        self.parse_damping()

    def __str__(self):
        s = 'WRF `dynamics` parameters\n'
        for dom in range(len(self.diff_opt)):
            s+=f'  d0{dom+1:d}: diffusion={self.diff_opt[dom]}, K option={self.km_opt[dom]}\n'
        s+=f'  upper damping: {self.damp_opt}\n'
        s+=f'  vertical damping: {self.w_damping}\n'
        return s.rstrip()

    def parse_advection(self):
        h_mom_adv_list = self.getarrayvar('h_mom_adv_order', default=5)
        v_mom_adv_list = self.getarrayvar('v_mom_adv_order', default=3)
        h_sca_adv_list = self.getarrayvar('h_sca_adv_order', default=5)
        v_sca_adv_list = self.getarrayvar('v_sca_adv_order', default=3)
        moist_adv_list = self.getarrayvar('moist_adv_opt', default=1)
        ndom = len(h_mom_adv_list)
        self.h_mom_adv_order = [adv_mapping.get(h_mom_adv_list[dom],' UNKNOWN')
                                for dom in range(ndom)]
        self.v_mom_adv_order = [adv_mapping.get(v_mom_adv_list[dom],' UNKNOWN')
                                for dom in range(ndom)]
        self.h_sca_adv_order = [adv_mapping.get(h_sca_adv_list[dom],' UNKNOWN')
                                for dom in range(ndom)]
        self.v_sca_adv_order = [adv_mapping.get(v_sca_adv_list[dom],' UNKNOWN')
                                for dom in range(ndom)]
        self.moist_adv_opt = [moist_adv_mapping.get(moist_adv_list[dom], 'UNKNOWN')
                              for dom in range(ndom)]

    def parse_diffusion(self):
        diff_opt_list = self.getarrayvar('diff_opt')
        km_opt_list = self.getarrayvar('km_opt')
        self.diff_opt = [diff_opt_mapping.get(diff_opt_list[dom], 'UNKNOWN')
                         for dom in range(len(diff_opt_list))]
        self.km_opt = [km_opt_mapping.get(km_opt_list[dom], 'UNKNOWN')
                       for dom in range(len(diff_opt_list))]
        nval = len(self.km_opt)
        self.c_s = self.getarrayvar('c_s', default=nval*[0.25])
        self.c_k = self.getarrayvar('c_k', default=nval*[0.15])
        self.khdif = self.getarrayvar('khdif', default=nval*[0])
        self.kvdif = self.getarrayvar('kvdif', default=nval*[0])
        self.diff_6th_opt = self.getarrayvar(
                'diff_6th_opt', default=0)
        self.diff_6th_factor = self.getarrayvar(
                'diff_6th_factor', default=0.12)

    def parse_damping(self):
        self.damp_opt = damp_opt_mapping[self.getvar('damp_opt')]
        ndamp = len(self.damp_opt)
        self.zdamp = self.getarrayvar('zdamp', default=ndamp*[5000.])
        self.dampcoef = self.getarrayvar('dampcoef', default=ndamp*[0.2])
        self.w_damping = bool(self.getvar('w_damping', default=0))
        self.epssm = self.getarrayvar('epssm', default=ndamp*[0.1])
        assert all([beta_s==self.epssm[0] for beta_s in self.epssm]), \
                'Different off-centering coefficients not handled'


class BoundaryControl(WRFNamelist):
    """&bdy_control namelist"""

    def __init__(self,nmldict):
        super().__init__(nmldict)
        self.parse()

    def parse(self):
        self.spec_bdy_width = int(self.getvar('spec_bdy_width', default=5))
        self.spec_zone = int(self.getvar('spec_zone', default=1))
        self.relax_zone = int(self.getvar('relax_zone', default=4))
        assert self.spec_bdy_width == self.spec_zone + self.relax_zone

        self.periodic_x = bool(self.getvar('periodic_x', default=False))
        self.periodic_y = bool(self.getvar('periodic_y', default=False))
