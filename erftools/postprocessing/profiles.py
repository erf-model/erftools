import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

class AveragedProfiles(object):
    """Process text diagnostic profiles that were written out with:
        ```
        erf.v           = 1
        erf.data_log    = scalars.txt profiles1.txt profiles2.txt profiles3.txt
        erf.profile_int = 100  # output interval
        ```
    These text files include:
    1. Time history of surface quantities (u*, θ*, L)
    2. Time history of mean profiles (ubar, vbar, wbar, thetabar, ...)
    3. Time history of resolved-scale stress profiles (u'u', u'v', u'w', ...)
    4. Time history of subfilter-scale (SFS) stress profiles (tau11, tau12,
       tau13, ...)

    All three horizontal-averaged profile output files need to be specified to
    have complete averaged profile data. Note that AveragedProfiles does not
    process the scalar surface time-history.
    """
    timename = 't' # 'time'
    heightname = 'z' # 'height'

    # output columns -- these must match the output order
    profile1vars = ['u','v','w',
                    'ρ','θ','e',
                    'Kmv','Khv',
                    'qv','qc','qr',
                    'qi','qs','qg']
    profile2vars = ["u'u'", "u'v'", "u'w'",
                    "v'v'", "v'w'", "w'w'",
                    "θ'u'", "θ'v'", "θ'w'", "θ'θ'",
                    "ui'ui'u'", "ui'ui'v'", "ui'ui'w'",
                    "p'u'", "p'v'", "p'w'",
                    "w'qv'", "w'qc'", "w'qr'",
                    "θv'w'"]
    profile3vars = ['τ11','τ12','τ13',
                    'τ22','τ23','τ33',
                    'τθw', "τqvw", "τqcw",'ε']

    # these variables will be assigned the staggered vertical coordinate
    staggeredvars = ["w",
                     "u'w'", "v'w'", "w'w'",
                     "ui'ui'w'",
                     "θ'w'", "θv'w'", "p'w'", "k'w'",
                     "w'qv'", "w'qc'", "w'qr'",
                     "τ13", "τ23",
                     "τθw", "τqvw", "τqcw"]

    def __init__(self, *args, t0=0.0, sampling_interval_s=None, zexact=None,
                 timedelta=False,resample=None):
        """Load diagnostic profile data from 3 datafiles

        Parameters
        ----------
        *args : list or glob string
            Averaged profile datafiles to load
        t0 : float, optional
            With `sampling_interval_s`, used to overwrite the
            timestamps in the text data to address issues with
            insufficient precision
        sampling_interval_s : float, optional
            Overwrite the time dimension coordinate with
            t0 + np.arange(Ntimes)*sampling_interval_s
        zexact : array-like, optional
            List of cell-centered heights in the computational
            domain, used to overwrite the height dimension coordinate
            and address issues with insufficient precision
        timedelta : bool, optional
            If true, convert times to a TimedeltaIndex
        resample : str, optional
            Calculate rolling mean with given interval; implies
            timedelta=True
        """
        assert (len(args) == 1) or (len(args) == 3)
        if len(args) == 1:
            if isinstance(args[0], (list,tuple)):
                fpathlist = args[0]
                assert len(fpathlist)<=3, \
                    'Expected list of 1-3 separate profile datafiles'
            else:
                assert isinstance(args[0], str)
                fpathlist = sorted(glob.glob(args[0]))
                assert len(fpathlist)<=3, \
                    f'Expected to find 3 files, found {len(fpathlist)} {fpathlist}'
        else:
            fpathlist = args
        self._load_profiles(*fpathlist)
        if sampling_interval_s is not None:
            texact = t0 \
                   + (np.arange(self.ds.dims[self.timename])+1) * sampling_interval_s
            self.ds = self.ds.assign_coords({self.timename: texact})
        if timedelta or (resample is not None):
            td = pd.to_timedelta(self.ds.coords[self.timename],unit='s')
            self.ds = self.ds.assign_coords({self.timename: td})
            if resample:
                assert isinstance(resample, str)
                self.ds = self.ds.resample({self.timename: resample}).mean()
        if zexact is not None:
            self.ds = self.ds.assign_coords({self.heightname: zexact})
        self._process_staggered()

    def _read_text_data(self, fpath, columns):
        df = pd.read_csv(fpath, sep='\s+', header=None, names=columns)
        df = df.set_index([self.timename,self.heightname])
        isdup = df.index.duplicated(keep='last')
        if np.any(isdup):
            print('Note: One or more restarts found, loading the latest')
        return df.loc[~isdup].sort_index()

    def _load_profiles(self, mean_fpath, flux_fpath=None, sfs_fpath=None):
        alldata = []
        idxvars = [self.timename, self.heightname]
        assert os.path.isfile(mean_fpath)
        print('Loading mean profiles from',mean_fpath)
        mean = self._read_text_data(mean_fpath, idxvars+self.profile1vars)
        alldata.append(mean)

        # optional profile data
        if (flux_fpath is not None) and os.path.isfile(flux_fpath):
            print('Loading resolved flux profiles from',flux_fpath)
            fluxes = self._read_text_data(flux_fpath, idxvars+self.profile2vars)
            alldata.append(fluxes)
        else:
            print('No resolved stress data available')
        if (sfs_fpath is not None) and os.path.isfile(sfs_fpath):
            print('Loading SFS stress profiles from',sfs_fpath)
            sfs = self._read_text_data(sfs_fpath, idxvars+self.profile3vars)
            alldata.append(sfs)
        else:
            print('No SFS data available')

        # trim dataframes (to handle data loading when simulation is still
        # running and the profile datafiles have different lengths)
        tmax = alldata[0].index.levels[0][-1]
        for i,df in enumerate(alldata):
            alldata[i] = df.loc[(slice(0,tmax),slice(None)),:]

        # create xarray dataset
        self.ds = pd.concat(alldata, axis=1).to_xarray()

    def _process_staggered(self):
        topval = self.ds['θ'].isel(t=-1).values[-1]
        if topval > 0:
            # profiles are not on staggered grid
            return
        assert topval == 0, f'Found θ[k=-1] = {topval}?!'
        print('**Staggered output detected**')
        zstag = self.ds.coords['z'].values
        zcc = 0.5 * (zstag[1:] + zstag[:-1])
        # collect cell-centered and staggered outputs that are available
        stagvars = []
        ccvars = []
        for varn in self.ds.data_vars:
            if varn in self.staggeredvars:
                stagvars.append(varn)
            else:
                ccvars.append(varn)
        # cell-centered: throw out values associated with highest z face, update coordinates
        cc = self.ds[ccvars].isel(z=slice(0,-1))
        cc = cc.assign_coords(z=zcc)
        # associate staggered outputs with new coordinate
        fc = self.ds[stagvars]
        fc = fc.rename(z='zstag')
        # average the staggered values to cell centers (destagger)
        fc_destag = 0.5 * ( self.ds[stagvars].isel(z=slice(0,  -1)).assign_coords(z=zcc)
                          + self.ds[stagvars].isel(z=slice(1,None)).assign_coords(z=zcc))
        fc_destag = fc_destag.assign_coords(z=zcc)
        fc_destag = fc_destag.rename_vars({var:f'{var}_destag' for var in stagvars})
        # average unstaggered values to face centers (stagger) -- just rho for now
        cc_stag = 0.5 * ( self.ds[['ρ']].isel(z=slice(0,-1)).assign_coords(z=zstag[:-1])
                        + self.ds[['ρ']].isel(z=slice(1,None)).assign_coords(z=zstag[1:]))
        cc_stag = cc_stag.assign_coords(z=zstag[1:-1])
        cc_stag = cc_stag.rename(z='zstag')
        cc_stag = cc_stag.rename_vars({'ρ':'ρ_stag'})
        # combine into single dataset
        self.ds = xr.merge([cc,fc_destag,fc,cc_stag])

    @property
    def t(self):
        return self.ds.coords['t']

    @property
    def z(self):
        return self.ds.coords['z']

    @property
    def zstag(self):
        return self.ds.coords['zstag']

    def __getitem__(self,key):
        return self.ds[key]

    def calc_ddt(self,*args):
        """Calculate time derivative, based on the given profile output
        interval.
        """
        dt = self.ds.coords[self.timename][1] - self.ds.coords[self.timename][0]
        print('dt=',dt.values)
        varlist = args if len(args) > 0 else self.profile1vars
        for varn in varlist:
            self.ds[f'd{varn}/dt'] = self.ds[varn].diff(self.timename) / dt

    def calc_grad_firstorder(self,*args):
        """Calculate vertical gradient for specified fields; if none
        specified, then all gradients are calculated.

        This performs a simple backward difference version and assumes a
        constant dz.
        """
        dz = self.ds.coords[self.heightname][1] - self.ds.coords[self.heightname][0]
        #print('dz=',dz.values)
        allvars = self.profile1vars + self.profile2vars + self.profile3vars
        varlist = args if len(args) > 0 else allvars
        for varn in varlist:
            self.ds[f'd{varn}/dz'] = self.ds[varn].diff(self.heightname) / dz

    def calc_grad(self,*args,two_point=False):
        """Calculate vertical gradient for specified fields; if none
        specified, then all gradients are calculated.

        This performs a second-order central difference for a two or three
        point stencil width.
        - If two_point=True, then differences are either calculated from k±1/2
          and staggered to k (unstaggered quantitites) or calculated from k,k+1
          and destaggered to k+1/2 (staggered quantities)
        - If two_point=False, then differences are calculated from k±1 and
          located at k (for both staggered and unstaggered quantities)
        - Except for staggered to unstaggered, differences revert to first
          order at boundaries
        """
        allvars = self.profile1vars + self.profile2vars + self.profile3vars
        varlist = args if len(args) > 0 else allvars
        for varn in varlist:
            fld = self.ds[varn]
            zdim = 'zstag' if 'zstag' in fld.coords else 'z'
            dz = np.diff(fld.coords[zdim])
            suffix = ''
            if two_point:
                if zdim=='zstag':
                    print(f'diff {varn} from fc to cc')
                    new_zdim = 'z'
                    suffix = '(destag)'
                    gradfld = np.diff(fld.values,axis=1) / dz
                else:
                    print(f'diff {varn} from cc to fc')
                    new_zdim = 'zstag'
                    suffix = '(stag)'
                    gradfld = np.zeros((fld.sizes['t'],fld.sizes[zdim]+1))
                    gradfld[:,1:-1] = np.diff(fld.values,axis=1) / dz
                    gradfld[:, 0] = gradfld[:, 1]
                    gradfld[:,-1] = gradfld[:,-2]
            else:
                if zdim=='zstag':
                    print(f'diff {varn} at fc')
                else:
                    print(f'diff {varn} at cc')
                new_zdim = zdim
                fld = fld.values
                gradfld = np.zeros_like(fld)
                gradfld[:, 1:-1] = (fld[:,2:] - fld[:,:-2]) / ( dz[1:] + dz[:-1])
                gradfld[:, 0] = (fld[:, 1] - fld[:, 0]) / dz[0]
                gradfld[:,-1] = (fld[:,-1] - fld[:,-2]) / dz[-1]
            self.ds[f'd{varn}/dz{suffix}'] = (('t',new_zdim), gradfld)

    def calc_stress(self,check=False,ustar=0):
        """Calculate total stresses and fluxes (note: τ are deviatoric stresses)

        If check==True, assert that the SFS stress tensor is traceless

        If ustar is specified, normalized momentum fluxes will be calculated
        """
        trace = self.ds['τ11'] + self.ds['τ22'] + self.ds['τ33']
        if check:
            assert np.abs(trace).max() < 1e-8, \
                    f'SFS stresses do not sum to zero: {np.abs(trace).max()}'
        self.ds['uu_tot'] = self.ds["u'u'"] + self.ds['τ11'] + 2./3.*self.ds['e']
        self.ds['vv_tot'] = self.ds["v'v'"] + self.ds['τ22'] + 2./3.*self.ds['e']
        try:
            # if output is staggered, w'w' are on staggered but τ33 and e are
            # unstaggered
            ww = self.ds["w'w'_destag"]
        except KeyError:
            ww = self.ds["w'w'"]
        self.ds['ww_tot'] = ww + self.ds['τ33'] + 2./3.*self.ds['e']
        self.ds['uv_tot'] = self.ds["u'v'"] + self.ds['τ12']
        self.ds['uw_tot'] = self.ds["u'w'"] + self.ds['τ13']
        self.ds['vw_tot'] = self.ds["v'w'"] + self.ds['τ23']
        self.ds['ustar_tot'] = (  self.ds['uw_tot']**2
                                + self.ds['vw_tot']**2)**0.25
        self.ds['hfx_tot'] = self.ds["θ'w'"] + self.ds['τθw']
        if ustar > 0:
            self.ds['uu_tot_norm'] = self.ds['uu_tot'] / ustar**2
            self.ds['vv_tot_norm'] = self.ds['vv_tot'] / ustar**2
            self.ds['ww_tot_norm'] = self.ds['ww_tot'] / ustar**2
            self.ds['uw_tot_norm'] = self.ds['uw_tot'] / ustar**2
            self.ds['uw_tot_norm'] = self.ds['uw_tot'] / ustar**2
            self.ds['vw_tot_norm'] = self.ds['vw_tot'] / ustar**2
            self.ds['shear_stress_norm'] = self.ds['ustar_tot']**2 / ustar**2

    def calc_wind(self):
        u = self.ds['u']
        v = self.ds['v']
        w = self.ds['w']
        self.ds['velmag'] = np.sqrt(u*u + v*v + w*w)
        self.ds['wspd'] = np.sqrt(u*u + v*v)
        self.ds['wdir'] = 180. + np.degrees(np.arctan2(u,v))
