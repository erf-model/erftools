import os
from .profiles import AveragedProfiles
from .surface import SurfaceHistory

class DataLog(AveragedProfiles):
    """Process all logged data, including surface history and planar
    averaged profiles
    """
    def __init__(self,surfhist,meanprof,fluxprof,sfsprof,**kwargs):
        super().__init__(meanprof,fluxprof,sfsprof,**kwargs)
        surf = SurfaceHistory(surfhist,**kwargs)
        self.surf = surf.df.to_xarray()
        self.ds['u*'] = self.surf['ustar'].interp(t=self.ds.t)
        self.ds['θ*'] = self.surf['tstar'].interp(t=self.ds.t)
        self.ds['L']  = self.surf['L'].interp(t=self.ds.t)
