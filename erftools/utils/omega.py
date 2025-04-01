import numpy as np
import xarray as xr

from .xarray import get_stag_dims
from ..constants import CONST_GRAV

def get_w_from_omega(omega_cc, rho_cc, stag_dims=None):
    """Input `rho_cc` is the _moist_ density at cell centers"""
    if stag_dims is None:
        assert isinstance(omega_cc, xr.DataArray)
        stag_dims = get_stag_dims(omega_cc,'bottom_top')

    # following wrf-python (Wallace & Hobbs, see Sec. 7.3.1, says this correct
    # to within 10%)
    w_cc = -omega_cc / (rho_cc * CONST_GRAV)

    # stagger to full level heights
    w_stag = 0.5 * (w_cc.isel(bottom_top=slice(1,None)).values +
                    w_cc.isel(bottom_top=slice(0,  -1)).values)

    # extrap to top face
    w1 = w_cc.isel(bottom_top=-2).values
    w2 = w_cc.isel(bottom_top=-1).values
    w_top = w2 + 0.5*(w2-w1)

    # create data array
    da = xr.DataArray(np.zeros(tuple(stag_dims.values())), dims=stag_dims.keys())
    da.loc[dict(bottom_top_stag=slice(1,-1))] = w_stag
    da.loc[dict(bottom_top_stag=-1)] = w_top

    return da
