import numpy as np
import xarray as xr

from ..constants import CONST_GRAV

def get_stag_dims(ds_cc):
    stag_dims = {dim if dim != 'bottom_top' else 'bottom_top_stag': size
                 for dim,size in ds_cc.sizes.items()}
    stag_dims['bottom_top_stag'] += 1
    return stag_dims

def get_lo_faces(da,dim='bottom_top_stag'):
    assert dim.endswith('_stag')
    return da.isel({dim:slice(0,-1)}).rename({dim:dim[:-5]})

def get_hi_faces(da,dim='bottom_top_stag'):
    assert dim.endswith('_stag')
    return da.isel({dim:slice(1,None)}).rename({dim:dim[:-5]})

def get_w_from_omega(omega_cc, rho_cc, stag_dims=None):
    if stag_dims is None:
        assert isinstance(omega_cc, xr.DataArray)
        stag_dims = get_stag_dims(omega_cc)

    # following wrf-python (Wallace & Hobbs says this is correct to within 10%)
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
