import numpy as np
import xarray as xr

def get_stag_dims(ds_cc, dim_to_stagger='bottom_top'):
    """Return dict of dataset dimensions with the specified dimension
    renamed and increased in size by 1
    """
    stag_dims = {}
    for dim,size in ds_cc.sizes.items():
        if dim==dim_to_stagger:
            stag_dims[dim+'_stag'] = size+1
        else:
            stag_dims[dim] = size
    return stag_dims

def get_lo_faces(da,dim='bottom_top_stag'):
    """Get low faces and rename staggered dimension, for destaggering"""
    assert dim.endswith('_stag')
    return da.isel({dim:slice(0,-1)}).rename({dim:dim[:-5]})

def get_hi_faces(da,dim='bottom_top_stag'):
    """Get high faces and rename staggered dimension, for destaggering"""
    assert dim.endswith('_stag')
    return da.isel({dim:slice(1,None)}).rename({dim:dim[:-5]})
