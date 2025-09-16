import xarray as xr

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
