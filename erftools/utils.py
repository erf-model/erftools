import numpy as np
import xarray as xr
import ussa1976

from .constants import CONST_GRAV
from .inputs import ERFInputs
from .input_sounding import InputSounding

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


def get_zcc(inp, verbose=False) -> np.array:
    """Automatically get modeled heights at cell centers, which is exact
    for const dz or stretched grids and approximate for general,
    variable dz grids over terrain.

    `inp` may be an ERF input file path or an instance of ERFInputs.
    """
    if not isinstance(inp, ERFInputs):
        inp = ERFInputs(inp)
    if len(inp.erf.terrain_z_levels) > 0:
        if verbose:
            print('Cell centers from input terrain_z_levels')
        zstag = np.array(inp.erf.terrain_z_levels)
        zcc = 0.5 * (zstag[1:] + zstag[:-1])
    elif inp.erf.grid_stretching_ratio > 1:
        if verbose:
            print('Cell centers from grid_stretching_ratio and initial_dz')
        s = inp.erf.grid_stretching_ratio
        dz0 = inp.erf.initial_dz
        nz = inp.amr.n_cell[2]
        zstag = dz0 * (s**np.arange(nz+1) - 1) / (s - 1) # partial sum
        zcc = 0.5 * (zstag[1:] + zstag[:-1])
    else:
        if verbose:
            print('Cell centers from geometry definition with constant dz')
        nz = inp.amr.n_cell[2]
        const_dz = (inp.geometry.prob_hi[2] - inp.geometry.prob_lo[2]) / nz
        zcc = inp.geometry.prob_lo[2] + np.arange(0.5, nz) * const_dz
    return zcc

def est_p(z, soundingfile=None) -> np.array:
    """Estimate air pressure [Pa] at a set of heights, using either the
    US Standard Atmosphere (1976) or a WRF-style input_sounding.
    """
    z = np.array(z)
    if soundingfile is not None:
        sound = InputSounding(soundingfile)
        sound.interp_levels(z)
        sound.integrate_column()
        p = sound.pm
    else:
        ds = ussa1976.compute(z,'p')
        p = ds['p'].values
    return p
