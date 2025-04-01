import xarray as xr

def get_mass_weighted(varname,ds,**dims):
    """Calculate "coupled", i.e., mass-weighted field quantities

    The input xarray dataset is expected to have the following wrfinpt
    fields: MU, MUB, C1H, C2H, C1F, C2F, MAPFAC_U, and MAPFAC_V.

    `dims` should be a single key=value pair used to select a boundary
    plane. Planes on the high end should be selected with negative
    indices.

    Notes:
    - The mass-weighted U and V are located at their respective
      staggered locations
    - The cell-centered column mass MU+MUB is staggered by averaging
      interior values to faces and extrapolating values to boundary
      faces

    See https://forum.mmm.ucar.edu/threads/definitions-of-_btxe-and-_bxe-in-wrfbdy-output.187/#post-23322
    E.g., the boundary values (the BXE, BYE, BXS, BYS arrays) for
    the moist fields would be
        qv(i,k,j)*C1(k) * ( mub(i,j) + mu(i,j) ) + C2(k).

    See also the subroutine `mass_weight` in WRF dyn_em/module_bc_em.F

    See also the momentum variable definitions in arw_v4 following Eqn. 2.17.
    """
    assert len(dims.keys()) == 1, 'Can only specify one boundary coordinate at a time'

    da = ds[varname].isel(**dims)

    # get unstag_dim, idx, bdy_width
    for dim,idx in dims.items():
        if dim.endswith('_stag'):
            unstag_dim = dim[:-5]
        else:
            unstag_dim = dim
    low_end = (idx >= 0)
    bdy_width = idx if low_end else -idx-1

    mut = (ds['MUB']+ds['MU']).isel({unstag_dim:idx})
    if varname == 'U':
        # stagger in west-east direction
        if dim == 'west_east_stag' and bdy_width > 0:
            idx1 = bdy_width - 1
            if not low_end:
                idx1 = -idx1 - 1
            mut1 = (ds['MUB']+ds['MU']).isel({unstag_dim:idx1})
            mut = 0.5 * (mut + mut1)
        elif dim == 'south_north':
            mut = xr.concat([mut,mut.isel(west_east=-1)],'west_east')
            mut = mut.rename(west_east='west_east_stag')
            mut.loc[dict(west_east_stag=slice(1,-1))] = \
                    0.5 * (mut.isel(west_east_stag=slice(1,-1)) +
                           mut.isel(west_east_stag=slice(0,-2)))
    elif varname == 'V':
        # stagger in south-north direction
        if dim == 'south_north_stag' and bdy_width > 0:
            idx1 = bdy_width - 1
            if not low_end:
                idx1 = -idx1 - 1
            mut1 = (ds['MUB']+ds['MU']).isel({unstag_dim:idx1})
            mut = 0.5 * (mut + mut1)
        elif dim == 'west_east':
            mut = xr.concat([mut,mut.isel(south_north=-1)],'south_north')
            mut = mut.rename(south_north='south_north_stag')
            mut.loc[dict(south_north_stag=slice(1,-1))] = \
                    0.5 * (mut.isel(south_north_stag=slice(1,-1)) +
                           mut.isel(south_north_stag=slice(0,-2)))
    else:
        assert len(mut.dims) == 1
        assert mut.dims[0] in da.dims

    if 'bottom_top' in da.dims:
        C1 = ds['C1H']
        C2 = ds['C2H']
    elif 'bottom_top_stag' in da.dims:
        C1 = ds['C1F']
        C2 = ds['C2F']
    else:
        print('wtf')

    # da    = f(z, x_or_y)
    # mut   = f(x_or_y)
    # C1,C2 = f(z)
    coupled = da * (C1*mut + C2)

    # couple momenta to inverse map factors
    if varname == 'U':
        coupled /= ds['MAPFAC_U'].isel(**dims)
    elif varname == 'V':
        coupled /= ds['MAPFAC_V'].isel(**dims)

    return coupled

