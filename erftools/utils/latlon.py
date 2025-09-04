import numpy as np


def find_latlon_indices(domain_lons, domain_lats, lon, lat):
    nlats = len(domain_lats)
    nlons = len(domain_lons)
    lat_idx = 0
    lon_idx = 0
    for i in range(0,nlats):
        if(lat < domain_lats[i]):
            lat_idx = i
            break

    for j in range(0,nlons):
        if(lon < domain_lons[j]):
            lon_idx = j
            break

    return lon_idx, lat_idx


def find_erf_domain_extents(x_grid, y_grid, nx, ny,
                            outfile=None,
                            offset=100e3):
    """Need to determine the box extents of the cartesian box
      (xmin, xmax), (ymin, ymax)
    that fits in this region.

    NOTE: Currently this method only works for the northern hemisphere
    """

    # Top is  [nx-1,:]
    # Bpttom is [0,:]
    # Left is [:,0]
    # Right is [:,ny-1]

    ymax = min(y_grid[nx-1,:]) - offset;

    # Intersect it with the leftmost longitude and rightmost longitude

    # Leftmost longitude is the line joined by
    # x1, y1 = x_grid[0   ,-1], y_grid[0   ,-1]
    # x2, y2 = x_grid[nx-1,-1], y_grid[nx-1,-1]

    #print("Values are ", x_grid[0,-1], y_grid[0,-1], x_grid[-1,-1], y_grid[-1,-1])

    i1 = 0
    for i in range(0, nx-1):
        if(y_grid[i,-1] < ymax and y_grid[i+1,-1] > ymax):
            i1 = i
            break
    xmax = min(x_grid[i1,-1], x_grid[0,-1]) - offset

    for i in range(0, nx-1):
        if(y_grid[i,0] < ymax and y_grid[i+1,0] > ymax):
            i1 = i
            break
    xmin = max(x_grid[i1,0], x_grid[0,0]) + offset

    j1 = 0
    for j in range(0, ny-1):
        if(x_grid[0,j] < xmax and x_grid[0,j+1] > xmax):
            j1 = j
            break
    y1 = y_grid[0,j1];

    for j in range(0, ny-1):
        if(x_grid[0,j] < xmin and x_grid[0,j+1] > xmin):
            j1 = j
            break
    y2 = y_grid[0,j1]

    ymin = max(y1,y2) + offset

    if outfile is not None:
        try:
            # Use exclusive creation mode ("x") so only one rank succeeds
            with open(outfile, "x") as f:
                print("geometry.prob_lo  =",
                       np.ceil(xmin + 50e3),
                       np.ceil(ymin + 50e3),
                       0.0,
                       file=f)
                print("geometry.prob_hi  =",
                       np.floor(xmax - 50e3),
                       np.floor(ymax - 50e3),
                       25000.0,
                       file=f)
        except FileExistsError:
            # Another rank already wrote the file — just skip
            pass
    else:
        print("geometry.prob_lo  = ",
              np.ceil(xmin+offset/2), np.ceil(ymin+offset/2), 0.0)
        print("geometry.prob_hi  = ",
              np.floor(xmax-offset/2), np.floor(ymax-offset/2), 25000.0)

    return xmin, xmax, ymin, ymax
