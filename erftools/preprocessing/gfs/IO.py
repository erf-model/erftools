import numpy as np
import struct
import os
from pyproj import Proj, Transformer, CRS
from math import sin, cos, atan2

from erftools.utils.latlon import (find_erf_domain_extents,
                                   find_latlon_indices)
from erftools.io import (write_binary_simple_erf,
                         write_binary_vtk_on_cartesian_grid)


def write_binary_vtk_cartesian(date_time_forecast_str,
                               output_binary,
                               domain_lats, domain_lons,
                               x_grid, y_grid, z_grid,
                               nx, ny, nz,
                               k_to_delete,
                               lambert_conformal,
                               point_data=None):

    xmin, xmax, ymin, ymax = find_erf_domain_extents(x_grid, y_grid, nx, ny)

    print("Value of nx and ny are ", nx, ny)

    nx_erf = ny
    ny_erf = nx
    nz_erf= nz - len(k_to_delete) + 1
    dx = (xmax - xmin)/(nx_erf-1)
    dy = (ymax - ymin)/(ny_erf-1)

    xvec_erf = np.zeros(nx_erf)
    yvec_erf = np.zeros(ny_erf)
    for i in range(nx_erf):
        xvec_erf[i] = xmin + dx*i

    for j in range(ny_erf):
        yvec_erf[j] = ymin + dy*j

    x_grid_erf, y_grid_erf = np.meshgrid(xvec_erf, yvec_erf)
    z_grid_erf = np.zeros((nx_erf, ny_erf, nz_erf))

    rho_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    temp_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    uvel_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    vvel_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    wvel_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    rh_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    theta_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    qv_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    qc_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    qr_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    theta_erf = np.zeros((nx_erf, ny_erf, nz_erf))

    lat_erf = np.zeros((nx_erf,ny_erf,nz_erf))
    lon_erf = np.zeros((nx_erf,ny_erf,nz_erf))

    scalars = {
         "density": rho_erf,
         "uvel": uvel_erf,
         "vvel": vvel_erf,
         "wvel": wvel_erf,
         "theta": theta_erf,
         "qv": qv_erf,
         "qc": qc_erf,
         "qr": qr_erf,
    }

    scalars_to_plot = {
         "latitude": lat_erf,
         "longitude": lon_erf,
         "density": rho_erf,
         "uvel": uvel_erf,
         "vvel": vvel_erf,
         "wvel": wvel_erf,
         "theta": theta_erf,
         "qv": qv_erf,
         "qc": qc_erf,
         "qr": qr_erf,
    }


    kcount = 1
    for k in range(nz):  # Iterate over the z-dimension
        if nz-1-k in k_to_delete:
            continue
        z_grid_erf[:,:,kcount] = np.mean(z_grid[:,:,nz-1-k])
        kcount = kcount+1
    z_grid_erf[:,:,0] = 0.0

    transformer = Transformer.from_crs(lambert_conformal, "EPSG:4326", always_xy=True)

    uvel_3d = point_data["uvel"]
    vvel_3d = point_data["vvel"]

    print("Values of nx_erf and ny_erf are", nx_erf, ny_erf);
    print("Shapes of xgrid and ygrid are", x_grid.shape, y_grid.shape);
    print("Shapes of xgrid_erf and ygrid_erf are", x_grid_erf.shape, y_grid_erf.shape);

    x1 = 0.0
    y1 = 0.0
    x2 = 0.0
    y2 = 0.0

    if point_data:
        for name, data in point_data.items():
            if name in scalars_to_plot:  # Check if the name exists in the scalars dictionary
                print("name is", name);
                for j in range(ny_erf):
                    for i in range(nx_erf):
                        lon, lat = transformer.transform(x_grid_erf[j,i], y_grid_erf[j,i])
                        lon_idx, lat_idx = find_latlon_indices(domain_lons,
                                                               domain_lats,
                                                               360.0+lon, lat)
                        lat_erf[i,j,0] = lat
                        lon_erf[i,j,0] = lon
                        kcount = 1
                        for k in range(nz):  # Iterate over the z-dimension
                            if nz-1-k in k_to_delete:
                                continue
                            if(name == "latitude"):
                                scalars_to_plot[name][i,j,kcount] = lat;
                                #print("Reaching here lat", lat)
                            elif(name == "longitude"):
                                scalars_to_plot[name][i,j,kcount] = lon;
                                #print("Reaching here lon", lon)
                            else:
                                scalars_to_plot[name][i,j,kcount] = (data[nx-1-lat_idx, lon_idx, nz-1-k] + data[nx-1-lat_idx, lon_idx-1, nz-1-k] +
                                                         data[nx-1-lat_idx+1, lon_idx-1, nz-1-k] + data[nx-1-lat_idx+1, lon_idx, nz-1-k])/4.0

                        lat0 = domain_lats[lat_idx-1]
                        lat1 = domain_lats[lat_idx]
                        lon0 = domain_lons[lon_idx-1]
                        lon1 = domain_lons[lon_idx]

                        # fractional distances
                        fx = (360.0 + lon - lon0) / (lon1 - lon0)
                        fy = (lat - lat0) / (lat1 - lat0)

                        if(i < nx_erf-1):
                            x1 = x_grid[j,i]
                            y1 = y_grid[j,i]

                            x2 = x_grid[j,i+1]
                            y2 = y_grid[j,i+1]
                        elif(i == nx_erf-1):
                            x1 = x_grid[j,i-1]
                            y1 = y_grid[j,i-1]

                            x2 = x_grid[j,i]
                            y2 = y_grid[j,i]

                        theta = atan2(y2-y1, x2-x1)

                        kcount = 1
                        for k in range(nz):  # Iterate over the z-dimension
                            if nz-1-k in k_to_delete:
                                continue
                            if(name == "latitude"):
                                scalars_to_plot[name][i,j,kcount] = lat;
                                #print("Reaching here lat", lat)
                            elif(name == "longitude"):
                                scalars_to_plot[name][i,j,kcount] = lon;
                                #print("Reaching here lon", lon)
                            elif(name == "uvel" or name == "vvel"):
                                u_tmp = (fx*fy*uvel_3d[nx-1-lat_idx, lon_idx, nz-1-k] + fx*(1-fy)*uvel_3d[nx-1-lat_idx, lon_idx-1, nz-1-k] +
                                                         (1-fx)*(1-fy)*uvel_3d[nx-1-lat_idx+1, lon_idx-1, nz-1-k] + (1-fx)*fy*uvel_3d[nx-1-lat_idx+1, lon_idx, nz-1-k])
                                v_tmp = (fx*fy*vvel_3d[nx-1-lat_idx, lon_idx, nz-1-k] + fx*(1-fy)*vvel_3d[nx-1-lat_idx, lon_idx-1, nz-1-k] +
                                                         (1-fx)*(1-fy)*vvel_3d[nx-1-lat_idx+1, lon_idx-1, nz-1-k] + (1-fx)*fy*vvel_3d[nx-1-lat_idx+1, lon_idx, nz-1-k])

                                if(name == "uvel"):
                                    scalars_to_plot[name][i,j,kcount] = u_tmp*cos(theta) - v_tmp*sin(theta)
                                elif(name == "vvel"):
                                    scalars_to_plot[name][i,j,kcount] = u_tmp*sin(theta) + v_tmp*cos(theta)
                            else:
                                scalars_to_plot[name][i,j,kcount] = (fx*fy*data[nx-1-lat_idx, lon_idx, nz-1-k] + fx*(1-fy)*data[nx-1-lat_idx, lon_idx-1, nz-1-k] +
                                                         (1-fx)*(1-fy)*data[nx-1-lat_idx+1, lon_idx-1, nz-1-k] + (1-fx)*fy*data[nx-1-lat_idx+1, lon_idx, nz-1-k])

                            if(name != "latitude" and name != "longitude"):
                                scalars[name][i,j,kcount] = scalars_to_plot[name][i,j,kcount]

                            kcount = kcount+1

                        scalars_to_plot[name][i,j,0] = scalars_to_plot[name][i,j,1]
                        if(name != "latitude" and name != "longitude"):
                            scalars[name][i,j,0] = scalars[name][i,j,1]

            else:
                print("Variable not found in scalars list", name)
                #sys.exit()

    output_cart_vtk = "./Output/" + "ERF_IC_" + date_time_forecast_str +".vtk"

    print("Writing write_binary_vtk_on_cartesian_grid")
    write_binary_vtk_on_cartesian_grid(output_cart_vtk,
                                       x_grid_erf, y_grid_erf, z_grid_erf,
                                       point_data=scalars_to_plot)

    print("Writing write_binary_simple_erf")
    write_binary_simple_erf(output_binary,
                            x_grid_erf, y_grid_erf, z_grid_erf,
                            lat_erf=lat_erf, lon_erf=lon_erf,
                            point_data=scalars)
