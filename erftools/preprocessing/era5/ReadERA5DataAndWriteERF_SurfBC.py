import pygrib
import struct
import sys
import os
import cdsapi
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj, Transformer, CRS
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI
import time

from erftools.utils.latlon import (find_erf_domain_extents,
                                   find_latlon_indices)
from erftools.io import (write_binary_vtk_on_native_grid,
                         write_binary_vtk_on_cartesian_grid,
                         write_binary_simple_erf)


def read_user_input(filename):
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if key == 'area':
                    data[key] = [float(x) for x in value.split(',')]
                elif key == 'time':
                    data[key] = value
                else:
                    data[key] = int(value)
    return data

def Download_ERA5_SurfaceData(inputs):
    # Load user input
    user_inputs = read_user_input(inputs)

    # Define dataset and request
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "sea_surface_temperature",
            "surface_sensible_heat_flux",
            "surface_latent_heat_flux",
            "friction_velocity",
            "land_sea_mask",
            "surface_pressure"  # <- added
        ],
        "data_format": "grib",
        "download_format": "unarchived"
    }
    request.update(user_inputs)

    # Area check
    area = user_inputs.get("area", None)
    assert area and len(area) == 4, "'area' must be a list of four values"
    assert area[0] > area[2], "Latitude order invalid"
    assert area[3] > area[1], "Longitude order invalid"

    # Download data
    client = cdsapi.Client()
    filename = client.retrieve(dataset, request).download()
    return [filename], area    


def write_binary_vtk_cartesian(date_time_forecast_str, output_binary, domain_lats, domain_lons, 
                               x_grid, y_grid, z_grid, 
                               nx, ny, nz, k_to_delete, lambert_conformal, point_data=None):

    xmin, xmax, ymin, ymax = find_erf_domain_extents(x_grid, y_grid, nx, ny)

    nx_erf = ny
    ny_erf = nx
    nz_erf= nz - len(k_to_delete)
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

    sea_surf_temp_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    surf_sen_hf_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    surf_lat_hf_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    fric_vel_erf = np.zeros((nx_erf, ny_erf, nz_erf))
    ls_mask_erf = np.zeros((nx_erf, ny_erf, nz_erf))

    scalars = {
         "sea_surf_temp": sea_surf_temp_erf,
         "surf_sen_hf": surf_sen_hf_erf,
         "surf_lat_hf": surf_lat_hf_erf,
         "fric_vel": fric_vel_erf,
         "ls_mask": ls_mask_erf,
    }

    kcount = 0
    for k in range(nz):  # Iterate over the z-dimension
        z_grid_erf[:,:,kcount] = np.mean(z_grid[:,:,k])+1e-6
        kcount = kcount+1
    z_grid_erf[:,:,0] = 0.0

    transformer = Transformer.from_crs(lambert_conformal, "EPSG:4326", always_xy=True)
    
    if point_data:
        for name, data in point_data.items():
            if name in scalars:  # Check if the name exists in the scalars dictionary
                for j in range(ny_erf):
                    for i in range(nx_erf):
                        lon, lat = transformer.transform(x_grid_erf[j,i], y_grid_erf[j,i])
                        lon = lon
                        lon_idx, lat_idx = find_latlon_indices(domain_lons, domain_lats, lon, lat)
                        #if(lat_idx > 110):
                        #    print("Lat value out of range", lat_idx, lon_idx, x_grid_erf[i,j], y_grid_erf[i,j])
                        #    sys.exit()
                        #print("The values of lat and lon are", x_grid_erf[i,j], y_grid_erf[i,j], lon, lat, lon_idx, lat_idx)
                        #sys.exit()
                        kcount = 0
                        for k in range(nz):  # Iterate over the z-dimension
                            scalars[name][i,j,kcount] = (data[nx-1-lat_idx, lon_idx, k] + data[nx-1-lat_idx, lon_idx-1, k] + 
                                                         data[nx-1-lat_idx+1, lon_idx-1, k] + data[nx-1-lat_idx+1, lon_idx, k])/4.0
                            kcount = kcount+1

            else:
                print("Variable not found in scalars list", name)
                #sys.exit()

    output_cart_vtk = "./Output/VTK/Surface/ERFDomain/" + "ERF_Surface_Cart_" + date_time_forecast_str +".vtk"

    write_binary_vtk_on_cartesian_grid(output_cart_vtk,
                                       x_grid_erf, y_grid_erf, z_grid_erf,
                                       scalars)    

    write_binary_simple_erf(output_binary,
                            x_grid_erf, y_grid_erf, z_grid_erf,
                            point_data=scalars)

def ReadERA5_SurfaceData(file_path, lambert_conformal):
    # Open the GRIB2 file
    sea_surf_temp_era5 = []
    surf_sen_hf_era5 = []
    surf_lat_hf_era5 = []
    fric_vel_era5 = []
    ls_mask_era5 = []
    lats, lons = None, None  # To store latitude and longitude grids

    printed_time = False
    date_time_forecast_str = ""

    with pygrib.open(file_path) as grbs:
        for grb in grbs:
            if not printed_time:
                year = grb.year
                month = grb.month
                day = grb.day
                hour = grb.hour

                minute = grb.minute if hasattr(grb, 'minute') else 0
                print(f"Date: {year}-{month:02d}-{day:02d}, Time: {hour:02d}:{minute:02d} UTC")
                date_time_forecast_str = f"{year:04d}_{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
                print(f"Datetime string: {date_time_forecast_str}")
                printed_time = True

            print(f"Variable: {grb.name}, Level: {grb.level}, Units: {grb.parameterUnits}")
            if "Sea surface temperature" in grb.name:
                # Append temperature values
                sea_surf_temp_era5.append(grb.values)
    
            if "Surface sensible heat flux" in grb.name:
                surf_sen_hf_era5.append(grb.values)
        
            if "Surface latent heat flux" in grb.name:
                surf_lat_hf_era5.append(grb.values)

            if "Friction velocity" in grb.name:
                fric_vel_era5.append(grb.values)
    
            if "Land-sea mask" in grb.name:
                ls_mask_era5.append(grb.values)
    
            # Retrieve latitude and longitude grids (once)
            if lats is None or lons is None:
                print("Reading lats and lons")
                lats, lons = grb.latlons()

    # Stack into a 3D array (level, lat, lon)
    sea_surf_temp_era5 = np.stack(sea_surf_temp_era5, axis=0)
    surf_sen_hf_era5 = np.stack(surf_sen_hf_era5, axis=0)
    surf_lat_hf_era5 = np.stack(surf_lat_hf_era5, axis=0)
    fric_vel_era5 = np.stack(fric_vel_era5, axis=0)
    ls_mask_era5 = np.stack(ls_mask_era5, axis=0)
    
    #pressure_3d_hr3 = np.stack(pressure_3d_hr3, axis=0)
    # Get the size of each dimension
    dim1, dim2, dim3 = sea_surf_temp_era5.shape
    
    # Print the sizes
    print(f"Size of dimension 1: {dim1}")
    print(f"Size of dimension 2: {dim2}")
    print(f"Size of dimension 3: {dim3}")
    
    
    nz = 1
    
    print("The number of lats and lons are levels are %d, %d, %d"%(lats.shape[0], lats.shape[1], nz));
    
    
    # Extract unique latitude and longitude values
    unique_lats = np.unique(lats[:, 0])  # Take the first column for unique latitudes
    unique_lons = np.unique(lons[0, :])  # Take the first row for unique longitudes
    
    print("Min max lat lons are ", unique_lats[0], unique_lats[-1], unique_lons[0], unique_lons[-1]);
    
    nlats = len(unique_lats)
    nlons = len(unique_lons)
    
    lat_start = 0
    lat_end = nlats
    lon_start = 0
    lon_end = nlons
    
    domain_lats = unique_lats[:]
    domain_lons = unique_lons[:]
    
    print("The min max are",(lat_start, lat_end, lon_start, lon_end));

    nx = domain_lats.shape[0]
    ny = domain_lons.shape[0]
    
    print("nx and ny here are ", nx, ny)
    
    z_grid = np.zeros((nx, ny, nz))
    sea_surf_temp = np.zeros((nx, ny, nz))
    surf_sen_hf = np.zeros((nx, ny, nz))
    surf_lat_hf = np.zeros((nx, ny, nz))
    fric_vel = np.zeros((nx, ny, nz))
    ls_mask = np.zeros((nx, ny, nz))
    
                    
    # Create meshgrid
    x_grid, y_grid = np.meshgrid(domain_lons, domain_lats)
    lon_grid, lat_grid = np.meshgrid(domain_lons, domain_lats)
   
    transformer = Transformer.from_crs("EPSG:4326", lambert_conformal, always_xy=True)

    # Convert the entire grid to UTM
    x_grid, y_grid = transformer.transform(lon_grid, lat_grid)
 
    k_to_delete = []

    # Find the index of the desired pressure level
    for k in np.arange(0, 1, 1):
    
        # Extract temperature at the desired pressure level
        sea_surf_temp_at_lev = sea_surf_temp_era5[k]
        surf_sen_hf_at_lev = surf_sen_hf_era5[k]
        surf_lat_hf_at_lev = surf_lat_hf_era5[k]
        fric_vel_at_lev = fric_vel_era5[k]
        ls_mask_at_lev = ls_mask_era5[k]

        sea_surf_temp[:, :, k] = sea_surf_temp_at_lev
        surf_sen_hf[:, :, k] = surf_sen_hf_at_lev
        surf_lat_hf[:, :, k] = surf_lat_hf_at_lev
        fric_vel[:, :, k] = fric_vel_at_lev
        ls_mask[:, :, k] = ls_mask_at_lev
    
    scalars = {
         "sea_surf_temp": sea_surf_temp,
         "surf_sen_hf": surf_sen_hf,
         "surf_lat_hf": surf_lat_hf,
         "fric_vel": fric_vel,
         "ls_mask": ls_mask,
    }

    output_vtk = "./Output/VTK/Surface/ERA5Domain/ERA5_Surface_" + date_time_forecast_str + ".vtk"

    output_binary = "./Output/ERA5Data_Surface/ERF_Surface_" + date_time_forecast_str + ".bin"
    
    write_binary_vtk_on_native_grid(output_vtk,
                                    x_grid, y_grid, z_grid,
                                    k_to_delete=k_to_delete,
                                    skip_latlon=False,
                                    zoffset=1e-6,
                                    point_data=scalars)

    write_binary_vtk_cartesian(date_time_forecast_str, output_binary, domain_lats, domain_lons, 
                               x_grid, y_grid, z_grid, 
                               nx, ny, nz, k_to_delete, lambert_conformal, scalars)


def generate_timestamps(start_dt, hours=72, interval=3):
    timestamps = []
    for i in range(0, hours + 1, interval):
        dt = start_dt + timedelta(hours=i)
        timestamps.append(dt)
    return timestamps

def download_one_timestep(cds_client, dataset, request, output_filename, idx):
    if os.path.exists(output_filename):
        print(f"[{idx}] Skipping existing: {output_filename}")
        return

    print(f"[{idx}] Downloading {output_filename} ...")
    cds_client.retrieve(dataset, request, output_filename)
    print(f"[{idx}] Done: {output_filename}")

# --------------------------
# Main download routine
# --------------------------

def Download_ERA5_ForecastSurfaceData(inputs_file, forecast_time, interval):
    user_inputs = read_user_input(inputs_file)

    dataset = "reanalysis-era5-single-levels"
    variables = [
        "sea_surface_temperature",
        "surface_sensible_heat_flux",
        "surface_latent_heat_flux",
        "friction_velocity",
        "land_sea_mask",
        "surface_pressure"
    ]

    area = user_inputs["area"]
    start_time = datetime(
        user_inputs["year"],
        user_inputs["month"],
        user_inputs["day"],
        int(user_inputs["time"].split(":")[0])
    )

    # 72 hours with 6-hour interval
    timestamps = generate_timestamps(start_time, forecast_time, interval)

     # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cds_client = cdsapi.Client()
    filenames = []

    max_download_ranks = 4   # only allow 4 ranks to download
    active_ranks = min(size, max_download_ranks)

    for idx, dt in enumerate(timestamps):
        # Assign only among the active ranks
        if (idx % active_ranks) != rank:
            continue

        if rank >= active_ranks:
            # This rank is idle for downloading
            continue
    
        y, m, d, h = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"), dt.strftime("%H:%M")
        request = {
                "product_type": "reanalysis",
                "variable": variables,
                "year": y,
                "month": m,
                "day": d,
                "time": [h],
                "format": "grib",
                "area": area,
         }

        fname = f"era5_surf_{y}{m}{d}_{h.replace(':', '')}.grib"
        filenames.append(fname)
        download_one_timestep(cds_client, dataset, request, fname, idx)

    return filenames, area

