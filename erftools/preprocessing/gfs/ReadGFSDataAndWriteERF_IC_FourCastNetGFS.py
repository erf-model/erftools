from importlib import resources
import pygrib
import numpy as np
import struct
from pyproj import Transformer, CRS
import matplotlib.pyplot as plt
import sys
import os
from scipy.interpolate import interp1d

from erftools.io import write_binary_vtk_on_native_grid
from erftools.preprocessing import write_binary_vtk_cartesian
from erftools.preprocessing import plot_1d

from erftools.constants import CONST_GRAV as const_g
from erftools.utils.projection import calculate_utm_zone


def ReadGFS_3DData_FourCastNetGFS(file_path, area, is_IC):
    # Open the GRIB2 file
    pressure_levels = []
    ght_3d_hr3 = []
    temp_3d_hr3 = []
    uvel_3d_hr3 = []
    vvel_3d_hr3 = []
    wvel_3d_hr3 = []
    qv_3d_hr3 = []
    qc_3d_hr3 = []
    qr_3d_hr3 = []
    rh_3d_hr3 = []
    theta_3d_hr3 = []
    vort_3d_hr3 = []
    pressure_3d_hr3 = []
    lats, lons = None, None  # To store latitude and longitude grids

    printed_time = False

    datetime_str = ""

    with pygrib.open(file_path) as grbs:
        for grb in grbs:

            if not printed_time:
                year = grb.year
                month = grb.month
                day = grb.day
                hour = grb.hour
                minute = grb.minute if hasattr(grb, 'minute') else 0
                print(f"Date: {year}-{month:02d}-{day:02d}, Time: {hour:02d}:{minute:02d} UTC")
                datetime_str = f"{year:04d}_{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
                print(f"Datetime string: {datetime_str}")
                printed_time = True

            #print(f"Variable: {grb.name}, Level: {grb.level}, Units: {grb.parameterUnits}")
            if "Temperature" in grb.name:
                # Append temperature values
                temp_3d_hr3.append(grb.values)

                # Append pressure level
                pressure_levels.append(grb.level)

            if "Geopotential height" in grb.name:
                ght_3d_hr3.append(grb.values)

            if "Potential temperature" in grb.name:
                theta_3d_hr3.append(grb.values)

            if "Pressure" in grb.name:
                pressure_3d_hr3.append(grb.values)

            if "U component of wind" in grb.name:
                uvel_3d_hr3.append(grb.values)

            if "V component of wind" in grb.name:
                vvel_3d_hr3.append(grb.values)

            if "Geometric vertical velocity" in grb.name:
                wvel_3d_hr3.append(grb.values)

            if "Specific humidity" in grb.name:
                qv_3d_hr3.append(grb.values)

            if "Cloud mixing ratio" in grb.name:
                qc_3d_hr3.append(grb.values)

            if "Rain mixing ratio" in grb.name:
                qr_3d_hr3.append(grb.values)

            if "Relative humidity" in grb.name:
                rh_3d_hr3.append(grb.values)

            if "Absolute vorticity" in grb.name:
                vort_3d_hr3.append(grb.values)

            # Retrieve latitude and longitude grids (once)
            if lats is None or lons is None:
                lats, lons = grb.latlons()

    # Stack into a 3D array (level, lat, lon)
    ght_3d_hr3 = np.stack(ght_3d_hr3, axis=0)
    uvel_3d_hr3 = np.stack(uvel_3d_hr3, axis=0)
    vvel_3d_hr3 = np.stack(vvel_3d_hr3, axis=0)
    #wvel_3d_hr3 = np.stack(wvel_3d_hr3, axis=0)
    #theta_3d_hr3 = np.stack(theta_3d_hr3, axis=0)
    #qv_3d_hr3 = np.stack(qv_3d_hr3, axis=0)
    #qc_3d_hr3 = np.stack(qc_3d_hr3, axis=0)
    #qr_3d_hr3 = np.stack(qr_3d_hr3, axis=0)
    rh_3d_hr3 = np.stack(rh_3d_hr3, axis=0)
    temp_3d_hr3 = np.stack(temp_3d_hr3, axis=0)
    #vort_3d_hr3 = np.stack(vort_3d_hr3, axis=0)

    #pressure_3d_hr3 = np.stack(pressure_3d_hr3, axis=0)
    # Get the size of each dimension
    dim1, dim2, dim3 = ght_3d_hr3.shape

    # Print the sizes
    print(f"Size of dimension 1: {dim1}")
    print(f"Size of dimension 2: {dim2}")
    print(f"Size of dimension 3: {dim3}")

    # Convert pressure levels to numpy array for indexing
    pressure_levels = np.array(pressure_levels)


    # Extract unique latitude and longitude values
    unique_lats = np.unique(lats[:, 0])  # Take the first column for unique latitudes
    unique_lons = np.unique(lons[0, :])  # Take the first row for unique longitudes

    print("Min max lat lons are ", unique_lats[0], unique_lats[-1], unique_lons[0], unique_lons[-1]);


    nlats = len(unique_lats)
    nlons = len(unique_lons)

    lat_max = area[0]
    lon_min = 360.0 + area[1]
    lat_min = area[2]
    lon_max = 360.0 + area[3]

    print("Lat/lon min/max are ", lat_min, lat_max, lon_min, lon_max)

    # Example: regular grid
    lat_resolution = unique_lats[1] - unique_lats[0]
    lon_resolution = unique_lons[1] - unique_lons[0]

    lat_start = int((lat_min - unique_lats[0]) / lat_resolution)
    lat_end   = int((lat_max - unique_lats[0]) / lat_resolution)
    lon_start = int((lon_min - unique_lons[0]) / lon_resolution)
    lon_end   = int((lon_max - unique_lons[0]) / lon_resolution)

    domain_lats = unique_lats[lat_start:lat_end+1]
    domain_lons = unique_lons[lon_start:lon_end+1]

    print("The min max are",(lat_start, lat_end, lon_start, lon_end));

    nx = domain_lats.shape[0]
    ny = domain_lons.shape[0]

    print("nx and ny here are ", nx, ny)

    ght_3d_hr3   = ght_3d_hr3[:, nlats-lat_end-1:nlats-lat_start, lon_start:lon_end+1]
    uvel_3d_hr3  = uvel_3d_hr3[:, nlats-lat_end-1:nlats-lat_start, lon_start:lon_end+1]
    vvel_3d_hr3  = vvel_3d_hr3[:, nlats-lat_end-1:nlats-lat_start, lon_start:lon_end+1]
    rh_3d_hr3    = rh_3d_hr3[:, nlats-lat_end-1:nlats-lat_start, lon_start:lon_end+1]
    temp_3d_hr3  = temp_3d_hr3[:, nlats-lat_end-1:nlats-lat_start, lon_start:lon_end+1]

    print("Size of rh_3d_hr3 is ", rh_3d_hr3.shape[0])


    prev_mean = np.mean(ght_3d_hr3[0])  # start from the top level
    for k in range(1, ght_3d_hr3.shape[0]):
        current_mean = np.mean(ght_3d_hr3[k])
        print("Val is", k, current_mean)
        if current_mean >= prev_mean:
            nz_admissible = k
            print(f"Mean starts increasing at index {k}")
            break
        prev_mean = current_mean
    else:
        print("Means are strictly decreasing through all levels.")

    #nz = nz_admissible
    nz = ght_3d_hr3.shape[0];

    print("The number of lats and lons are levels are %d, %d, %d"%(lats.shape[0], lats.shape[1], nz));

    #sys.exit("Stopping the script here.")

    z_grid = np.zeros((nx, ny, nz))
    rhod_3d = np.zeros((nx, ny, nz))
    uvel_3d = np.zeros((nx, ny, nz))
    vvel_3d = np.zeros((nx, ny, nz))
    #wvel_3d = np.zeros((nx, ny, nz))
    #theta_3d = np.zeros((nx, ny, nz))
    #qv_3d = np.zeros((nx, ny, nz))
    #qc_3d = np.zeros((nx, ny, nz))
    #qr_3d = np.zeros((nx, ny, nz))
    rh_3d = np.zeros((nx, ny, nz))
    temp_3d = np.zeros((nx, ny, nz))
    #qsat_3d = np.zeros((nx, ny, nz))

    velocity = np.zeros((nx, ny, nz, 3))

    #vort_3d = np.zeros((nx, ny, nz))
    #pressure_3d = np.zeros((nx, ny, nz))
    #theta_3d = np.zeros((nx, ny, nz))


    # Create meshgrid
    x_grid, y_grid = np.meshgrid(domain_lons, domain_lats)
    lon_grid, lat_grid = np.meshgrid(domain_lons, domain_lats)

    lambert_conformal = CRS.from_proj4(
    "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38.5 +lon_0=-97 +datum=WGS84 +units=m +no_defs")

    transformer = Transformer.from_crs("EPSG:4326", lambert_conformal, always_xy=True)

    # Convert the entire grid to UTM
    x_grid, y_grid = transformer.transform(lon_grid, lat_grid)

    k_to_delete = []

    print("size is ", len(qv_3d_hr3))

    with resources.open_text('erftools.data.typical_atmosphere', 'pressure_vs_z_actual.txt') as f:
        pressure_typical = np.loadtxt(f)
    pressure_interp_func = interp1d(pressure_typical[:,1], pressure_typical[:,0], kind='linear', fill_value='extrapolate')

    # Find the index of the desired pressure level
    for k in np.arange(nz-1, -1, -1):

        # Extract temperature at the desired pressure level
        ght_at_lev = ght_3d_hr3[k]
        #pressure_at_lev = pressure_3d_hr3[k]
        uvel_at_lev = uvel_3d_hr3[k]
        vvel_at_lev = vvel_3d_hr3[k]
        rh_at_lev = rh_3d_hr3[k]
        temp_at_lev = temp_3d_hr3[k]
        # Print temperature for the specified level using nested loops
        #for i in range(lats.shape[0]):  # Latitude index
            #for j in range(lats.shape[1]):  # Longitude index
        #lat = lats[i, j]
        #lon = lons[i, j]
        temp_3d[:, :, k] = temp_at_lev
        z_grid[:,:,k] = ght_at_lev
        #pressure_3d[:,:,k] = pressure_at_lev

        uvel_3d[:, :, k] = uvel_at_lev
        vvel_3d[:, :, k] = vvel_at_lev
        rh_3d[:, :, k] = rh_at_lev
        rh_val = rh_at_lev

        print("Avg val is ", k, np.mean(z_grid[:,:,k]),  )

        # Find indices of elements that are zero or less
        #indices = np.argwhere(qv_3d[:, :, k] <= 0)
        indices = np.argwhere(rh_val <= 0)

        # Print indices and values
        for index in indices:
            row, col = index
            value = rh_val[row, col]
            #print(f"Element at index ({row}, {col}, {k}) is zero or less: {value}")

        velocity[:,:,k,0] = uvel_at_lev
        velocity[:,:,k,1] = vvel_at_lev
        velocity[:,:,k,2] = 0.0


        #print(f"Lat and lon are: {lat_grid[0,0]:.2f}, {lon_grid[0,0]:.2f}")
        #print(f"Temperature: {temp_3d[0,0,k]:.2f} K, Pressure: {pressure_3d[0,0,k]:.2f}, Geo height : {z_grid[0,0,k]:.2f} ")


    scalars = {
         "uvel": uvel_3d,
         "vvel": vvel_3d,
         "rh": rh_3d,
         "temperature": temp_3d,
    }


    dir_path = "Images"
    os.makedirs(dir_path, exist_ok=True)

    dir_path = "Output"
    os.makedirs(dir_path, exist_ok=True)

    # Extract the filename from the full path
    filename = os.path.basename(file_path)

    # Extract the forecast hour string, e.g., 'f024'
    forecast_hour = filename.split('.')[-1]

    output_vtk = "./Output/GFS_" + datetime_str + "_" + forecast_hour + ".vtk"

    output_binary = "./Output/ERF_IC_" + datetime_str + "_" + forecast_hour + ".bin"

    write_binary_vtk_on_native_grid(output_vtk,
                                    x_grid, y_grid, z_grid,
                                    k_to_delete=k_to_delete,
                                    point_data=scalars,
                                    velocity=velocity)

    write_binary_vtk_cartesian(output_binary, domain_lats, domain_lons,
                               x_grid, y_grid, z_grid,
                               nx, ny, nz, k_to_delete, scalars)

    scalars_for_ERF = {
         "uvel": uvel_3d,
         "vvel": vvel_3d,
    }

    #plot_1d(temp_3d, pressure_3d, theta_3d, qv_3d, qsat_3d, z_grid, k_to_delete)

