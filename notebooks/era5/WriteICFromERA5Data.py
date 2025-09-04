import sys
import os
import glob
import argparse
import time
import numpy as np

from mpi4py import MPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from erftools.preprocessing import Download_ERA5_Data
from erftools.preprocessing import Download_ERA5_ForecastData
from erftools.preprocessing import Download_ERA5_SurfaceData
from erftools.preprocessing import Download_ERA5_ForecastSurfaceData
from erftools.preprocessing import ReadERA5_3DData
from erftools.preprocessing import ReadERA5_SurfaceData

from erftools.utils.projection import create_lcc_mapping


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = time.time()

    if len(sys.argv) == 1:
        print("Usage: python3 WriteICFromERA5Data.py <input_filename> [--do_forecast=true]")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Download and process ERA5 data.")
    parser.add_argument("input_filename", help="Input filename, e.g. inputs_for_Laura")

    parser.add_argument(
        "--do_forecast", type=lambda x: x.lower() == "true", default=False,
        help="Set to true to download forecast data"
    )   

    # Forecast args (only required if --do_forecast is set)
    parser.add_argument("--forecast_time_hours", type=int,
                        help="Forecast length in hours, e.g. 72")
    parser.add_argument("--interval_hours", type=int,
                        help="Forecast interval in hours, e.g. 6")

    args = parser.parse_args()

    # Enforce requirement only if do_forecast is true
    if args.do_forecast:
        if args.forecast_time_hours is None or args.interval_hours is None:
            parser.error("--do_forecast requires --forecast_time_hours and --interval_hours")

    input_filename = args.input_filename
    do_forecast = args.do_forecast
    forecast_time_hours = args.forecast_time_hours
    interval_hours = args.interval_hours

    os.path.exists("Output/domain_extents.txt") and os.remove("Output/domain_extents.txt")

    os.makedirs("Output/ERA5Data_3D", exist_ok=True)
    os.makedirs("Output/ERA5Data_Surface", exist_ok=True)
    os.makedirs("Output/VTK/3D/ERA5Domain", exist_ok=True)
    os.makedirs("Output/VTK/3D/ERFDomain", exist_ok=True)
    os.makedirs("Output/VTK/Surface/ERA5Domain", exist_ok=True)
    os.makedirs("Output/VTK/Surface/ERFDomain", exist_ok=True)

   # Download surface data
    if do_forecast:
        if(rank==0):
            print("Running forecast (pressure-level) download + processing...")
        filenames, area = Download_ERA5_ForecastSurfaceData(input_filename, forecast_time_hours, interval_hours)
    else:
        print("Running surface download + processing...")
        filenames, area = Download_ERA5_SurfaceData(input_filename)
    comm.Barrier();

    #lambert_conformal = CRS.from_proj4(
    #    "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38.5 +lon_0=-97 +datum=WGS84 +units=m +no_defs")

    lambert_conformal = create_lcc_mapping(area)

    # Each rank scans the local directory for matching files
    filenames = sorted(glob.glob("era5_surf_*.grib"))

    my_files = filenames[rank::size]

    for filename in my_files:
        print(f"[Rank {rank}] Processing file: {filename}")
        ReadERA5_SurfaceData(filename, lambert_conformal)

    comm.Barrier();
    if rank == 0:
        print("All ranks finished successfully. Exiting.", flush=True)
 
    # Download 3d data over pressure levels
    if do_forecast:
        filenames, area = Download_ERA5_ForecastData(input_filename, forecast_time_hours, interval_hours)
        comm.Barrier();

        filenames = sorted(glob.glob("era5_3d_*.grib"))

        my_files = filenames[rank::size]
        # Create the directory if it doesn't exist
        for filename in my_files:
            print(f"Processing file: {filename}")
            ReadERA5_3DData(filename, lambert_conformal)

    else:
        filename, area = Download_ERA5_Data(input_filename)
        print("Filename is ", filename)
        print(f"Processing file: {filename}")
        ReadERA5_3DData(filename, lambert_conformal) 

    comm.Barrier()
    
    end_time = time.time()
    elapsed = end_time - start_time

    # Gather or print only on rank 0
    max_elapsed = comm.reduce(elapsed, op=MPI.MAX, root=0)  # max over all ranks

    if rank == 0:
        print(f"Total runtime (wall-clock, across ranks): {max_elapsed:.2f} seconds", flush=True)
