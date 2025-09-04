from importlib import resources
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from erftools.preprocessing import Download_GFS_Data
from erftools.preprocessing import Download_GFS_ForecastData
from erftools.preprocessing import ReadGFS_3DData
from erftools.preprocessing import ReadGFS_3DData_UVW

from erftools.utils.map import write_US_map_vtk


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Usage: python3 WriteICFromGFSData.py <input_filename> [--do_forecast=true]")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Download and process GFS data.")
    parser.add_argument("input_filename", help="Input filename, e.g. inputs_for_Laura")
    parser.add_argument("--do_forecast", type=bool, default=False, help="Set to true to download forecast data")

    args = parser.parse_args()

    input_filename = args.input_filename
    do_forecast = args.do_forecast

    if do_forecast:
        filenames, area = Download_GFS_ForecastData(input_filename)
        lambert_conformal = write_US_map_vtk("USMap_LambertProj.vtk", area)
        # Create the directory if it doesn't exist
        os.makedirs("Output", exist_ok=True)
        for filename in filenames:
            print(f"Processing file: {filename}")
            ReadGFS_3DData(filename, area, lambert_conformal)
        
    else:
        filename, area = Download_GFS_Data(input_filename)
        lambert_conformal = write_US_map_vtk("USMap_LambertProj.vtk", area)
        print("Filename is ", filename)
        print(f"Processing file: {filename}")
        ReadGFS_3DData(filename, area, lambert_conformal)
        #ReadGFS_3DData_UVW(filename, area, lambert_conformal)
