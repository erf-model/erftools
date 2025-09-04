import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from erftools.preprocessing import Download_ERA5_Data
from erftools.preprocessing import Download_ERA5_ForecastData
from erftools.preprocessing import ReadERA5_3DData

from erftools.utils.projection import create_lcc_mapping

from pyproj import CRS, Transformer
from numpy import *
import pandas as pd
import pyvista as pv

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

if __name__ == "__main__":

     # --- Parse arguments ---
    parser = argparse.ArgumentParser(description="Write NOAA forecast cone to ASCII VTK")
    parser.add_argument("input_filename", help="Some input file (not used here, can be metadata)")
    parser.add_argument("input_csv_filename", help="CSV file with lon,lat points")
    args = parser.parse_args()

    input_filename = args.input_filename
    input_csv_filename = args.input_csv_filename
    output_vtk_filename = "NOAAForecastCone.vtk"

    args = parser.parse_args()

    input_filename = args.input_filename

    user_inputs = read_user_input(input_filename)
    print("User inputs:", user_inputs)
    area = user_inputs.get("area", None)

    lambert_conformal = create_lcc_mapping(area)

    transformer = Transformer.from_crs("EPSG:4326", lambert_conformal, always_xy=True)

    # Read CSV
    df = pd.read_csv(input_csv_filename)

    # Transform coordinates
    lcc_coords = [transformer.transform(lon, lat) for lon, lat in zip(df.iloc[:,0], df.iloc[:,1])]
    
    # Create points array (z = 0)
    points = [(x, y, 5000.0) for x, y in lcc_coords]

    # Create PolyData object
    polydata = pv.PolyData(points)

    # Optionally connect the points into a polyline (the cone outline)
    lines = [len(points)] + list(range(len(points)))  # format: [n_points, p0, p1, ...]
    polydata.lines = lines

    # Add original lon/lat as point data
    polydata.point_data["Longitude"] = df.iloc[:, 0].to_numpy()
    polydata.point_data["Latitude"] = df.iloc[:, 1].to_numpy()

    # Save to VTK
    polydata.save(output_vtk_filename, binary=False)

    print(f"VTK file written to: {output_vtk_filename}")

