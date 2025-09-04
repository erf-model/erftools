import cdsapi
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI

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


def Download_ERA5_ForecastData(inputs_file, forecast_time, interval):
    user_inputs = read_user_input(inputs_file)

    dataset = "reanalysis-era5-pressure-levels"
    variables = [
        "geopotential",
        "relative_humidity",
        "specific_cloud_ice_water_content",
        "specific_cloud_liquid_water_content",
        "specific_humidity",
        "specific_rain_water_content",
        "specific_snow_water_content",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
        "vorticity"
    ]

    pressure_levels = [
        "1", "2", "3", "5", "7", "10", "20", "30", "50", "70", "100", "125",
        "150", "175", "200", "225", "250", "300", "350", "400", "450", "500",
        "550", "600", "650", "700", "750", "775", "800", "825", "850", "875",
        "900", "925", "950", "975", "1000"
    ]

    area = user_inputs["area"]
    start_time = datetime(
        user_inputs["year"],
        user_inputs["month"],
        user_inputs["day"],
        int(user_inputs["time"].split(":")[0])
    )

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
                "pressure_level": pressure_levels,
                "year": y,
                "month": m,
                "day": d,
                "time": [h],
                "format": "grib",
                "area": area,
            }
        fname = f"era5_3d_{y}{m}{d}_{h.replace(':', '')}.grib"
        filenames.append(fname)
        download_one_timestep(cds_client, dataset, request, fname, idx)

    return filenames, area

def Download_ERA5_Data(inputs):

    # Load values from user input file
    user_inputs = read_user_input(inputs)

    # Define dataset and static request fields
    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "geopotential",
            "relative_humidity",
            "specific_cloud_ice_water_content",
            "specific_cloud_liquid_water_content",
            "specific_humidity",
            "specific_rain_water_content",
            "specific_snow_water_content",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "vorticity"
        ],
        "pressure_level": [
            "1", "2", "3", "5", "7", "10",
            "20", "30", "50", "70", "100", "125",
            "150", "175", "200", "225", "250", "300",
            "350", "400", "450", "500", "550", "600",
            "650", "700", "750", "775", "800", "825",
            "850", "875", "900", "925", "950", "975", "1000"
        ],
        "data_format": "grib",
        "download_format": "unarchived"
    }

    # Merge user input into request
    request.update(user_inputs)

    print("User inputs:", user_inputs)
    area = user_inputs.get("area", None)
    assert area and len(area) == 4, "'area' must be a list of four values"
    assert area[0] > area[2], "Latitude order invalid: North (1st) must be greater than South (3rd)"
    assert area[3] > area[1], "Longitude order invalid: East (4th) must be greater than West (2nd)"

    # Instantiate Copernicus client
    client = cdsapi.Client()
    result = client.retrieve(dataset, request)
    filename = result.location.split('/')[-1]

    if os.path.isfile(filename):
        print(f"File already downloaded: {filename}")
    else:
        print(f"Downloading {filename}")
        downloaded = result.download()
        print(f"Downloaded file: {downloaded}")
        assert downloaded == filename

    return filename, area
