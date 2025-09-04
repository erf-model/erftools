from importlib import resources
import numpy as np
from pyproj import Transformer
import click

from erftools.utils.projection import create_lcc_mapping
from erftools.io.vtk import write_vtk_map


def create_map(area,
               mapfile='US_state_borders_coordinates.txt',
               shift_to_origin=False):
    """Calculate Lambert conformal conic projection for the given area
    and transform the specified map (textfile with whitespace-delimited
    longitude, latitude, and state/country/region IDs)

    The area is described by:
        (lat_max, lon_min, lat_min, lon_max)
    """
    with resources.open_text('erftools.data', mapfile) as f:
        coordinates = np.loadtxt(f)

    lambert_conformal = create_lcc_mapping(area)

    # Create transformer FROM geographic (lon/lat) TO Lambert
    transformer = Transformer.from_crs("EPSG:4326",
                                       lambert_conformal,
                                       always_xy=True)

    # Process each latitude and longitude, convert to projected coordinates
    x_trans = []
    y_trans = []
    id_vec = []
    for lon, lat, state_id in coordinates:
        x, y = transformer.transform(lon, lat)
        x_trans.append(x)
        y_trans.append(y)
        id_vec.append(state_id)

    # Shift coordinates to ensure minimum x and y start at 0
    if shift_to_origin:
        x_trans = np.array(x_trans) - min(x_trans)
        y_trans = np.array(y_trans) - min(y_trans)

    return x_trans, y_trans, id_vec, lambert_conformal


@click.command()
@click.argument('output_file', type=click.Path(writable=True))
@click.option('--area', nargs=4, type=float,
              help='Bounding box: lat_max, lon_min, lat_min, lon_max')
def write_US_map_vtk(output_file, area, **kwargs):
    """Write out map of the United States forvisualization.

    A map of the US within the specified area is transformed with the
    Lambert conformal conic projection and then written to a VTK file in
    ASCII polydata format
    """
    x_trans, y_trans, id_vec, lcc_proj = create_map(area, **kwargs)
    write_vtk_map(x_trans, y_trans, id_vec, output_file)
    return lcc_proj
