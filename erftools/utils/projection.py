#from pyproj import CRS
#default_lambert_conformal = CRS.from_proj4(
#    "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38.5 +lon_0=-97 +datum=WGS84 +units=m +no_defs")

def create_lcc_mapping(area):
    """Create a PROJ string describing a Lambert conformal conic (LCC)
    projection centered in the given area described by:
        (lat_max, lon_min, lat_min, lon_max)
    """
    lat1 = area[2]
    lat2 = area[0]
    lon1 = area[1]
    lon2 = area[3]

    # Build CRS
    delta = lat2 - lat1
    lon0 = (lon1 + lon2) / 2
    lat0 = (lat1 + lat2) / 2

    lat_1 = lat1 + delta/6
    lat_2 = lat2 - delta/6

    return (
        f"+proj=lcc +lat_1={lat_1:.6f} +lat_2={lat_2:.6f} "
        f"+lat_0={lat0:.6f} +lon_0={lon0:.6f} +datum=WGS84 +units=m +no_defs"
    )

def calculate_utm_zone(longitude):
     """Calculate the UTM zone for a given longitude."""
     return int((longitude + 180) // 6) + 1
