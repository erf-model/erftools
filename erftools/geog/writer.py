"""Write a geo_em-compatible NetCDF file (same names/dims/attributes as geogrid)."""
import glob
import hashlib
import json
import os

import netCDF4
import numpy as np

from . import __version__

GRID_META = {
    "XLAT_M": ("degrees latitude", "Latitude on mass grid"),
    "XLONG_M": ("degrees longitude", "Longitude on mass grid"),
    "XLAT_V": ("degrees latitude", "Latitude on V grid"),
    "XLONG_V": ("degrees longitude", "Longitude on V grid"),
    "XLAT_U": ("degrees latitude", "Latitude on U grid"),
    "XLONG_U": ("degrees longitude", "Longitude on U grid"),
    "CLAT": ("degrees latitude", "Computational latitude on mass grid"),
    "CLONG": ("degrees longitude", "Computational longitude on mass grid"),
    "MAPFAC_M": ("none", "Mapfactor on mass grid"),
    "MAPFAC_V": ("none", "Mapfactor on V grid"),
    "MAPFAC_U": ("none", "Mapfactor on U grid"),
    "MAPFAC_MX": ("none", "Mapfactor (x-dir) on mass grid"),
    "MAPFAC_VX": ("none", "Mapfactor (x-dir) on V grid"),
    "MAPFAC_UX": ("none", "Mapfactor (x-dir) on U grid"),
    "MAPFAC_MY": ("none", "Mapfactor (y-dir) on mass grid"),
    "MAPFAC_VY": ("none", "Mapfactor (y-dir) on V grid"),
    "MAPFAC_UY": ("none", "Mapfactor (y-dir) on U grid"),
    "E": ("-", "Coriolis E parameter"),
    "F": ("-", "Coriolis F parameter"),
    "SINALPHA": ("none", "Sine of rotation angle"),
    "COSALPHA": ("none", "Cosine of rotation angle"),
    "XLAT_C": ("degrees latitude", "Latitude at grid cell corners"),
    "XLONG_C": ("degrees longitude", "Longitude at grid cell corners"),
    "SINALPHA_U": ("none", "Sine of rotation angle on U grid"),
    "COSALPHA_U": ("none", "Cosine of rotation angle on U grid"),
    "SINALPHA_V": ("none", "Sine of rotation angle on V grid"),
    "COSALPHA_V": ("none", "Cosine of rotation angle on V grid"),
}


def code_hash():
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in sorted(glob.glob(os.path.join(here, "*.py"))):
        with open(fn, "rb") as fh:
            h.update(os.path.basename(fn).encode() + b"\0" + fh.read())
    return h.hexdigest()


def write_geo_em(path, grid, outs, attrs):
    g = grid.spec
    nx, ny = g.nx, g.ny
    # Build the file in memory and write it with a single call at the end: CFS is
    # very slow with netCDF's many small writes, and diskless+persist does not
    # persist with netcdf-c 4.10.1.
    ds = netCDF4.Dataset(os.path.basename(path), "w", format="NETCDF3_64BIT_OFFSET", memory=1 << 20)
    ds.createDimension("Time", None)
    ds.createDimension("DateStrLen", 19)
    ds.createDimension("west_east", nx)
    ds.createDimension("south_north", ny)
    ds.createDimension("south_north_stag", ny + 1)
    ds.createDimension("west_east_stag", nx + 1)
    zsizes = {}
    for o in outs.values():
        if o.z_dim:
            zsizes[o.z_dim] = o.data.shape[0]
    for zd in ("land_cat", "soil_cat", "month", "crop"):
        if zd in zsizes:
            ds.createDimension(zd, zsizes.pop(zd))
    for zd, n in zsizes.items():
        ds.createDimension(zd, n)
    t = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
    t[0, :] = np.array([c.encode() for c in "0000-00-00_00:00:00"], "S1")
    for name, o in outs.items():
        d = o.data
        ydim = "south_north_stag" if d.shape[-2] == ny + 1 else "south_north"
        xdim = "west_east_stag" if d.shape[-1] == nx + 1 else "west_east"
        dims = ("Time",) + ((o.z_dim,) if o.z_dim else ()) + (ydim, xdim)
        v = ds.createVariable(name, "f4", dims)
        units, desc = GRID_META.get(name, (o.units, o.description))
        v.FieldType = np.int32(104)
        v.MemoryOrder = "XYZ" if o.z_dim else "XY "
        v.units = units
        v.description = desc
        v.stagger = o.stagger
        v.sr_x = np.int32(1)
        v.sr_y = np.int32(1)
        v[0] = d
    lu = attrs["luinfo"]
    clat, clon = grid.corner_attrs()
    ga = {
        "TITLE": "OUTPUT FROM ERFTOOLS.GEOG %s (geo_em compatible)" % __version__,
        "SIMULATION_START_DATE": "0000-00-00_00:00:00",
        "WEST-EAST_GRID_DIMENSION": np.int32(g.e_we),
        "SOUTH-NORTH_GRID_DIMENSION": np.int32(g.e_sn),
        "BOTTOM-TOP_GRID_DIMENSION": np.int32(0),
        "WEST-EAST_PATCH_START_UNSTAG": np.int32(1),
        "WEST-EAST_PATCH_END_UNSTAG": np.int32(nx),
        "WEST-EAST_PATCH_START_STAG": np.int32(1),
        "WEST-EAST_PATCH_END_STAG": np.int32(nx + 1),
        "SOUTH-NORTH_PATCH_START_UNSTAG": np.int32(1),
        "SOUTH-NORTH_PATCH_END_UNSTAG": np.int32(ny),
        "SOUTH-NORTH_PATCH_START_STAG": np.int32(1),
        "SOUTH-NORTH_PATCH_END_STAG": np.int32(ny + 1),
        "GRIDTYPE": "C",
        "DX": np.float32(g.dx),
        "DY": np.float32(g.dy),
        "DYN_OPT": np.int32(2),
        "CEN_LAT": np.float32(grid.fields["XLAT_M"][grid.sl_m][ny // 2, nx // 2]) if nx % 2 else
        np.float32(g.ref_lat),
        "CEN_LON": np.float32(grid.fields["XLONG_M"][grid.sl_m][ny // 2, nx // 2]) if nx % 2 else
        np.float32(g.ref_lon),
        "TRUELAT1": np.float32(g.truelat1),
        "TRUELAT2": np.float32(g.truelat2),
        "MOAD_CEN_LAT": None,
        "STAND_LON": np.float32(g.stand_lon),
        "POLE_LAT": np.float32(90.0),
        "POLE_LON": np.float32(0.0),
        "corner_lats": clat,
        "corner_lons": clon,
        "MAP_PROJ": np.int32(1),
        "MMINLU": lu["mminlu"],
        "NUM_LAND_CAT": np.int32(outs["LANDUSEF"].data.shape[0]),
        "ISWATER": np.int32(lu["iswater"]),
        "ISLAKE": np.int32(lu["islake"]),
        "ISICE": np.int32(lu["isice"]),
        "ISURBAN": np.int32(lu["isurban"]),
        "ISOILWATER": np.int32(lu["isoilwater"]),
        "grid_id": np.int32(1),
        "parent_id": np.int32(1),
        "i_parent_start": np.int32(1),
        "j_parent_start": np.int32(1),
        "i_parent_end": np.int32(g.e_we),
        "j_parent_end": np.int32(g.e_sn),
        "parent_grid_ratio": np.int32(1),
        "sr_x": np.int32(1),
        "sr_y": np.int32(1),
    }
    ga["MOAD_CEN_LAT"] = ga["CEN_LAT"]
    for k, v in attrs["flags"].items():
        ga[k] = np.int32(v)
    ga["ERFTOOLS_GEOG_VERSION"] = __version__
    ga["ERFTOOLS_GEOG_CODE_SHA256"] = code_hash()
    ga["ERFTOOLS_GEOG_PRESET"] = attrs["preset"]["name"]
    ga["ERFTOOLS_GEOG_LANDUSE"] = attrs["landuse"]
    ga["ERFTOOLS_GEOG_GRID_METHOD"] = attrs["preset"].get("grid", "wps_f32")
    ga["ERFTOOLS_GEOG_GRID_SPEC"] = json.dumps(g.as_dict())
    ga["ERFTOOLS_GEOG_PRESET_JSON"] = json.dumps(attrs["preset"], default=str)
    ga["ERFTOOLS_GEOG_SOURCES_JSON"] = json.dumps(attrs["provenance"], default=str)
    ga["ERFTOOLS_GEOG_RUNTIME_S"] = np.float32(attrs["runtime_s"])
    for k, v in ga.items():
        ds.setncattr(k, v)
    buf = ds.close()
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(bytes(buf))
    os.replace(tmp, path)
