"""Command line: python -m erftools.geog --preset slope_limited --landuse modis --out FILE

Grid parameters use namelist.wps names (e_we, e_sn, dx, dy, ref_lat, ref_lon,
truelat1, truelat2, stand_lon); ``--namelist`` reads them from a namelist.wps.
``--grid`` takes a namelist.wps or a JSON GridSpec dict (for example the
ERFTOOLS_GEOG_GRID_SPEC attribute of an earlier output); explicit flags override both.
``--preset`` takes a preset name or a JSON file with the preset structure.
The default grid is a 500 m grid (see GridSpec); the default data root is WPS_GEOG_PATH
(environment) or presets.DATA_ROOT.
"""
import argparse
import json
import os
import re
import sys

from . import presets
from .grid import GridSpec
from .pipeline import run
from .writer import write_geo_em


def read_namelist(path):
    txt = open(path).read()
    out = {}
    for key, typ in (("e_we", int), ("e_sn", int), ("dx", float), ("dy", float),
                     ("ref_lat", float), ("ref_lon", float), ("truelat1", float),
                     ("truelat2", float), ("stand_lon", float), ("ref_x", float), ("ref_y", float)):
        mt = re.search(r"^\s*%s\s*=\s*([^,\n/]+)" % key, txt, re.M | re.I)
        if mt:
            out[key] = typ(mt.group(1).strip().strip("'"))
    mt = re.search(r"^\s*map_proj\s*=\s*'([^']+)'", txt, re.M | re.I)
    if mt:
        out["map_proj"] = mt.group(1).strip().lower()
    return out


def read_grid(path):
    """Grid parameters from a JSON GridSpec dict or a namelist.wps."""
    txt = open(path).read()
    if txt.lstrip().startswith("{"):
        d = json.loads(txt)
        return {k: v for k, v in d.items() if k in GridSpec.__dataclass_fields__}
    return read_namelist(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="slope_limited")
    ap.add_argument("--landuse", default="modis", choices=["modis", "nlcd"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--namelist")
    ap.add_argument("--grid", help="namelist.wps or JSON GridSpec file")
    ap.add_argument("--data-root", default=presets.DATA_ROOT)
    for k in ("e_we", "e_sn"):
        ap.add_argument("--" + k, type=int)
    for k in ("dx", "dy", "ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon"):
        ap.add_argument("--" + k, type=float)
    a = ap.parse_args(argv)
    kw = read_grid(a.grid) if a.grid else {}
    if a.namelist:
        kw.update(read_namelist(a.namelist))
    for k in ("e_we", "e_sn", "dx", "dy", "ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon"):
        if getattr(a, k) is not None:
            kw[k] = getattr(a, k)
    gs = GridSpec(**kw)
    if os.path.exists(a.preset):
        preset = json.load(open(a.preset))
    else:
        preset = presets.get(a.preset)
    print("erftools.geog: preset=%s landuse=%s grid=%s" % (preset["name"], a.landuse, gs.as_dict()))
    grid, outs, attrs = run(preset, a.landuse, gs, a.data_root)
    write_geo_em(a.out, grid, outs, attrs)
    print("wrote %s  (%.1f s)" % (a.out, attrs["runtime_s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
