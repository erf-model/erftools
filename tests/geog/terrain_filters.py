"""Terrain option study: horizontal method x filter -> max slope, stations, change, vs geogrid.

usage: python tests/geog/terrain_filters.py [OUT.txt]
"""
import os
import sys

import netCDF4
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _geog_config as _config  # noqa: E402
from erftools.geog import filters, methods, wps_engine  # noqa: E402
from erftools.geog.dataset import open_source  # noqa: E402
from erftools.geog.grid import GridSpec, ModelGrid  # noqa: E402
from compare import STATIONS, ARM_ELEV  # noqa: E402

DATA_ROOT = os.path.join(_config.DATA_ROOT, "")
REF = _config.REF["modis"]


def station_vals(h, grid):
    out = []
    for s, (la, lo) in STATIONS.items():
        i, j = grid.latlon_to_ij(la, lo)
        i0, j0 = int(np.floor(i)), int(np.floor(j))
        fx, fy = i - i0, j - j0
        a = h[j0 - 1:j0 + 1, i0 - 1:i0 + 1]
        out.append((1 - fy) * ((1 - fx) * a[0, 0] + fx * a[0, 1]) + fy * ((1 - fx) * a[1, 0] + fx * a[1, 1]))
    return out


def main(outpath=None):
    lines = []
    pr = lines.append
    gs = GridSpec()
    src = open_source(DATA_ROOT + "topo_gmted2010_30s/")
    ref = netCDF4.Dataset(REF)["HGT_M"][0].astype(float)
    gw = ModelGrid(gs, "wps_f32")
    gp = ModelGrid(gs, "pyproj")
    raws = {}
    raw_mem = wps_engine.calc_continuous(gw, src, dict(interp="average_gcell(4.0)+four_pt+average_4pt",
                                                       fill_missing=0.0), None, gs.dx)[0]
    raws["wps_four_pt"] = (gw, raw_mem)
    for meth in ("bilinear", "bicubic", "cell_bilinear", "area_mean"):
        raws[meth] = (gp, methods.sample_continuous(gp, src, meth, {"supersample": 8})[0])
    flist = [("none", {}), ("smth_desmth_special", {"passes": 1}), ("smth_desmth_special", {"passes": 3}),
             ("shapiro", {"order": 4, "passes": 1}), ("gaussian", {"sigma_m": 500.0}), ("slope_limited", {"max_slope": 0.6}),
             ("slope_limited", {"max_slope": 0.5}), ("slope_limited", {"max_slope": 0.4}),
             ("slope_limited", {"max_slope": 0.3}),
             ("slope_limited", {"max_slope": 0.5, "metric": "face"}),
             ("slope_limited", {"max_slope": 0.5, "max_curvature": 0.4})]
    pr("Terrain options on the SAIL 500 m grid (GMTED2010 30\" source). slope = |grad h| from centred")
    pr("differences x map factor (m/m); 'face' = max one-sided face slope; dmean = domain-mean change")
    pr("(conservation); rms/max vs geogrid = difference from the reference geo_em HGT_M.")
    pr("%-14s %-34s %6s %5s %5s %5s %6s %6s %6s %7s %7s %7s %6s %6s" % (
        "horizontal", "filter", "slope", "deg", "face", "curv", "rmsChg", "maxChg", "dmean",
        "M1", "S2", "PH", "rmsRef", "maxRef"))
    pr("%-14s %-34s %6s %5s %5s %5s %6s %6s %6s %7.0f %7.0f %7.0f  (ARM station elevations)" % (
        "", "", "", "", "", "", "", "", "", ARM_ELEV["M1"], ARM_ELEV["S2"], ARM_ELEV["PH"]))
    for rname, (g, raw) in raws.items():
        mf = g.patch("MAPFAC_M").astype(float)
        rawp = g.patch(raw).astype(float)
        for fname, fo in flist:
            if fname in ("smth_desmth_special", "one_two_one", "smth_desmth"):
                hp = g.patch(filters.apply(fname, raw.astype(np.float32), fo, gs.dx, gs.dy)[0]).astype(float)
            else:
                hp = np.asarray(filters.apply(fname, rawp, fo, gs.dx, gs.dy, mapfac=mf)[0], float)
            s, loc, sface = filters.max_slope(hp, gs.dx, gs.dy, mf)
            d = hp - rawp
            st = station_vals(hp, g)
            dr = hp - ref
            label = fname + ("(" + ",".join("%s=%s" % kv for kv in fo.items()) + ")" if fo else "")
            cv = filters.curvature(hp, gs.dx, gs.dy, mf)[1:-1, 1:-1].max()
            pr("%-14s %-34s %6.3f %5.1f %5.2f %5.2f %6.2f %6.1f %6.3f %7.1f %7.1f %7.1f %6.2f %6.1f" % (
                rname, label, s, np.degrees(np.arctan(s)), sface, cv, np.sqrt(np.mean(d ** 2)), np.abs(d).max(),
                d.mean(), st[0], st[1], st[2], np.sqrt(np.mean(dr ** 2)), np.abs(dr).max()))
    txt = "\n".join(lines)
    print(txt)
    if outpath:
        open(outpath, "w").write(txt + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
