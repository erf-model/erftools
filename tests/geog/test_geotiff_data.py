"""erftools.geog GeoTIFF reader tests on the real USGS 3DEP 1" tiles (skipped when absent).

  * the four SAIL tiles sit on one lattice and agree exactly where they overlap;
  * point elevations at the ARM M1 / S2 / PH sites match the ARM site elevations to < 5 m
    (checks decoding and georeferencing together);
  * slope_limited_3dep: HGT_M slope limit reached, filter conserves the domain mean.

Data root: ERFTOOLS_GEOG_3DEP_PATH (see presets.DEP3_ROOT), plus WPS_GEOG_PATH for the other
slope_limited_3dep fields.  About 30 s in total.
run: pytest tests/geog, or python tests/geog/test_geotiff_data.py
"""
import os
import sys
import unittest

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _geog_config as _config  # noqa: E402
from erftools.geog import filters, presets  # noqa: E402
from erftools.geog.dataset import open_source  # noqa: E402
from erftools.geog.grid import GridSpec  # noqa: E402
from erftools.geog.pipeline import run  # noqa: E402

# ARM site coordinates and elevations (m)
ARM_SITES = {"M1": (38.9562, -106.988, 2886.0), "S2": (38.898361, -106.94314, 3137.0),
             "PH": (38.922019, -106.9509, 2764.0)}


def _quiet(*a, **k):
    pass


def _source():
    d = presets.DEP3_ROOT
    if not d or not os.path.isdir(d) or not any(f.endswith(".tif") for f in os.listdir(d)):
        raise unittest.SkipTest("3DEP tiles not available: %s (set ERFTOOLS_GEOG_3DEP_PATH)" % d)
    return open_source(d)


def test_3dep_tiles_overlap_exactly():
    s = _source()
    worst = s.check_overlaps()
    assert worst, "no overlapping tiles found"
    for k, (d, n) in worst.items():
        assert n > 0 and d == 0.0, (k, d, n)


def test_3dep_arm_site_elevations():
    s = _source()
    for k, (lat, lon, z_arm) in ARM_SITES.items():
        x, y = s.ll_to_xy(np.array([lat]), np.array([lon]), f32=False)
        x0, y0 = int(x[0]) - 3, int(y[0]) - 3
        w = s.window(x0, x0 + 6, y0, y0 + 6)[0].astype(float)
        z = ndimage.map_coordinates(w, [[y[0] - y0], [x[0] - x0]], order=1)[0]
        assert abs(z - z_arm) < 5.0, (k, z, z_arm)


def test_slope_limited_3dep_terrain():
    _source()
    p = presets.get("slope_limited_3dep")
    _config.require_data(*[s["source"] for s in [p["landuse"]["modis"]] + p["fields"][1:]])
    gs = GridSpec()
    _, outs, _ = run(p, "modis", gs, _config.DATA_ROOT, log=_quiet)
    h = outs["HGT_M"].data.astype(float)
    raw = outs["HGT_M_RAW"].data.astype(float)
    smax = filters.max_slope(h, gs.dx, gs.dy, outs["MAPFAC_M"].data.astype(float))[0]
    assert smax <= 0.5 * (1 + 1e-3), smax
    assert abs(h.mean() - raw.mean()) < 1e-3, (h.mean(), raw.mean())
    assert np.all(np.isfinite(h)) and h.min() > 1000.0, h.min()


if __name__ == "__main__":
    sys.exit(_config.run_as_script(globals()))
