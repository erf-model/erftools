"""erftools.geog acceptance tests on the real WPS_GEOG_V4 data (skipped when it is absent).

  * wps_equivalent reproduces the geogrid reference geo_em files bit for bit (every
    variable geogrid wrote), for MODIS and NLCD land use;
  * slope_limited: HGT_M slope limit reached, filter conserves the domain mean.

Data roots: WPS_GEOG_PATH, ERFTOOLS_GEOG_REF_DIR (see _config.py). About 40 s in total.
run: pytest tests/geog, or python tests/geog/test_presets_data.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _geog_config as _config  # noqa: E402
from erftools.geog import filters, presets  # noqa: E402
from erftools.geog.grid import GridSpec  # noqa: E402
from erftools.geog.pipeline import run  # noqa: E402

CATEGORICAL = ("LU_INDEX", "SCT_DOM", "SCB_DOM", "LANDMASK")


def _quiet(*a, **k):
    pass


def _wps_vs_geogrid(landuse):
    import netCDF4
    _config.require_ref(landuse)
    p = presets.get("wps_equivalent")
    _config.require_data(*[s["source"] for s in [p["landuse"][landuse]] + p["fields"]])
    _, outs, _ = run(p, landuse, GridSpec(), _config.DATA_ROOT, log=_quiet)
    ref = netCDF4.Dataset(_config.REF[landuse])
    names = [v for v in ref.variables if v != "Times"]
    bad = []
    for n in names:
        assert n in outs, "missing variable %s" % n
        a = np.asarray(outs[n].data, np.float32)
        b = np.asarray(ref[n][0], np.float32)
        if a.shape != b.shape or not np.array_equal(a, b):
            bad.append(n)
    ref.close()
    assert not bad, "not bit-identical to geogrid: %s" % bad
    return len(names)


def test_wps_equivalent_modis_bit_identical():
    _wps_vs_geogrid("modis")


def test_wps_equivalent_nlcd_bit_identical():
    _wps_vs_geogrid("nlcd")


def test_slope_limited_terrain():
    p = presets.get("slope_limited")
    _config.require_data(*[s["source"] for s in [p["landuse"]["modis"]] + p["fields"]])
    gs = GridSpec()
    _, outs, attrs = run(p, "modis", gs, _config.DATA_ROOT, log=_quiet)
    h = outs["HGT_M"].data.astype(float)
    raw = outs["HGT_M_RAW"].data.astype(float)
    smax = filters.max_slope(h, gs.dx, gs.dy, outs["MAPFAC_M"].data.astype(float))[0]
    assert smax <= 0.5 * (1 + 1e-3), smax
    assert abs(h.mean() - raw.mean()) < 1e-3, (h.mean(), raw.mean())
    for n in CATEGORICAL:
        assert np.all(np.isfinite(outs[n].data)), n


if __name__ == "__main__":
    sys.exit(_config.run_as_script(globals()))
