"""erftools.geog unit tests on synthetic data; no WPS_GEOG data needed.

run: pytest tests/geog, or python tests/geog/test_units.py
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _geog_config as _config  # noqa: E402,F401  (puts erftools on the path from a source checkout)
from erftools.geog import filters  # noqa: E402
from erftools.geog.f32 import nint  # noqa: E402
from erftools.geog.grid import GridSpec, ModelGrid  # noqa: E402
from erftools.geog.wps_interp import Tile, interp_chain  # noqa: E402

F = np.float32
MSG = F(-999.0)


def _tile(fun, n=20):
    y, x = np.mgrid[1:n + 1, 1:n + 1].astype(F)
    return Tile(fun(x, y).astype(F), 1, 1)


def test_four_pt_linear():
    t = _tile(lambda x, y: 2 * x + 3 * y)
    xx = np.array([3.25, 10.5, 7.0], F)
    yy = np.array([4.75, 2.0, 7.0], F)
    v = interp_chain(t, xx, yy, MSG, [("four_pt", None)])
    assert np.allclose(v, 2 * xx + 3 * yy, atol=1e-4)


def test_sixteen_pt_linear_exact():
    t = _tile(lambda x, y: 2 * x - y + 5)
    xx = np.array([5.3, 9.9], F)
    yy = np.array([6.6, 3.2], F)
    v = interp_chain(t, xx, yy, MSG, [("sixteen_pt", None)])
    assert np.allclose(v, 2 * xx - yy + 5, atol=1e-3)


def test_fallthrough_and_search():
    a = np.full((10, 10), MSG, F)
    a[7, 8] = 42.0                    # global (x=9, y=8)
    t = Tile(a, 1, 1)
    v = interp_chain(t, np.array([3.2], F), np.array([3.6], F), MSG,
                     [("four_pt", None), ("average_4pt", None), ("search", 1200)])
    assert v[0] == F(42.0)
    v = interp_chain(t, np.array([3.2], F), np.array([3.6], F), MSG, [("four_pt", None)])
    assert v[0] == MSG


def test_average_4pt_skips_missing():
    a = np.arange(1, 17, dtype=F).reshape(4, 4)
    a[1, 1] = MSG
    t = Tile(a, 1, 1)
    v = interp_chain(t, np.array([2.5], F), np.array([2.5], F), MSG, [("average_4pt", None)])
    assert np.isclose(v[0], (a[1, 2] + a[2, 1] + a[2, 2]) / 3)


def test_nint_half_away():
    x = np.array([0.5, 1.5, -0.5, np.nextafter(F(2.5), F(0))], F)
    assert list(nint(x)) == [1, 2, -1, 2]


def test_slope_limited_conservative():
    rng = np.random.default_rng(0)
    h = np.cumsum(rng.normal(0, 200, (60, 60)), axis=1)
    out, it = filters.slope_limited(h, 500.0, 500.0, 0.4)
    s = filters.max_slope(out, 500.0, 500.0)[0]
    assert s <= 0.4 * (1 + 1e-3), s
    assert abs(out.mean() - h.mean()) < 1e-8 * abs(h).max()


def test_smth_desmth_special_constant():
    h = np.full((10, 12), 1234.5, F)
    assert np.array_equal(filters.smth_desmth_special(h, 2), h)


def test_grid_backends_agree():
    try:
        import pyproj  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("pyproj not installed")
    gs = GridSpec()
    a, b = ModelGrid(gs, "wps_f32"), ModelGrid(gs, "pyproj")
    for n in ("XLAT_M", "XLONG_M", "XLAT_U", "XLONG_V", "MAPFAC_M"):
        assert np.abs(a.fields[n].astype(float) - b.fields[n]).max() < 2e-5, n
    # reference point sits on mass point (101, 101)
    i, j = b.latlon_to_ij(gs.ref_lat, gs.ref_lon)
    assert abs(i - 101) < 1e-9 and abs(j - 101) < 1e-9


if __name__ == "__main__":
    sys.exit(_config.run_as_script(globals()))
