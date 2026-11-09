"""Static LSM fields from a geo_em-style dataset (terrain_remap.map_static_fields with
lsm_fields=True), on synthetic data.

run: python tests/preprocessing/test_static_lsm.py
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pre_config as _config  # noqa: E402
from erftools.preprocessing import terrain_remap as TR  # noqa: E402

DATE = pd.Timestamp("2021-12-24 00:00")
# real.exe weights for 2021-12-24: day 358, Dec 15 = 349, Jan 15 = 349 + 31
W_JAN, W_DEC = 9.0 / 31.0, 22.0 / 31.0


def _fields(geo=None, **kw):
    geo = _config.synthetic_geo() if geo is None else geo
    out, attrs = TR.map_static_fields(geo, DATE, lsm_fields=True, **kw)
    return geo, out, attrs


def test_monthly_values_at_date():
    geo, out, _ = _fields()
    land = geo["LANDMASK"].values > 0.5
    # month m holds 0.05 m (GREENFRAC), 0.5 m (LAI12M), 10 + m (ALBEDO12M)
    assert np.allclose(out["VEGFRA"][land], 100.0 * 0.05 * (W_JAN * 1 + W_DEC * 12))
    assert np.allclose(out["LAI"], 0.5 * (W_JAN * 1 + W_DEC * 12))
    assert np.allclose(out["ALBBCK"][land], (10.0 + W_JAN * 1 + W_DEC * 12) / 100.0)
    assert np.allclose(out["SHDMIN"], 5.0) and np.allclose(out["SHDMAX"], 60.0)


def test_mid_month_is_that_month():
    geo = _config.synthetic_geo()
    out, _ = TR.map_static_fields(geo, pd.Timestamp("2021-07-15"), lsm_fields=True)
    land = geo["LANDMASK"].values > 0.5
    assert np.allclose(out["VEGFRA"][land], 100.0 * 0.05 * 7)
    assert np.allclose(out["ALBBCK"][land], 0.17)


def test_snoalb_and_water_values():
    geo, out, _ = _fields(sst=280.0)
    land = geo["LANDMASK"].values > 0.5
    assert np.allclose(out["SNOALB"][land], 0.70)
    for name in ("ALBBCK", "SNOALB"):
        assert np.allclose(out[name][~land], TR.WATER_ALBEDO)
    assert np.all(out["VEGFRA"][~land] == 0.0)
    assert np.allclose(out["TMN"][land], 290.0 - TR.STD_LAPSE * geo["HGT_M"].values[land])
    assert np.allclose(out["TMN"][~land], 280.0)


def test_month_axis_found_by_name():
    geo = _config.synthetic_geo()
    swapped = geo.transpose("south_north", "west_east", "month")
    _, a, _ = _fields(geo)
    _, b, _ = _fields(swapped)
    for k in ("VEGFRA", "LAI", "ALBBCK", "SHDMIN", "SHDMAX"):
        assert np.array_equal(a[k], b[k]), k


def test_greenfrac_in_percent_raises():
    geo = _config.synthetic_geo()
    geo["GREENFRAC"] = 100.0 * geo["GREENFRAC"]
    try:
        _fields(geo)
    except ValueError as e:
        assert "GREENFRAC" in str(e)
    else:
        raise AssertionError("percent GREENFRAC was accepted")


def test_missing_input_named():
    geo = _config.synthetic_geo().drop_vars("ALBEDO12M")
    try:
        _fields(geo)
    except KeyError as e:
        assert "ALBEDO12M" in str(e)
    else:
        raise AssertionError("missing ALBEDO12M was accepted")
    # the non-LSM mapping does not need it
    TR.map_static_fields(geo, DATE, lsm_fields=False)


def test_add_geogrid_land_fields_writes_time_dim():
    geo = _config.synthetic_geo()
    ny, nx = geo["HGT_M"].shape
    inp = xr.Dataset({"SST": (("Time", "south_north", "west_east"), np.full((1, ny, nx), 275.0))})
    inp = TR.add_geogrid_land_fields(inp, geo, DATE, lsm_fields=True)
    for name in ("VEGFRA", "SHDMIN", "SHDMAX", "LAI", "ALBBCK", "SNOALB", "TMN"):
        assert inp[name].dims == ("Time", "south_north", "west_east"), name
        assert inp[name].dtype == np.float32, name
    land = geo["LANDMASK"].values > 0.5
    assert np.allclose(inp["TMN"].values[0][~land], 275.0)
    assert inp.attrs["ISWATER"] == 17


if __name__ == "__main__":
    sys.exit(_config.run_as_script(globals()))
