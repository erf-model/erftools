"""Noah-MP wrfinput (noahmp_input.add_noahmp_land_state, check_noahmp_inputs) end to end on
synthetic data placed in HRRR's projection; one test checks a real atmosphere-only wrfinput
when ERFTOOLS_WRFINPUT is set.

run: python tests/preprocessing/test_noahmp_input.py
"""
import os
import sys

import json

import cartopy.crs as ccrs
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pre_config as _config  # noqa: E402
from erftools.preprocessing import hrrr_land as HL  # noqa: E402
from erftools.preprocessing import noahmp_input as NI  # noqa: E402
from erftools.preprocessing.hrrr import hrrr_projection  # noqa: E402

NY, NX = 6, 8  # output grid, = synthetic_geo


def _case():
    x0, y0 = HL.hrrr_xy(np.array(39.0), np.array(-107.0))
    land = _config.synthetic_hrrr_land(ny=10, nx=10)
    land = land.assign_coords(x=land["x"] - land["x"][0] + x0 - 13.5e3,
                              y=land["y"] - land["y"][0] + y0 - 13.5e3)
    land["SNOW"] = (("y", "x"), np.full((10, 10), 300.0))
    land["SNOWH"] = (("y", "x"), np.full((10, 10), 1.0))
    land["CANWAT"] = (("y", "x"), np.full((10, 10), 0.2))
    land["SNOWC"] = (("y", "x"), np.full((10, 10), 0.9))
    xo, yo = np.meshgrid(x0 - 1750.0 + 500.0 * np.arange(NX), y0 - 1250.0 + 500.0 * np.arange(NY))
    ll = ccrs.Geodetic().transform_points(hrrr_projection, xo, yo)
    m2 = ("Time", "south_north", "west_east")
    one = np.ones((1, NY, NX))
    inp = xr.Dataset({"XLAT": (m2, ll[None, ..., 1]), "XLONG": (m2, ll[None, ..., 0]),
                      "HGT": (m2, 2100.0 * one), "TSK": (m2, 270.0 * one), "SST": (m2, 276.0 * one)})
    inp.attrs.update({"WEST-EAST_GRID_DIMENSION": NX + 1, "SOUTH-NORTH_GRID_DIMENSION": NY + 1,
                      "DX": 500.0, "DY": 500.0, "TRUELAT1": 38.5, "TRUELAT2": 38.5,
                      "STAND_LON": -97.5, "MAP_PROJ": 1, "SIMULATION_START_DATE": "2021-12-24_00:00:00",
                      "I_PARENT_START": 1, "J_PARENT_START": 1, "PARENT_GRID_RATIO": 1})
    return inp, _config.synthetic_geo(NY, NX), land


def test_end_to_end_passes_checker():
    inp, geo, land = _case()
    before = NI.check_noahmp_inputs(inp.copy())
    assert not before["ok"] and "SNOW" in before["missing"] and "attribute GRID_ID" in before["missing"]
    out, rec = NI.add_noahmp_land_state(inp, geo, land)
    res = NI.check_noahmp_inputs(out)
    assert res["ok"], res
    assert out.attrs["GRID_ID"] == 1
    assert out["TSLB"].dims == ("Time", "soil_layers_stag", "south_north", "west_east")
    assert out["DZS"].dims == ("Time", "soil_layers_stag")
    land_pts = geo["LANDMASK"].values > 0.5
    assert np.all(out["XLAND"].values[0][land_pts] == 1.0) and np.all(out["XLAND"].values[0][~land_pts] == 2.0)
    assert np.allclose(out["SNOW"].values[0][land_pts], 300.0)  # flat HRRR snow, no gradient
    assert np.allclose(out["SNOWH"].values[0][land_pts], 1.0, rtol=1e-6)
    assert np.all(out["SNOW"].values[0][~land_pts] == 0.0)
    assert np.allclose(out["SNOWC"].values[0][land_pts], 0.9)  # HRRR's cover, the default
    assert np.all(out["SNOWC"].values[0][~land_pts] == 0.0)
    # defaults: TSK and soil temperature as given, plain snow interpolation, HRRR's cover
    assert np.allclose(out["TSK"].values, 270.0)
    assert np.allclose(out["TSLB"].values[0][0][land_pts], 280.0 - 2.0 * 0.05)
    assert rec["soil"]["fallback"] is None and rec["snow"]["snow_method"] == "interp"
    opts = json.loads(out.attrs["LAND_STATE_OPTIONS"])
    assert opts["adjust_tsk"] is False and opts["adjust_soil_temp"] is False
    assert opts["snow_method"] == "interp" and opts["snow"]["snowc"] == "hrrr"


def test_terrain_options_on():
    inp, geo, land = _case()
    out, rec = NI.add_noahmp_land_state(inp, geo, land, adjust_tsk=True, adjust_soil_temp=True,
                                        snow_method="elevation", radius=6000.0, cap=None)
    land_pts = geo["LANDMASK"].values > 0.5
    # TSK on land: HRRR skin (268 K) lapse-adjusted from 2000 m to 2100 m; water keeps 270 K
    assert np.allclose(out["TSK"].values[0][land_pts], 268.0 - 0.0065 * 100.0, atol=1e-4)
    assert np.allclose(out["TSK"].values[0][~land_pts], 270.0)
    assert np.allclose(out["TSLB"].values[0][0][land_pts], 280.0 - 2.0 * 0.05 - 0.0065 * 100.0,
                       atol=1e-4)
    assert np.allclose(out["SNOW"].values[0][land_pts], 300.0)  # flat HRRR snow, no gradient
    opts = json.loads(out.attrs["LAND_STATE_OPTIONS"])
    assert opts["adjust_tsk"] and opts["adjust_soil_temp"] and opts["snow_method"] == "elevation"
    assert opts["snow"]["radius"] == 6000.0 and opts["snow"]["cap"] is None
    assert opts["snow"]["max_gradient"] == 1.0  # defaults filled in
    assert NI.check_noahmp_inputs(out)["ok"]


def test_checker_flags_bad_values():
    inp, geo, land = _case()
    out, _ = NI.add_noahmp_land_state(inp, geo, land)
    bad = out.copy(deep=True)
    bad["SNOWC"] = bad["SNOWC"] + 2.0
    bad["XLAND"] = 3.0 - bad["XLAND"]
    bad["TSLB"] = bad["TSLB"] * np.nan
    bad.attrs.pop("MMINLU")
    res = NI.check_noahmp_inputs(bad)
    text = " ".join(res["invalid"])
    assert not res["ok"] and "attribute MMINLU" in res["missing"]
    for key in ("SNOWC", "XLAND disagrees", "TSLB"):
        assert key in text, key
    short = out.isel(soil_layers_stag=slice(0, 3))
    assert any("DZS" in s for s in NI.check_noahmp_inputs(short)["invalid"])


def test_lowercase_grid_id_accepted():
    inp, geo, land = _case()
    out, _ = NI.add_noahmp_land_state(inp, geo, land)
    out.attrs["grid_id"] = out.attrs.pop("GRID_ID")
    assert NI.check_noahmp_inputs(out)["ok"]


def test_real_atmosphere_only_wrfinput_lists_missing():
    _config.require_wrfinput()
    res = NI.check_noahmp_inputs(_config.WRFINPUT)
    print("   missing:", res["missing"])
    assert not res["ok"]


if __name__ == "__main__":
    sys.exit(_config.run_as_script(globals()))
