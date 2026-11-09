"""Soil state from HRRR (erftools.preprocessing.hrrr_land) on a synthetic HRRR-like grid;
one test reads a real HRRR GRIB2 file when ERFTOOLS_HRRR_GRIB is set.

run: python tests/preprocessing/test_soil_state.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pre_config as _config  # noqa: E402
from erftools.preprocessing import hrrr_land as HL  # noqa: E402
from erftools.preprocessing import terrain_remap as TR  # noqa: E402


def _setup(hgt_new=2500.0, **kw):
    land = _config.synthetic_hrrr_land(**kw)
    xo, yo = _config.output_points(land)
    regrid = HL.LandRegridder.from_dataset(land, xo, yo)
    return land, xo, yo, regrid, np.full(xo.shape, hgt_new)


def test_regridder_linear_exact():
    land, xo, yo, regrid, _ = _setup()
    xx, yy = np.meshgrid(land["x"].values, land["y"].values)
    out, nfall = regrid(1e-3 * xx - 2e-3 * yy, np.ones(xx.shape, bool))
    assert nfall == 0
    assert np.allclose(out, 1e-3 * xo - 2e-3 * yo)


def test_regridder_never_mixes_water():
    land, xo, yo, regrid, _ = _setup()
    valid = land["LANDMASK"].values > 0.5
    out, nfall = regrid(land["SKINT"].values, valid)
    assert np.allclose(out, 268.0), "a water (999) value leaked into a land point"
    # points inside the lake, away from its shore, have no land corner: nearest land point
    assert nfall > 0


def test_levels_to_layers():
    zs = HL.layer_centres(HL.NOAHMP_DZS)
    assert np.allclose(zs, [0.05, 0.25, 0.70, 1.50])
    f = (280.0 - 2.0 * _config.RUC_DEPTHS)[:, None]
    assert np.allclose(HL.soil_levels_to_layers(_config.RUC_DEPTHS, f, zs)[:, 0], 280.0 - 2.0 * zs)
    # constant beyond the source column
    out = HL.soil_levels_to_layers(np.array([0.0, 0.01]), np.array([[1.0], [2.0]]), zs)
    assert np.allclose(out[:, 0], 2.0)


def test_soil_profile_default_no_lapse_adjustment():
    land, xo, yo, regrid, hgt_new = _setup(hgt_new=2500.0)
    f, st = HL.soil_state_from_hrrr(land, regrid, hgt_new, np.ones(xo.shape), tmn=275.0)
    zs = HL.layer_centres(HL.NOAHMP_DZS)
    assert np.allclose(f["TSLB"], (280.0 - 2.0 * zs)[:, None, None])  # HRRR's profile as is
    assert st["adjust_soil_temp"] is False and st["lapse_rate_K_per_m"] is None


def test_soil_profile_and_lapse_adjustment():
    land, xo, yo, regrid, hgt_new = _setup(hgt_new=2500.0)
    lm = np.ones(xo.shape)
    f, st = HL.soil_state_from_hrrr(land, regrid, hgt_new, lm, tmn=275.0, sst=277.0,
                                    adjust_soil_temp=True)
    zs = HL.layer_centres(HL.NOAHMP_DZS)
    expect = 280.0 - 2.0 * zs - TR.STD_LAPSE * 500.0  # new ground 500 m above HRRR's
    assert np.allclose(f["TSLB"], expect[:, None, None])
    assert np.allclose(f["SMOIS"], (0.2 + 0.05 * zs)[:, None, None])
    assert np.allclose(f["SH2O"], 0.9 * f["SMOIS"])
    assert np.allclose(f["DZS"], HL.NOAHMP_DZS) and np.allclose(f["ZS"], zs)
    assert st["fallback"] is None and np.allclose(st["dhgt_hrrr_minus_new_m"], -500.0)


def test_water_points_take_water_values():
    land, xo, yo, regrid, hgt_new = _setup()
    lm = np.ones(xo.shape)
    lm[:, :3] = 0.0
    f, _ = HL.soil_state_from_hrrr(land, regrid, hgt_new, lm, tmn=275.0, sst=277.0)
    assert np.allclose(f["TSLB"][:, :, :3], 277.0)
    assert np.all(f["SMOIS"][:, :, :3] == 1.0) and np.all(f["SH2O"][:, :, :3] == 1.0)
    assert f["TSLB"][:, :, 3:].max() < 300.0


def test_sh2o_defaults_to_smois():
    land, xo, yo, regrid, hgt_new = _setup()
    land = land.drop_vars("SOIL_L")
    f, _ = HL.soil_state_from_hrrr(land, regrid, hgt_new, np.ones(xo.shape), tmn=275.0)
    assert np.array_equal(f["SH2O"], f["SMOIS"])


def test_wrfnat_like_file_needs_fallback():
    land, xo, yo, regrid, hgt_new = _setup(hgt_new=2500.0)
    nat = land.drop_vars(["SOIL_T", "SOIL_L"]).isel(soil_level=[0, 1])  # 0 and 1 cm moisture
    lm = np.ones(xo.shape)
    try:
        HL.soil_state_from_hrrr(nat, regrid, hgt_new, lm, tmn=275.0)
    except ValueError as e:
        assert "wrfprs" in str(e)
    else:
        raise AssertionError("a moisture-only, 1 cm deep soil column was accepted")
    f, st = HL.soil_state_from_hrrr(nat, regrid, hgt_new, lm, tmn=275.0, fallback="skin_tmn",
                                    adjust_soil_temp=True)
    zs = HL.layer_centres(HL.NOAHMP_DZS)
    skin = 268.0 - TR.STD_LAPSE * 500.0
    expect = skin + (275.0 - skin) * zs / HL.NOAHMP_ZBOT
    assert np.allclose(f["TSLB"], expect[:, None, None])
    assert np.allclose(f["SMOIS"], 0.2 + 0.05 * 0.01)  # the 1 cm value, held below
    assert st["fallback"].startswith("skin_tmn")


def test_terrain_remap_name_forwards():
    land, xo, yo, regrid, hgt_new = _setup()
    a, _ = TR.soil_state_from_hrrr(land, regrid, hgt_new, np.ones(xo.shape), tmn=275.0)
    b, _ = HL.soil_state_from_hrrr(land, regrid, hgt_new, np.ones(xo.shape), tmn=275.0)
    assert np.array_equal(a["TSLB"], b["TSLB"])


def test_read_real_soil_file():
    _config.require_hrrr_soil_grib()
    land = HL.read_hrrr_land([_config.HRRR_GRIB, _config.HRRR_SOIL_GRIB])
    assert np.allclose(land["SOIL_DEPTH"].values, _config.RUC_DEPTHS)  # not wrfnat's 0 / 1 cm
    assert "SOIL_T" in land and "SOIL_M" in land and "SOIL_L" not in land
    lm = land["LANDMASK"].values > 0.5
    t, m = land["SOIL_T"].values[:, lm], land["SOIL_M"].values[:, lm]
    assert 200.0 < t.min() and t.max() < 330.0, (t.min(), t.max())
    assert 0.0 <= m.min() and m.max() <= 1.0, (m.min(), m.max())
    # the deep soil is warmer and steadier than the skin in late December
    assert t[-1].std() < t[0].std() and t[-1].mean() > t[0].mean()
    try:
        HL.read_hrrr_land(_config.HRRR_SOIL_GRIB)  # soil only: no land mask or terrain
    except KeyError as e:
        assert "LANDMASK" in str(e)
    else:
        raise AssertionError("a soil-only file was accepted alone")


def test_read_real_hrrr_file():
    _config.require_hrrr_grib()
    land = HL.read_hrrr_land(_config.HRRR_GRIB)
    for v in ("LANDMASK", "HGT", "SKINT", "SOIL_M"):
        assert v in land, v
    assert land["SOIL_M"].dims == ("soil_level", "y", "x")
    assert np.all(np.diff(land["SOIL_DEPTH"].values) > 0)
    assert np.all(np.diff(land["x"].values) > 0) and np.all(np.diff(land["y"].values) > 0)


if __name__ == "__main__":
    sys.exit(_config.run_as_script(globals()))
