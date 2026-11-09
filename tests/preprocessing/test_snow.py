"""Snow from HRRR (hrrr_land.snow_from_hrrr) on a synthetic HRRR-like grid with hills: HRRR's
smooth 3 km terrain and SWE, and a rougher output terrain.

run: python tests/preprocessing/test_snow.py
"""
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pre_config as _config  # noqa: E402
from erftools.preprocessing import hrrr_land as HL  # noqa: E402

NY, NX = 20, 22


def _hills(x, y):
    return 2600.0 + 600.0 * np.sin(x / 9.0e3) * np.cos(y / 11.0e3)


def _setup(swe_of_z, density=250.0, bumps=150.0):
    land = _config.synthetic_hrrr_land(ny=NY, nx=NX)
    xx, yy = np.meshgrid(land["x"].values, land["y"].values)
    land["HGT"] = (("y", "x"), _hills(xx, yy))
    lake = land["LANDMASK"].values < 0.5
    swe = np.where(lake, 999.0, swe_of_z(land["HGT"].values))
    land["SNOW"] = (("y", "x"), swe)
    land["SNOWH"] = (("y", "x"), np.where(lake, 999.0, swe / density))
    land["CANWAT"] = (("y", "x"), np.where(lake, 999.0, 0.3))
    land["SNOWC"] = (("y", "x"), np.where(lake, 999.0, np.where(swe > 0, 1.0, 0.0)))
    xo, yo = _config.output_points(land, n=60, pad=4.0)
    # the 500 m terrain: HRRR's hills plus ridges and valleys HRRR cannot resolve
    hgt_new = _hills(xo, yo) + bumps * np.sin(xo / 700.0) * np.sin(yo / 900.0)
    regrid = HL.LandRegridder.from_dataset(land, xo, yo)
    return land, regrid, hgt_new, np.ones(xo.shape)


def _linear(z):
    return 0.4 * (z - 1500.0)  # 440 .. 920 kg m-2 over the hills, 0.4 kg m-2 per m


def _snowline(z):
    return 600.0 * np.clip((z - 2400.0) / 900.0, 0.0, 1.0) ** 1.5


def test_elevation_method_exact_for_linear_swe():
    land, regrid, hgt_new, lm = _setup(_linear)
    f, st = HL.snow_from_hrrr(land, regrid, hgt_new, lm, method="elevation", cap=None)
    assert np.allclose(f["SNOW"], _linear(hgt_new), rtol=0, atol=1e-6)
    g, _ = HL.snow_from_hrrr(land, regrid, hgt_new, lm, method="interp")
    assert np.abs(g["SNOW"] - _linear(hgt_new)).max() > 0.3 * 150.0  # the bumps are missed
    assert np.allclose(st["snow_gradient_kg_m2_per_km_p10_p50_p90"], 400.0)


def test_monotone_with_height_and_better_than_interp():
    land, regrid, hgt_new, lm = _setup(_snowline)
    f, _ = HL.snow_from_hrrr(land, regrid, hgt_new, lm, method="elevation")
    g, _ = HL.snow_from_hrrr(land, regrid, hgt_new, lm, method="interp")
    truth = _snowline(hgt_new)
    z = hgt_new.ravel()
    order = np.argsort(z)
    bins = np.array_split(order, 12)
    means = [f["SNOW"].ravel()[b].mean() for b in bins]
    assert np.all(np.diff(means) >= 0.0), means
    # rank correlation with height (ties averaged; the truth itself has many zeros): the
    # elevation method follows the terrain, interp less so
    def spearman(a):
        return spearmanr(a.ravel(), z)[0]
    assert spearman(f["SNOW"]) > 0.94 and spearman(f["SNOW"]) > spearman(g["SNOW"])
    assert np.abs(f["SNOW"] - truth).mean() < 0.5 * np.abs(g["SNOW"] - truth).mean()


def test_no_negative_snow_and_cap():
    land, regrid, hgt_new, lm = _setup(_snowline, bumps=400.0)
    for method in ("elevation", "interp"):
        f, _ = HL.snow_from_hrrr(land, regrid, hgt_new, lm, method=method)
        assert f["SNOW"].min() >= 0.0 and f["SNOWH"].min() >= 0.0, method
    f, st = HL.snow_from_hrrr(land, regrid, hgt_new, lm, method="elevation", cap="local_max")
    hrrr_land = land["SNOW"].values[land["LANDMASK"].values > 0.5]
    assert f["SNOW"].max() <= hrrr_land.max() + 1e-9
    assert st["snow_points_capped"] > 0  # 400 m bumps rise above HRRR's hills


def test_zero_over_water_and_no_lake_leak():
    land, regrid, hgt_new, lm = _setup(_linear)
    lm[:, :10] = 0.0
    for method in ("interp", "elevation"):
        f, _ = HL.snow_from_hrrr(land, regrid, hgt_new, lm, method=method)
        for k in ("SNOW", "SNOWH", "SNOWC", "CANWAT"):
            assert np.all(f[k][:, :10] == 0.0), (method, k)
        assert f["SNOW"].max() < 999.0 and np.allclose(f["CANWAT"][:, 10:], 0.3), method


def test_density_kept_and_default():
    land, regrid, hgt_new, lm = _setup(_linear, density=250.0)
    for method in ("interp", "elevation"):
        f, _ = HL.snow_from_hrrr(land, regrid, hgt_new, lm, method=method)
        snowy = f["SNOW"] > 0
        assert np.allclose(f["SNOWH"][snowy], f["SNOW"][snowy] / 250.0), method
        f, _ = HL.snow_from_hrrr(land.drop_vars("SNOWH"), regrid, hgt_new, lm, method=method)
        assert np.allclose(f["SNOWH"][snowy], 0.005 * f["SNOW"][snowy]), method  # Noah-MP's rule


def test_default_is_plain_interpolation():
    land, regrid, hgt_new, lm = _setup(_snowline)
    f, st = HL.snow_from_hrrr(land, regrid, hgt_new, lm)
    g, _ = HL.snow_from_hrrr(land, regrid, hgt_new, lm, method="interp")
    assert st["snow_method"] == "interp" and "snow_radius_m" not in st
    assert np.array_equal(f["SNOW"], g["SNOW"])
    # equal to the land-only interpolation of HRRR's SWE itself
    expect, _ = regrid(land["SNOW"].values, land["LANDMASK"].values > 0.5)
    assert np.allclose(f["SNOW"], expect)


def test_snowc_threshold_and_hrrr_cover():
    land, regrid, hgt_new, lm = _setup(lambda z: np.where(z > 2600.0, 10.0, 9.9))
    f, _ = HL.snow_from_hrrr(land, regrid, hgt_new, lm, method="interp", snowc="threshold")
    assert set(np.unique(f["SNOWC"])) <= {0.0, 1.0}
    assert np.array_equal(f["SNOWC"] == 1.0, f["SNOW"] >= HL.SNOWC_THRESHOLD)
    land["SNOWC"] = (("y", "x"), np.full((NY, NX), 0.6))
    f, st = HL.snow_from_hrrr(land, regrid, hgt_new, lm)  # snowc='hrrr' is the default
    assert np.allclose(f["SNOWC"][f["SNOW"] > 0], 0.6) and st["snowc"] == "hrrr"
    assert np.all(f["SNOWC"][f["SNOW"] == 0] == 0.0)
    try:
        HL.snow_from_hrrr(land.drop_vars("SNOWC"), regrid, hgt_new, lm)
    except KeyError as e:
        assert "SNOWC" in str(e)
    else:
        raise AssertionError("snowc='hrrr' without HRRR SNOWC was accepted")


def test_flat_or_sparse_neighbourhood_has_no_gradient():
    land = _config.synthetic_hrrr_land(ny=NY, nx=NX)  # flat 2000 m
    land["SNOW"] = (("y", "x"), np.full((NY, NX), 300.0))
    slope, fmax = HL.local_gradient(land["SNOW"].values, land["HGT"].values,
                                    land["LANDMASK"].values > 0.5, land["x"].values, land["y"].values)
    assert np.all(slope == 0.0) and np.allclose(fmax, 300.0)


def test_read_real_hrrr_snow():
    _config.require_hrrr_grib()
    land = HL.read_hrrr_land(_config.HRRR_GRIB)
    for v in ("SNOW", "SNOWH", "SNOWC", "CANWAT"):
        assert v in land and land[v].dims == ("y", "x"), v
    lm = land["LANDMASK"].values > 0.5
    assert float(land["SNOWC"].values[lm].mean()) > 0.0
    assert 0.0 <= float(land["SNOWC"].min()) and float(land["SNOWC"].max()) <= 1.0
    assert float(land["SNOW"].min()) >= 0.0 and float(land["SNOWH"].min()) >= 0.0


if __name__ == "__main__":
    sys.exit(_config.run_as_script(globals()))
