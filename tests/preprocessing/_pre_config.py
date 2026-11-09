"""Shared configuration for the erftools.preprocessing land-state tests.

Data locations (environment variables), only for the tests that read real files:
  ERFTOOLS_HRRR_GRIB  an HRRR GRIB2 file (wrfnat or wrfsfc, f00)
  ERFTOOLS_HRRR_SOIL_GRIB  HRRR's RUC soil levels at the same time (TSOIL / SOILW messages
                      of wrfprsf00 in 2021 HRRR)
Tests that need data raise unittest.SkipTest when the data are not available. Every test
file also runs as a plain script (python tests/preprocessing/test_*.py).
"""
import os
import sys
import unittest

import numpy as np
import xarray as xr

try:
    import erftools.preprocessing  # noqa: F401
except ImportError:  # running from a source checkout without an install
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

HRRR_GRIB = os.environ.get("ERFTOOLS_HRRR_GRIB", "")
HRRR_SOIL_GRIB = os.environ.get("ERFTOOLS_HRRR_SOIL_GRIB", "")


def require_hrrr_grib():
    if not os.path.exists(HRRR_GRIB):
        raise unittest.SkipTest("HRRR GRIB2 file not available: %r (set ERFTOOLS_HRRR_GRIB)" % HRRR_GRIB)


def require_hrrr_soil_grib():
    require_hrrr_grib()
    if not os.path.exists(HRRR_SOIL_GRIB):
        raise unittest.SkipTest("HRRR soil GRIB2 file not available: %r (set ERFTOOLS_HRRR_SOIL_GRIB)"
                                % HRRR_SOIL_GRIB)


def synthetic_geo(ny=6, nx=8, water_cols=2):
    """A geo_em-like Dataset (no Time dimension): land everywhere except the last
    `water_cols` columns (MODIS water, category 17). Monthly fields vary by month only, so
    their date interpolation can be checked by hand."""
    m = np.arange(1, 13, dtype=np.float64)[:, None, None] * np.ones((12, ny, nx))
    hgt = np.tile(np.linspace(1000.0, 3000.0, nx), (ny, 1))
    lu = np.full((ny, nx), 10.0)
    lu[:, nx - water_cols:] = 17.0
    land = (lu != 17.0).astype(np.float64)
    geo = xr.Dataset({
        "HGT_M": (("south_north", "west_east"), hgt),
        "LU_INDEX": (("south_north", "west_east"), lu),
        "SCT_DOM": (("south_north", "west_east"), np.where(land > 0, 6.0, 14.0)),
        "LANDMASK": (("south_north", "west_east"), land),
        "SOILTEMP": (("south_north", "west_east"), np.full((ny, nx), 290.0)),
        "GREENFRAC": (("month", "south_north", "west_east"), 0.05 * m),       # 0.05 .. 0.60
        "LAI12M": (("month", "south_north", "west_east"), 0.5 * m),           # 0.5 .. 6.0
        "ALBEDO12M": (("month", "south_north", "west_east"), 10.0 + m),       # percent
        "SNOALB": (("south_north", "west_east"), np.full((ny, nx), 70.0)),    # percent
    })
    geo.attrs.update({"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "NUM_LAND_CAT": 21})
    return geo


RUC_DEPTHS = np.array([0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00])  # m


def synthetic_hrrr_land(ny=12, nx=14, dx=3000.0, hgt=None, lake=((4, 7), (5, 8))):
    """A read_hrrr_land-like Dataset on a regular 3 km grid (dims y, x; x / y in metres):
    land except a rectangular lake (rows lake[0], columns lake[1], end exclusive). Soil:
    SOIL_T = 280 - 2 z [K] (z depth in m), SOIL_M = 0.2 + 0.05 z, SOIL_L = 0.9 SOIL_M;
    on the lake every field is 999 so any leak of water values is obvious."""
    x = 1.0e5 + dx * np.arange(nx)
    y = -2.0e5 + dx * np.arange(ny)
    hgt = np.full((ny, nx), 2000.0) if hgt is None else np.asarray(hgt, np.float64)
    landmask = np.ones((ny, nx))
    (j0, j1), (i0, i1) = lake
    landmask[j0:j1, i0:i1] = 0.0
    z = RUC_DEPTHS[:, None, None]
    one = np.ones((1, ny, nx))
    soil_t = (280.0 - 2.0 * z) * one
    soil_m = (0.2 + 0.05 * z) * one
    ds = xr.Dataset({
        "LANDMASK": (("y", "x"), landmask), "HGT": (("y", "x"), hgt),
        "SKINT": (("y", "x"), np.full((ny, nx), 268.0)),
        "SOIL_T": (("soil_level", "y", "x"), soil_t), "SOIL_M": (("soil_level", "y", "x"), soil_m),
        "SOIL_L": (("soil_level", "y", "x"), 0.9 * soil_m),
    }, coords={"x": x, "y": y, "SOIL_DEPTH": ("soil_level", RUC_DEPTHS)})
    for v in ("SKINT", "SOIL_T", "SOIL_M", "SOIL_L"):
        ds[v] = ds[v].where(ds["LANDMASK"] > 0.5, 999.0)
    return ds


def output_points(land, n=25, pad=1.5):
    """A regular n x n set of output points inside the source grid, `pad` source cells in."""
    x1, y1 = land["x"].values, land["y"].values
    d = x1[1] - x1[0]
    xo = np.linspace(x1[0] + pad * d, x1[-1] - pad * d, n)
    yo = np.linspace(y1[0] + pad * d, y1[-1] - pad * d, n)
    return np.meshgrid(xo, yo)  # (south_north, west_east)


def run_as_script(namespace):
    """Minimal runner for ``python tests/preprocessing/test_*.py`` (no pytest needed)."""
    n_ok = n_skip = 0
    for k, f in list(namespace.items()):
        if k.startswith("test_") and callable(f):
            try:
                f()
            except unittest.SkipTest as e:
                n_skip += 1
                print("skip", k, "-", e)
                continue
            n_ok += 1
            print("ok  ", k)
    print("%d passed, %d skipped" % (n_ok, n_skip))
    return 0
