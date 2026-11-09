"""Shared configuration for the erftools.preprocessing land-state tests.

Data locations (environment variables), only for the tests that read real files:
  ERFTOOLS_HRRR_GRIB  an HRRR GRIB2 file (wrfnat or wrfsfc, f00)
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


def require_hrrr_grib():
    if not os.path.exists(HRRR_GRIB):
        raise unittest.SkipTest("HRRR GRIB2 file not available: %r (set ERFTOOLS_HRRR_GRIB)" % HRRR_GRIB)


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
