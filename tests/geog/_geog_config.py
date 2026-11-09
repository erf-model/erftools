"""Shared configuration for the erftools.geog tests.

Data locations (environment variables):
  WPS_GEOG_PATH          WPS_GEOG_V4 root (default: erftools.geog.presets.DATA_ROOT)
  ERFTOOLS_GEOG_REF_DIR  directory with the geogrid reference files
                         geo_em.d01.modis_default.nc and geo_em.d01.nlcd2011_9s.nc
Tests that need data raise unittest.SkipTest (reported as skipped by pytest) when the
data are not available. Every test file also runs as a plain script.
"""
import os
import sys
import unittest

try:
    import erftools.geog  # noqa: F401
except ImportError:  # running from a source checkout without an install
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from erftools.geog import presets  # noqa: E402

DATA_ROOT = os.environ.get("WPS_GEOG_PATH", presets.DATA_ROOT)
REF_DIR = os.environ.get("ERFTOOLS_GEOG_REF_DIR", "")
REF = {"modis": os.path.join(REF_DIR, "geo_em.d01.modis_default.nc"),
       "nlcd": os.path.join(REF_DIR, "geo_em.d01.nlcd2011_9s.nc")}


def require_data(*sources):
    """Skip unless the WPS_GEOG root (and the given dataset sub-directories) exist."""
    if not os.path.isdir(DATA_ROOT):
        raise unittest.SkipTest("WPS_GEOG data root not available: %s (set WPS_GEOG_PATH)" % DATA_ROOT)
    for s in sources:
        if not os.path.exists(os.path.join(DATA_ROOT, s, "index")):
            raise unittest.SkipTest("dataset not available: %s" % os.path.join(DATA_ROOT, s))


def require_ref(landuse):
    if not os.path.exists(REF[landuse]):
        raise unittest.SkipTest("geogrid reference not available: %s (set ERFTOOLS_GEOG_REF_DIR)" % REF[landuse])


def run_as_script(namespace):
    """Minimal runner for ``python tests/geog/test_*.py`` (no pytest needed)."""
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
