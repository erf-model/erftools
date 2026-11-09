"""Compare both grid back ends with the geogrid reference (bitwise and max |diff|).

usage: python tests/geog/check_grid.py
"""
import os
import sys

import netCDF4
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _geog_config as _config  # noqa: E402
from erftools.geog.grid import GridSpec, ModelGrid  # noqa: E402

REF = _config.REF["modis"]
NAMES = ["XLAT_M", "XLONG_M", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V", "XLAT_C", "XLONG_C",
         "CLAT", "CLONG", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E",
         "SINALPHA", "COSALPHA", "SINALPHA_U", "COSALPHA_U", "SINALPHA_V", "COSALPHA_V"]


def main():
    ds = netCDF4.Dataset(REF)
    for method in ("wps_f32", "pyproj"):
        g = ModelGrid(GridSpec(), method=method)
        print("== grid method", method)
        for n in NAMES:
            ref = ds[n][0].astype(np.float32)
            mine = g.patch(n).astype(np.float32)
            d = np.abs(mine.astype(np.float64) - ref)
            print(f"  {n:12s} bit-identical {np.mean(mine == ref):8.5f}  max|d| {d.max():.3e}  "
                  f"max rel {np.max(d / np.maximum(np.abs(ref), 1e-30)):.2e}")
        la, lo = g.corner_attrs()
        print("  corner_lats max|d|", np.abs(la - ds.corner_lats).max(),
              " corner_lons max|d|", np.abs(lo - ds.corner_lons).max())


if __name__ == "__main__":
    main()
