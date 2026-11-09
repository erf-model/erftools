"""Compare an erftools.geog geo_em file with a geogrid reference; prints a table.

usage: python tests/geog/compare.py MINE.nc REF.nc [--mismatch-detail]
"""
import os
import sys

import netCDF4
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _geog_config as _config  # noqa: E402,F401

CATEGORICAL = ["LU_INDEX", "SCT_DOM", "SCB_DOM", "LANDMASK"]
STATIONS = {"M1": (38.956158, -106.987856), "S2": (38.898361, -106.94314), "PH": (38.922019, -106.9509)}
ARM_ELEV = {"M1": 2886.0, "S2": 3137.0, "PH": 2764.0}


def bilinear_at(field, xlat, xlon, lat, lon, grid=None):
    """Bilinear interpolation of a mass-point field at (lat, lon) using the WPS
    float32 projection to locate fractional indices."""
    from erftools.geog.grid import GridSpec, ModelGrid
    g = grid or ModelGrid(GridSpec(), method="pyproj")
    i, j = g.latlon_to_ij(lat, lon)
    i0, j0 = int(np.floor(i)), int(np.floor(j))
    fx, fy = i - i0, j - j0
    a = field[j0 - 1:j0 + 1, i0 - 1:i0 + 1].astype(float)
    return ((1 - fy) * ((1 - fx) * a[0, 0] + fx * a[0, 1]) + fy * ((1 - fx) * a[1, 0] + fx * a[1, 1])), i, j


def compare(mine, ref, detail=False, out=print):
    A = netCDF4.Dataset(mine)
    B = netCDF4.Dataset(ref)
    names = [v for v in B.variables if v in A.variables and v != "Times"]
    missing = [v for v in B.variables if v not in A.variables]
    rows = []
    out("%-11s %-5s %10s %11s %11s %10s %9s" % ("field", "type", "exact_frac", "max|d|", "mean|d|",
                                               "max_rel", "status"))
    lm = B["LANDMASK"][0]
    for n in names:
        a = np.asarray(A[n][:], np.float64)
        b = np.asarray(B[n][:], np.float64)
        if a.shape != b.shape:
            out("%-11s shape mismatch %s vs %s" % (n, a.shape, b.shape))
            continue
        d = np.abs(a - b)
        exact = float(np.mean(a == b))
        scale = np.maximum(np.abs(b), 1e-30)
        rel = d / scale
        typ = "cat" if n in CATEGORICAL else "cont"
        rows.append((n, typ, exact, d.max(), d.mean(), rel.max()))
        if typ == "cat":
            status = "PASS" if exact == 1.0 else "DIFF"
        elif n == "HGT_M":
            status = "PASS" if d.max() <= 0.01 else "DIFF"
        else:
            # float32 rounding: <= a few ulp of the field's magnitude
            tol = 4 * np.finfo(np.float32).eps * max(np.abs(b).max(), 1e-30)
            status = "PASS" if (d.max() <= tol or exact == 1.0) else "DIFF"
        out("%-11s %-5s %10.6f %11.3e %11.3e %10.2e %9s" % (n, typ, exact, d.max(), d.mean(), rel.max(),
                                                          status))
        if detail and status == "DIFF":
            idx = np.argwhere(d > 0)
            out("    %d differing values; first few (index, mine, ref):" % len(idx))
            for k in idx[:6]:
                out("     ", tuple(int(x) for x in k), a[tuple(k)], b[tuple(k)])
    if missing:
        out("not in mine: %s" % ", ".join(missing))
    # global attributes
    for k in ("MMINLU", "NUM_LAND_CAT", "ISWATER", "ISLAKE", "ISICE", "ISURBAN", "ISOILWATER", "CEN_LAT",
              "CEN_LON", "corner_lats", "corner_lons", "FLAG_LAI12M", "FLAG_VAR_SSO"):
        va, vb = getattr(A, k, None), getattr(B, k, None)
        same = np.array_equal(np.asarray(va), np.asarray(vb))
        if not same:
            out("  attr %s: mine %s ref %s" % (k, va, vb))
    # stations
    out("station terrain (bilinear on HGT_M):")
    for s, (la, lo) in STATIONS.items():
        ha, i, j = bilinear_at(A["HGT_M"][0], None, None, la, lo)
        hb, _, _ = bilinear_at(B["HGT_M"][0], None, None, la, lo)
        out("  %-3s i=%7.2f j=%7.2f  mine %8.2f  ref %8.2f  diff %+8.3f  ARM %6.0f  mine-ARM %+7.1f" % (
            s, i, j, ha, hb, ha - hb, ARM_ELEV[s], ha - ARM_ELEV[s]))
    return rows


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2], detail="--mismatch-detail" in sys.argv)
