"""Acceptance test and cross-check against the geogrid reference.

  1. wps_equivalent preset vs geogrid (both land uses): every common variable,
     exact-match fraction / max|d| / mean|d| / max rel; PASS criteria:
     categorical 100 % exact, continuous within 4 float32 ulp of the field
     magnitude, HGT_M <= 0.01 m.
  2. slope_limited preset vs geogrid: differences reported and explained
     (not tuned): grid, terrain (rms/max/mean, slope, stations vs ARM),
     categorical agreement, continuous fields on common land points.

usage: python tests/geog/validate.py OUTDIR
  (expects geo_em.d01.optionB_wps_{modis,nlcd}.nc and geo_em.d01.optionB_{modis,nlcd}.nc;
   writes OUTDIR/validation_output.txt; reference files from ERFTOOLS_GEOG_REF_DIR)
"""
import io
import os
import sys
from contextlib import redirect_stdout

import netCDF4
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _geog_config import REF  # noqa: E402
from compare import ARM_ELEV, STATIONS, bilinear_at, compare  # noqa: E402
from erftools.geog import filters  # noqa: E402


def crosscheck(mine, ref, out=print):
    A, B = netCDF4.Dataset(mine), netCDF4.Dataset(ref)
    g = lambda ds, n: np.asarray(ds[n][0], float)  # noqa: E731
    out("  grid (same projection definition, pyproj double precision vs geogrid REAL*4):")
    for n in ("XLAT_M", "XLONG_M", "XLAT_U", "XLONG_V", "XLAT_C", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E",
              "SINALPHA", "COSALPHA"):
        d = np.abs(g(A, n) - g(B, n))
        rel = d / np.maximum(np.abs(g(B, n)), 1e-30)
        out("    %-9s max|d| %.3e  max rel %.2e" % (n, d.max(), rel.max()))
    h, r = g(A, "HGT_M"), g(B, "HGT_M")
    mf = g(A, "MAPFAC_M")
    d = h - r
    sA = filters.max_slope(h, 500.0, 500.0, mf)
    sB = filters.max_slope(r, 500.0, 500.0, mf)
    out("  terrain HGT_M: mine-ref mean %+.2f m, rms %.2f m, max|d| %.1f m (at j,i=%s); "
        "max slope mine %.3f (%.1f deg) ref %.3f (%.1f deg)" % (
            d.mean(), np.sqrt((d ** 2).mean()), np.abs(d).max(),
            tuple(int(x) for x in np.unravel_index(np.argmax(np.abs(d)), d.shape)),
            sA[0], np.degrees(np.arctan(sA[0])), sB[0], np.degrees(np.arctan(sB[0]))))
    if "HGT_M_RAW" in A.variables:
        d0 = g(A, "HGT_M_RAW") - r
        out("                 (before our filter: mean %+.2f rms %.2f max %.1f; filter changed rms %.2f "
            "max %.1f m)" % (d0.mean(), np.sqrt((d0 ** 2).mean()), np.abs(d0).max(),
                             np.sqrt(((h - g(A, 'HGT_M_RAW')) ** 2).mean()),
                             np.abs(h - g(A, "HGT_M_RAW")).max()))
    for s, (la, lo) in STATIONS.items():
        ha, i, j = bilinear_at(A["HGT_M"][0], None, None, la, lo)
        hb, _, _ = bilinear_at(B["HGT_M"][0], None, None, la, lo)
        out("    %-3s mine %7.1f  ref %7.1f  (mine-ref %+6.1f)  ARM %5.0f  mine-ARM %+6.1f  ref-ARM %+6.1f" % (
            s, ha, hb, ha - hb, ARM_ELEV[s], ha - ARM_ELEV[s], hb - ARM_ELEV[s]))
    out("  categorical agreement (fraction of the 201x201 cells):")
    for n in ("LANDMASK", "LU_INDEX", "SCT_DOM", "SCB_DOM"):
        a, b = g(A, n), g(B, n)
        out("    %-8s %.4f   (%d cells differ)" % (n, np.mean(a == b), int(np.sum(a != b))))
    for n in ("LANDUSEF", "SOILCTOP", "SOILCBOT"):
        a, b = np.asarray(A[n][0], float), np.asarray(B[n][0], float)
        out("    %-8s mean |d frac| %.4f, max %.3f; domain-mean fractions max |d| %.4f" % (
            n, np.abs(a - b).sum(axis=0).mean() / 2, np.abs(a - b).max(),
            np.abs(a.mean(axis=(1, 2)) - b.mean(axis=(1, 2))).max()))
    land = (g(A, "LANDMASK") == 1) & (g(B, "LANDMASK") == 1)
    out("  continuous fields on common land points (%d cells):" % land.sum())
    for n in ("SOILTEMP", "ALBEDO12M", "GREENFRAC", "LAI12M", "SNOALB", "VAR_SSO"):
        a, b = np.asarray(A[n][0], float), np.asarray(B[n][0], float)
        if a.ndim == 2:
            a, b = a[None], b[None]
        dd = (a - b)[:, land]
        bb = b[:, land]
        out("    %-9s mean(ref) %10.4g  mean d %+10.3g  rms %9.3g  max|d| %9.3g  rel rms %.3f" % (
            n, bb.mean(), dd.mean(), np.sqrt((dd ** 2).mean()), np.abs(dd).max(),
            np.sqrt((dd ** 2).mean()) / max(abs(bb.mean()), 1e-30)))


def main(outdir):
    buf = io.StringIO()
    with redirect_stdout(buf):
        for lu in ("modis", "nlcd"):
            print("=" * 100)
            print("1. wps_equivalent (%s) vs geogrid %s" % (lu, os.path.basename(REF[lu])))
            compare(os.path.join(outdir, "geo_em.d01.optionB_wps_%s.nc" % lu), REF[lu], detail=True)
        for lu in ("modis", "nlcd"):
            print("=" * 100)
            print("2. slope_limited (%s) vs geogrid %s  [cross-check, differences explained in README]" % (
                lu, os.path.basename(REF[lu])))
            crosscheck(os.path.join(outdir, "geo_em.d01.optionB_%s.nc" % lu), REF[lu])
    txt = buf.getvalue()
    print(txt)
    with open(os.path.join(outdir, "validation_output.txt"), "w") as fh:
        fh.write(txt)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
