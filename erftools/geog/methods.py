"""Non-WPS horizontal methods (double precision, independent of geogrid code).

Every method samples a source dataset onto the model memory grid (domain +
halo) given only the model grid's ij->lat/lon mapping and the source's
lat/lon->index mapping.

Continuous fields
  * ``nearest``         value of the source cell containing the model point.
  * ``bilinear``        bilinear in source-index space at the model point
                        (missing-aware: weights of missing neighbours dropped
                        and the rest renormalised).
  * ``bicubic``         cubic B-spline (scipy.ndimage.map_coordinates order 3);
                        only for sources without missing values.
  * ``cell_bilinear``   average of the bilinear surface over the model cell,
                        sampled on an N x N sub-grid (= conservative remap of
                        the bilinear reconstruction; best when source is
                        coarser than the target).
  * ``area_mean``       area-weighted mean of the source cells overlapping the
                        model cell (N x N sub-grid point-in-cell estimate of the
                        overlap areas; conservative; best when the source is
                        finer than the target).
  Missing results can be filled with ``fill: nearest_valid`` (nearest valid
  source cell, Euclidean in index space).

Categorical fields
  * ``area_fraction``   fraction of the model-cell area covered by each source
                        category (same N x N overlap estimate).  For sources
                        that already store per-category fractions (NLCD 9s),
                        the area-weighted mean of those fractions.
  * ``nearest``         one-hot of the category at the model point.

The sub-grid overlap estimate converges to the exact polygon overlap as 1/N;
with N=8 and 30"/9" sources the per-cell area-fraction error is <~2 %.  The
LCC map factor varies by <1e-4 inside a 500 m cell, so equal sub-grid weights
are equal-area weights to that accuracy.
"""
import numpy as np
from scipy import ndimage

F = np.float32


def _subpoints(grid, n):
    """lat/lon of N x N sub-cell centres for every memory mass cell -> [n*n, ny, nx]."""
    I, J = np.meshgrid(grid.i_m.astype(float), grid.j_m.astype(float))
    off = (np.arange(n) + 0.5) / n - 0.5
    ox, oy = np.meshgrid(off, off)
    ii = I[None] + ox.ravel()[:, None, None]
    jj = J[None] + oy.ravel()[:, None, None]
    if grid.method == "wps_f32":
        lat, lon = grid.proj.ij_to_ll(ii.astype(F), jj.astype(F))
        return lat.astype(float), lon.astype(float)
    return grid.proj.ij_to_ll(ii, jj)


def _points(grid, stagger="M"):
    return (grid.fields["XLAT_" + stagger].astype(float), grid.fields["XLONG_" + stagger].astype(float))


class _Window:
    """Source raw values over the index window covering a set of points."""

    def __init__(self, src, x, y, pad=4):
        self.x0 = int(np.floor(np.nanmin(x))) - pad
        self.y0 = int(np.floor(np.nanmin(y))) - pad
        x1 = int(np.ceil(np.nanmax(x))) + pad
        y1 = int(np.ceil(np.nanmax(y))) + pad
        self.raw = src.window(self.x0, x1, self.y0, y1).astype(float)   # [z, y, x]
        msg = float(src.missing_value)
        self.valid = ~(np.isnan(self.raw) | (self.raw == msg))
        sf = 1.0 if src.scale_factor is None else float(src.scale_factor)
        self.val = np.where(self.valid, self.raw * sf, np.nan)

    def local(self, x, y):
        return x - self.x0, y - self.y0


def _nearest_valid_index(valid2d):
    """For each window cell, index of nearest valid cell (EDT)."""
    if valid2d.all():
        return None
    _, (iy, ix) = ndimage.distance_transform_edt(~valid2d, return_indices=True)
    return iy, ix


def sample_continuous(grid, src, method, opts, stagger="M"):
    """Return [k, j, i] float64 on the memory grid for source levels z0..z1."""
    opts = dict(opts or {})
    n = int(opts.get("supersample", 8))
    if method in ("cell_bilinear", "area_mean"):
        lat, lon = _subpoints(grid, n)
    else:
        lat, lon = _points(grid, stagger)
        lat, lon = lat[None], lon[None]
    x, y = src.ll_to_xy(lat, lon, f32=False)
    w = _Window(src, x, y)
    lx, ly = w.local(x, y)            # 1-based global -> 0-based local index = g - x0
    out = []
    for z in range(w.val.shape[0]):
        v = w.val[z]
        ok = w.valid[z]
        if opts.get("fill") == "nearest_valid_source":
            idx = _nearest_valid_index(ok)
            if idx is not None:
                v = v[idx]
                ok = np.ones_like(ok)
        if method in ("nearest", "area_mean"):
            ix = np.floor(lx + 0.5).astype(int)
            iy = np.floor(ly + 0.5).astype(int)
            s = v[iy, ix]
            if method == "area_mean":
                good = ~np.isnan(s)
                cnt = good.sum(axis=0)
                with np.errstate(invalid="ignore"):
                    r = np.where(cnt > 0, np.nansum(s, axis=0) / np.maximum(cnt, 1), np.nan)
            else:
                r = s[0]
        elif method in ("bilinear", "cell_bilinear"):
            vv = np.where(ok, v, 0.0)
            num = ndimage.map_coordinates(vv, [ly.ravel(), lx.ravel()], order=1, mode="nearest")
            den = ndimage.map_coordinates(ok.astype(float), [ly.ravel(), lx.ravel()], order=1,
                                          mode="nearest")
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(den > 1e-12, num / den, np.nan).reshape(lx.shape)
            if method == "cell_bilinear":
                r = np.nanmean(r, axis=0) if np.isnan(r).any() else r.mean(axis=0)
            else:
                r = r[0]
        elif method == "bicubic":
            if not ok.all():
                raise ValueError("bicubic needs a source without missing values")
            r = ndimage.map_coordinates(v, [ly.ravel(), lx.ravel()], order=3,
                                        mode="nearest").reshape(lx.shape)[0]
        else:
            raise ValueError("unknown continuous method %s" % method)
        if opts.get("fill") == "nearest_valid" and np.isnan(r).any():
            idx = _nearest_valid_index(ok)
            if idx is not None:
                cx = np.clip(np.floor(lx.mean(axis=0) + 0.5).astype(int), 0, v.shape[1] - 1)
                cy = np.clip(np.floor(ly.mean(axis=0) + 0.5).astype(int), 0, v.shape[0] - 1)
                fv = v[idx][cy, cx]
                r = np.where(np.isnan(r), fv, r)
        out.append(r)
    return np.array(out)


def sample_std(grid, src, n=8):
    """Std-dev of the bilinear source surface inside each model cell (sub-grid
    variability resolvable by the source; a lower bound on true sub-grid std)."""
    lat, lon = _subpoints(grid, n)
    x, y = src.ll_to_xy(lat, lon, f32=False)
    w = _Window(src, x, y)
    lx, ly = w.local(x, y)
    r = ndimage.map_coordinates(np.nan_to_num(w.val[0]), [ly.ravel(), lx.ravel()], order=1,
                                mode="nearest").reshape(lx.shape)
    return r.std(axis=0)


def sample_categorical(grid, src, method, opts, cat_range):
    """Return category fractions [cat, j, i] (float64) on the memory grid."""
    opts = dict(opts or {})
    n = int(opts.get("supersample", 8))
    c0, c1 = cat_range
    nc = c1 - c0 + 1
    if method == "area_fraction":
        lat, lon = _subpoints(grid, n)
    elif method == "nearest":
        lat, lon = _points(grid)
        lat, lon = lat[None], lon[None]
    else:
        raise ValueError("unknown categorical method %s" % method)
    x, y = src.ll_to_xy(lat, lon, f32=False)
    w = _Window(src, x, y)
    lx, ly = w.local(x, y)
    ix = np.floor(lx + 0.5).astype(int)
    iy = np.floor(ly + 0.5).astype(int)
    npt, ny, nx = ix.shape
    if src.type == "categorical":
        cat = w.raw[0][iy, ix]
        good = w.valid[0][iy, ix]
        c = cat.astype(int)
        good &= (c >= c0) & (c <= c1)
        cell = np.broadcast_to(np.arange(ny * nx).reshape(ny, nx), ix.shape)
        cnt = np.bincount((cell[good] * nc + (c[good] - c0)).ravel(), minlength=ny * nx * nc)
        cnt = cnt.reshape(ny, nx, nc).transpose(2, 0, 1).astype(float)
    else:
        cnt = np.zeros((nc, ny, nx))
        for z in range(w.val.shape[0]):
            lev = src.z0 + z
            if c0 <= lev <= c1:
                s = w.val[z][iy, ix]
                cnt[lev - c0] = np.nansum(s, axis=0)
    tot = cnt.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(tot > 0, cnt / tot, np.nan)
    return frac
