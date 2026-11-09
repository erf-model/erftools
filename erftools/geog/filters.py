"""Terrain (and generic 2-D) filters.  All take/return [.., j, i] arrays.

WPS-equivalent (geogrid smooth_module.F, REAL*4, operate on the whole memory
array incl. halo; the outermost row/column is never updated):
  * ``smth_desmth_special(passes)`` -- 1-2-1 smoother then 1.52/-0.26
    desmoother in x then y; afterwards negative values that were >= 0 before
    smoothing are reset to the original value.
  * ``one_two_one(passes)``, ``smth_desmth(passes)``.

Alternatives (double precision, domain-edge handling by reflection):
  * ``none``
  * ``gaussian(sigma_m)``   -- isotropic Gaussian of physical std-dev sigma_m.
  * ``shapiro(order, passes)`` -- Shapiro (1970) filter of order 2/4/8, which
    removes the 2*dx wave completely per pass and leaves long waves ~untouched.
  * ``slope_limited(max_slope, ...)`` -- conservative, flux-form nonlinear
    diffusion that only acts on cell faces whose slope exceeds ``max_slope``
    (dimensionless rise/run, map-factor aware), iterated until the maximum
    face slope is <= max_slope.  Domain-integrated volume is conserved to
    round-off; terrain below the threshold is left exactly unchanged.
"""
import numpy as np

F = np.float32


# ---------------------------------------------------------------- WPS family
def smth_desmth_special(a, passes=1):
    a = np.array(a, F, copy=True)
    orig = a.copy()
    for _ in range(passes):
        for c0, c1 in ((F(0.5), F(0.25)), (F(1.52), F(-0.26))):
            # x pass into scratch (all rows, interior columns)
            s = (c0 * a[..., :, 1:-1] + c1 * (a[..., :, :-2] + a[..., :, 2:])).astype(F)
            # y pass back into a (interior rows and columns)
            a[..., 1:-1, 1:-1] = (c0 * s[..., 1:-1, :] + c1 * (s[..., :-2, :] + s[..., 2:, :])).astype(F)
    reset = (a < 0) & (orig >= 0)
    a[reset] = orig[reset]
    return a


def smth_desmth(a, passes=1):
    """smooth_module smth_desmth: same stencils but without the sign protection."""
    a = np.array(a, F, copy=True)
    for _ in range(passes):
        for c0, c1 in ((F(0.5), F(0.25)), (F(1.52), F(-0.26))):
            s = (c0 * a[..., :, 1:-1] + c1 * (a[..., :, :-2] + a[..., :, 2:])).astype(F)
            a[..., 1:-1, 1:-1] = (c0 * s[..., 1:-1, :] + c1 * (s[..., :-2, :] + s[..., 2:, :])).astype(F)
    return a


def one_two_one(a, passes=1):
    a = np.array(a, F, copy=True)
    for _ in range(passes):
        s = (F(0.5) * a[..., :, 1:-1] + F(0.25) * (a[..., :, :-2] + a[..., :, 2:])).astype(F)
        a[..., 1:-1, 1:-1] = (F(0.5) * s[..., 1:-1, :] + F(0.25) * (s[..., :-2, :] + s[..., 2:, :])).astype(F)
    return a


# ---------------------------------------------------------------- alternatives
def gaussian(a, sigma_m, dx):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(np.asarray(a, float), sigma=float(sigma_m) / float(dx), mode="reflect")


def shapiro(a, order=4, passes=1):
    """Shapiro filter: (1 - (-d2/4)^(order/2)) applied separably in x and y."""
    a = np.asarray(a, float).copy()
    p = order // 2

    def d2(f, axis):
        g = np.pad(f, [(1, 1) if ax == axis else (0, 0) for ax in range(f.ndim)], mode="reflect")
        sl = [slice(None)] * f.ndim
        lo, mid, hi = list(sl), list(sl), list(sl)
        lo[axis], mid[axis], hi[axis] = slice(0, -2), slice(1, -1), slice(2, None)
        return g[tuple(lo)] - 2 * g[tuple(mid)] + g[tuple(hi)]

    for _ in range(passes):
        for axis in (a.ndim - 1, a.ndim - 2):
            t = a.copy()
            for _ in range(p):
                t = -d2(t, axis) / 4.0
            a = a - t
    return a


def face_slopes(h, dx, dy, mapfac=None):
    """|dh/ds| on x-faces [ny, nx-1] and y-faces [ny-1, nx], with ds = d/m."""
    h = np.asarray(h, float)
    mf = np.ones_like(h) if mapfac is None else np.asarray(mapfac, float)
    mx = 0.5 * (mf[:, 1:] + mf[:, :-1])
    my = 0.5 * (mf[1:, :] + mf[:-1, :])
    return np.abs(np.diff(h, axis=1)) * mx / dx, np.abs(np.diff(h, axis=0)) * my / dy


def max_slope(h, dx, dy, mapfac=None):
    """Max slope magnitude (m/m) from centred differences at mass points, and
    max face slope; centred-difference definition matches the reference README."""
    h = np.asarray(h, float)
    mf = np.ones_like(h) if mapfac is None else np.asarray(mapfac, float)
    gx = (h[1:-1, 2:] - h[1:-1, :-2]) / (2 * dx) * mf[1:-1, 1:-1]
    gy = (h[2:, 1:-1] - h[:-2, 1:-1]) / (2 * dy) * mf[1:-1, 1:-1]
    s = np.hypot(gx, gy)
    sx, sy = face_slopes(h, dx, dy, mapfac)
    k = np.unravel_index(np.argmax(s), s.shape)
    return float(s.max()), (k[0] + 2, k[1] + 2), float(max(sx.max(), sy.max()))


def grad_mag(h, dx, dy, mapfac):
    """Centred-difference |grad h| at every point (one-sided at the edges), m/m."""
    gy, gx = np.gradient(np.asarray(h, float), dy, dx)
    return np.hypot(gx, gy) * mapfac


def curvature(h, dx, dy, mapfac):
    """Grid-scale curvature max(|d2h/dx2|, |d2h/dy2|) * ds (m/m per cell); a
    2*dx-noise indicator: equals the slope change between adjacent faces."""
    h = np.asarray(h, float)
    c = np.zeros_like(h)
    c[:, 1:-1] = np.abs(h[:, 2:] - 2 * h[:, 1:-1] + h[:, :-2]) / dx
    c[1:-1, :] = np.maximum(c[1:-1, :], np.abs(h[2:, :] - 2 * h[1:-1, :] + h[:-2, :]) / dy)
    return c * mapfac


def slope_limited(h, dx, dy, max_slope=0.4, mapfac=None, kappa=0.2, max_iter=20000, tol=1e-4,
                  metric="gradient", max_curvature=None):
    """Conservative flux-form diffusion acting only where the terrain is too steep.

    metric="gradient": a cell is flagged when its centred |grad h| (map-factor
    scaled) exceeds max_slope, or -- if max_curvature is given -- when its
    grid-scale curvature exceeds max_curvature; every face touching a flagged
    cell carries the diffusive flux kappa*(h_R - h_L).
    metric="face": a face is active when its own slope exceeds max_slope.
    Zero flux through the domain boundary, so the domain mean is conserved to
    round-off; kappa <= 0.25 keeps the explicit update stable and monotone.
    Iterates until no cell/face is flagged (within tol) or max_iter.
    """
    h = np.asarray(h, float).copy()
    mf = np.ones_like(h) if mapfac is None else np.asarray(mapfac, float)
    mx = 0.5 * (mf[:, 1:] + mf[:, :-1])
    my = 0.5 * (mf[1:, :] + mf[:-1, :])
    it = 0
    for it in range(1, max_iter + 1):
        dxh = np.diff(h, axis=1)
        dyh = np.diff(h, axis=0)
        if metric == "face":
            sx = np.abs(dxh) * mx / dx
            sy = np.abs(dyh) * my / dy
            if max(sx.max(), sy.max()) <= max_slope * (1 + tol):
                break
            ax, ay = sx > max_slope, sy > max_slope
        else:
            flag = grad_mag(h, dx, dy, mf) > max_slope * (1 + tol)
            if max_curvature is not None:
                flag |= curvature(h, dx, dy, mf) > max_curvature * (1 + tol)
            if not flag.any():
                break
            ax = flag[:, 1:] | flag[:, :-1]
            ay = flag[1:, :] | flag[:-1, :]
        fx = np.where(ax, kappa * dxh, 0.0)
        fy = np.where(ay, kappa * dyh, 0.0)
        div = np.zeros_like(h)
        div[:, :-1] += fx
        div[:, 1:] -= fx
        div[:-1, :] += fy
        div[1:, :] -= fy
        h += div
    return h, it


def apply(name, h, opts, dx, dy, mapfac=None):
    """Dispatch a filter by name. Returns (filtered, info dict)."""
    opts = dict(opts or {})
    if name in (None, "none"):
        return np.asarray(h), {}
    if name == "smth_desmth_special":
        return smth_desmth_special(h, opts.get("passes", 1)), {}
    if name == "smth_desmth":
        return smth_desmth(h, opts.get("passes", 1)), {}
    if name == "one_two_one":
        return one_two_one(h, opts.get("passes", 1)), {}
    if name == "gaussian":
        return gaussian(h, opts["sigma_m"], dx), {}
    if name == "shapiro":
        return shapiro(h, opts.get("order", 4), opts.get("passes", 1)), {}
    if name == "slope_limited":
        out, it = slope_limited(h, dx, dy, opts.get("max_slope", 0.4), mapfac,
                                opts.get("kappa", 0.2), opts.get("max_iter", 20000),
                                metric=opts.get("metric", "gradient"),
                                max_curvature=opts.get("max_curvature"))
        return out, {"iterations": it}
    raise ValueError("unknown filter %s" % name)
