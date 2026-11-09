"""geogrid-equivalent horizontal processing (method family ``wps``).

Python transcription of the logic in WPS v4.4 geogrid/src/process_tile_module.F
(calc_field, landmask / dominant-category handling) and proc_point_module.F
(accum_categorical / accum_continuous = "each source pixel goes to its nearest
model grid point"; get_point = interpolator chain at a model point), executed
in IEEE single precision.

Known, documented deviation: for domains that straddle a source-tile boundary
geogrid's result for a few grid points near that boundary depends on its tile
traversal order (a point may receive a nearest-neighbour value from one tile
and pixel counts from the next).  Here all tiles are accumulated first and only
points without any pixel fall back to the interpolator chain.
"""
import numpy as np

from . import f32 as m
from .wps_interp import Tile, interp_chain, parse_chain

F = np.float32
NAN_WPS = F(1.0e20)          # misc_definitions_module: NAN = 1.E20
MASK_VAL = {"water": F(0.0), "land": F(1.0), None: F(-1.0)}


def _mem_points(grid, stagger="M"):
    lat = grid.fields["XLAT_" + stagger]
    lon = grid.fields["XLONG_" + stagger]
    return lat, lon


def _tiles_for(src, rx, ry):
    tx, ty = src.tile_of(rx, ry)
    key = tx.astype(np.int64) * 10 ** 7 + ty
    return tx, ty, key


def accumulate(grid, src, mode, cat_range=None):
    """Pixel -> nearest model point accumulation over every tile touched by the
    memory region.  mode 'categorical' returns counts[cat, j, i]; mode
    'continuous' returns (sum[k, j, i], n[k, j, i]).  Also returns the boolean
    'assigned' map (geogrid's new_pts)."""
    lat, lon = _mem_points(grid)
    ny, nx = lat.shape
    i_start, j_start = grid.i_m[0], grid.j_m[0]
    i_end, j_end = grid.i_m[-1], grid.j_m[-1]
    rx, ry = src.ll_to_xy(lat, lon)
    tx, ty, key = _tiles_for(src, rx, ry)
    # source-index window that can map into the memory region (+ margin)
    margin = 4
    xlo, xhi = int(np.floor(rx.min())) - margin, int(np.ceil(rx.max())) + margin
    ylo, yhi = int(np.floor(ry.min())) - margin, int(np.ceil(ry.max())) + margin
    if mode == "categorical":
        c0, c1 = cat_range
        acc = np.zeros((c1 - c0 + 1, ny, nx), np.float64)
        cnt = None
    else:
        acc = np.zeros((src.nz, ny, nx), np.float64)
        cnt = np.zeros((src.nz, ny, nx), np.float64)
    assigned = np.zeros((ny, nx), bool)
    for k in np.unique(key):
        t0x, t0y = int(k // 10 ** 7), int(k % 10 ** 7)
        # interior pixels of the tile, clipped to the useful window
        x0, x1 = max(t0x, xlo), min(t0x + src.tile_x - 1, xhi)
        y0, y1 = max(t0y, ylo), min(t0y + src.tile_y - 1, yhi)
        if x0 > x1 or y0 > y1:
            continue
        t = src.read_tile(t0x, t0y, xr=(x0, x1), yr=(y0, y1))
        if t is None:
            continue
        arr, ax0, ay0 = t
        b = src.bdr
        X, Y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        plat, plon = src.xy_to_ll(X.astype(F), Y.astype(F))
        gx, gy = grid.proj.ll_to_ij(plat, plon)
        gx = (gx - F(0.5)) * F(1.0) + F(0.5)
        gy = (gy - F(0.5)) * F(1.0) + F(0.5)
        inside = ((F(i_start) <= gx) & (gx <= F(i_end)) & (F(j_start) <= gy) & (gy <= F(j_end)))
        ii = m.nint(gx[inside]) - i_start
        jj = m.nint(gy[inside]) - j_start
        vals = arr[:, Y[inside] - ay0, X[inside] - ax0]   # [z, npix]
        msg = src.missing_value
        if mode == "categorical":
            v = vals[0]
            good = v != msg
            cat = v.astype(np.int64)      # Fortran int() truncation
            inr = good & (cat >= c0) & (cat <= c1)
            np.add.at(acc, (cat[inr] - c0, jj[inr], ii[inr]), 1.0)
            assigned[jj[inr], ii[inr]] = True
        else:
            for z in range(src.nz):
                v = vals[z]
                good = v != msg
                np.add.at(acc[z], (jj[good], ii[good]), v[good].astype(np.float64))
                np.add.at(cnt[z], (jj[good], ii[good]), 1.0)
                assigned[jj[good], ii[good]] = True
        del b
    return acc, cnt, assigned


def point_values(grid, src, chain, levels, stagger="M", only=None):
    """get_point for every memory point (or the subset 'only'); returns [k, j, i]
    float32 raw values, msg where the chain failed."""
    lat, lon = _mem_points(grid, stagger)
    shape = lat.shape
    rx, ry = src.ll_to_xy(lat.ravel(), lon.ravel())
    out = np.full((len(levels),) + (lat.size,), src.missing_value, F)
    sel = np.arange(lat.size) if only is None else np.nonzero(only.ravel())[0]
    if sel.size == 0 or not chain:
        return out.reshape((len(levels),) + shape)
    tx, ty, key = _tiles_for(src, rx[sel], ry[sel])
    for k in np.unique(key):
        pts = sel[key == k]
        t = src.read_tile(int(k // 10 ** 7), int(k % 10 ** 7))
        if t is None:
            continue
        arr, ax0, ay0 = t
        for li, lev in enumerate(levels):
            tile = Tile(arr[lev - src.z0], ax0, ay0)
            out[li, pts] = interp_chain(tile, rx[pts], ry[pts], src.missing_value, chain)
    return out.reshape((len(levels),) + shape)


def calc_continuous(grid, src, spec, landmask, dom_dx):
    """calc_field for a continuous destination field (one priority level)."""
    gcell, chain = parse_chain(spec["interp"])
    stagger = spec.get("stagger", "M")
    lat, _ = _mem_points(grid, stagger)
    shape = lat.shape
    levels = list(range(src.z0, src.z1 + 1))
    fill = F(spec["fill_missing"]) if spec.get("fill_missing") is not None else NAN_WPS
    mask_val = MASK_VAL[spec.get("masked")]
    use_mask = landmask is not None and stagger == "M"
    msg = src.missing_value
    sp = False
    if gcell is not None:
        thr = F(gcell) * max(src.dx, src.dy) * F(111.0)
        sp = bool(thr <= F(dom_dx) / F(1000.0))
    field = np.full((len(levels),) + shape, fill, F)
    level_domain = np.zeros(shape, bool)
    if sp:
        s, n, assigned = accumulate(grid, src, "continuous")
        field_sum = s.astype(F)
        cnt = n.astype(F)
        rest = ~assigned
        if use_mask:
            rest &= landmask != mask_val
        vals = point_values(grid, src, chain, levels, stagger, only=rest)
        good = (vals != msg) & rest[None]
        field_sum = np.where(good, vals, field_sum)
        cnt = np.where(good, F(1.0), cnt)
        level_domain = assigned | np.any(good, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            q = (field_sum / cnt).astype(F)
        field = np.where(cnt > 0, q, fill).astype(F)
        if use_mask:
            field = np.where((landmask == mask_val)[None], fill, field)
    else:
        active = np.ones(shape, bool)
        if use_mask:
            active = landmask != mask_val
        vals = point_values(grid, src, chain, levels, stagger, only=active)
        good = (vals != msg) & active[None]
        field = np.where(good, vals, fill).astype(F)
        level_domain = np.any(good, axis=0)
    if src.scale_factor is not None:
        doit = level_domain[None] & (field != fill)
        field = np.where(doit, field * src.scale_factor, field).astype(F)
    return field


def calc_categorical(grid, src, spec, landmask, cat_range, dom_dx):
    """calc_field for a categorical destination field: returns (unnormalised)
    category weights [cat, j, i] float32."""
    gcell, chain = parse_chain(spec["interp"])
    lat, _ = _mem_points(grid)
    shape = lat.shape
    c0, c1 = cat_range
    msg = src.missing_value
    mask_val = MASK_VAL[spec.get("masked")]
    use_mask = landmask is not None
    if src.type == "categorical":
        cnt, _, assigned = accumulate(grid, src, "categorical", cat_range)
        field = cnt.astype(F)
        rest = ~assigned
        if use_mask:
            rest &= landmask != mask_val
        vals = point_values(grid, src, chain, [src.z0], only=rest)[0]
        good = rest & (vals != msg)
        cat = np.where(good, vals, 0).astype(np.int64)
        inr = good & (cat >= c0) & (cat <= c1)
        jj, ii = np.nonzero(inr)
        field[:, rest] = 0.0
        field[cat[inr] - c0, jj, ii] += F(1.0)
        if use_mask:
            field[:, landmask == mask_val] = 0.0
        level_domain = assigned | good
    else:
        # continuous source of per-category fractions (e.g. NLCD 9s, 40 levels)
        sub = dict(spec)
        sub["fill_missing"] = None
        field = calc_continuous(grid, src, sub, None if spec.get("masked") is None else landmask, dom_dx)
        # pad / cut to destination categories
        out = np.zeros((c1 - c0 + 1,) + shape, F)
        for li, lev in enumerate(range(src.z0, src.z1 + 1)):
            if c0 <= lev <= c1:
                out[lev - c0] = field[li]
        field = out
        level_domain = None
    return field


def fractions(field, fill):
    """process_tile 'Find fractions' (sequential REAL*4 sum over categories)."""
    s = np.zeros(field.shape[1:], F)
    for k in range(field.shape[0]):
        s = (s + field[k]).astype(F)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = (field / s[None]).astype(F)
    return np.where((s > 0)[None], q, fill).astype(F)


def landmask_from(frac, water_cats, c0, fill):
    wt = np.full(frac.shape[1:], F(-1.0), F)
    bad = np.zeros(frac.shape[1:], bool)
    for c in water_cats:
        v = frac[c - c0]
        miss = v == fill
        bad |= miss
        wt = np.where(wt < 0, F(0.0), wt)
        wt = (wt + v).astype(F)
    lm = np.where(wt < F(0.5), F(1.0), F(0.0)).astype(F)
    return np.where(bad, F(-1.0), lm).astype(F)


def dominant_with_mask(frac, landmask, water_cats, c0):
    """LU_INDEX rule: on land the largest non-water category, on water the largest water category."""
    nc = frac.shape[0]
    cats = np.arange(c0, c0 + nc)
    iswat = np.isin(cats, water_cats)
    dom = np.full(frac.shape[1:], F(c0 - 1), F)
    best = np.zeros(frac.shape[1:], F)
    land = landmask == F(1.0)
    for k in range(nc):
        v = frac[k]
        upd = (v > best) & np.where(land, ~iswat[k], iswat[k])
        dom = np.where(upd, F(cats[k]), dom)
        best = np.where(upd, v, best)
    return dom


def dominant_plain(frac, c0, fill):
    nc = frac.shape[0]
    dom = np.full(frac.shape[1:], F(c0 - 1), F)
    best = np.zeros(frac.shape[1:], F)
    for k in range(nc):
        v = frac[k]
        upd = (v > best) & (v != fill)
        dom = np.where(upd, F(c0 + k), dom)
        best = np.where(upd, v, best)
    return np.where(dom == F(c0 - 1), fill, dom).astype(F)
