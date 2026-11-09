"""Vectorised Python re-implementations of geogrid's point interpolators.

Transcribed from WPS v4.4 geogrid/src/interp_module.F (read-only reference),
including its REAL*4 arithmetic order, edge clamping and fall-through rules:
each method either returns a value or "falls through" to the next method of
the chain; the chain returns ``msgval`` when every method falls through.

All functions operate on one source tile ``arr`` (float32 [y, x] for one
level) whose element [0, 0] has global 1-based indices (x0, y0), and on
float32 arrays of fractional global source indices (xx, yy).

Methods (names as in GEOGRID.TBL): four_pt, sixteen_pt, average_4pt,
average_16pt, wt_average_4pt, wt_average_16pt, nearest_neighbor, search[(depth)].
"""
from collections import deque

import numpy as np

F = np.float32
TINY = F(1.0e-20)


class Tile:
    def __init__(self, arr, x0, y0):
        self.a = arr
        self.x0 = int(x0)
        self.y0 = int(y0)
        self.sx, self.sy = self.x0, self.y0
        self.ex = self.x0 + arr.shape[1] - 1
        self.ey = self.y0 + arr.shape[0] - 1

    def get(self, ix, iy):
        return self.a[iy - self.y0, ix - self.x0]


def _floor(x):
    return np.floor(x).astype(np.int64)


def _ceil(x):
    return np.ceil(x).astype(np.int64)


def four_pt(t, xx, yy, msg):
    n = xx.shape[0]
    val = np.full(n, msg, F)
    mnx, mny, mxx, mxy = _floor(xx), _floor(yy), _ceil(xx), _ceil(yy)
    inb = (mnx >= t.sx) & (mxx <= t.ex) & (mny >= t.sy) & (mxy <= t.ey)
    ok = inb.copy()
    idx = np.nonzero(inb)[0]
    x, y = xx[idx], yy[idx]
    a, b, c, d = mnx[idx], mny[idx], mxx[idx], mxy[idx]
    ll = t.get(a, b)
    lr = t.get(c, b)
    ul = t.get(a, d)
    ur = t.get(c, d)
    good = (ll != msg) & (lr != msg) & (ul != msg) & (ur != msg)
    ok[idx] = good
    rmx, rmn, rmxy, rmny = c.astype(F), a.astype(F), d.astype(F), b.astype(F)
    v = (y - rmny) * (ul * (rmx - x) + ur * (x - rmn)) + (rmxy - y) * (ll * (rmx - x) + lr * (x - rmn))
    sx = a == c
    sy = b == d
    v = np.where(sx & sy, ll, v)
    v = np.where(sx & ~sy, ll * (rmxy - y) + ul * (y - rmny), v)
    v = np.where(~sx & sy, ll * (rmx - x) + lr * (x - rmn), v)
    val[idx] = np.where(good, v, msg).astype(F)
    return val, ok


def _clamp_avg_indices(t, xx, yy):
    ifx, icx, ify, icy = _floor(xx), _ceil(xx), _floor(yy), _ceil(yy)
    out = (ifx < t.sx) | (icx > t.ex) | (ify < t.sy) | (icy > t.ey)
    c1 = out & (xx > F(t.sx) - F(0.5)) & (ifx < t.sx)
    c2 = out & ~c1 & (xx < F(t.ex) + F(0.5)) & (icx > t.ex)
    ifx = np.where(c1, t.sx, np.where(c2, t.ex, ifx))
    icx = np.where(c1, t.sx, np.where(c2, t.ex, icx))
    return ifx, icx, ify, icy, out


def four_pt_average(t, xx, yy, msg, weighted=False):
    n = xx.shape[0]
    val = np.full(n, msg, F)
    ifx, icx, ify, icy = _floor(xx), _ceil(xx), _floor(yy), _ceil(yy)
    if weighted:
        w = [np.maximum(F(0.0), F(1.0) - np.sqrt((xx - a.astype(F)) ** 2 + (yy - b.astype(F)) ** 2))
             for a, b in ((ifx, ify), (ifx, icy), (icx, ify), (icx, icy))]
    else:
        w = [np.ones(n, F) for _ in range(4)]
    out = (ifx < t.sx) | (icx > t.ex) | (ify < t.sy) | (icy > t.ey)
    c1 = out & (xx > F(t.sx) - F(0.5)) & (ifx < t.sx)
    c2 = out & ~c1 & (xx < F(t.ex) + F(0.5)) & (icx > t.ex)
    ifx = np.where(c1, t.sx, np.where(c2, t.ex, ifx))
    icx = np.where(c1, t.sx, np.where(c2, t.ex, icx))
    # (wt_ version has a typo in WPS: compares ifx with start_y; reproduced)
    cy = ifx if weighted else ify
    d1 = out & (yy > F(t.sy) - F(0.5)) & (cy < t.sy)
    d2 = out & ~d1 & (yy < F(t.ey) + F(0.5)) & (icy > t.ey)
    ify = np.where(d1, t.sy, np.where(d2, t.ey, ify))
    icy = np.where(d1, t.sy, np.where(d2, t.ey, icy))
    bad = (ifx < t.sx) | (icx > t.ex) | (ify < t.sy) | (icy > t.ey)
    idx = np.nonzero(~bad)[0]
    ok = np.zeros(n, bool)
    if idx.size:
        corners = ((ifx, ify), (ifx, icy), (icx, ify), (icx, icy))
        vals = [t.get(a[idx], b[idx]) for a, b in corners]
        ww = [np.where(v == msg, F(0.0), wi[idx]).astype(F) for v, wi in zip(vals, w)]
        anyw = (ww[0] != 0) | (ww[1] != 0) | (ww[2] != 0) | (ww[3] != 0)
        num = ww[0] * vals[0] + ww[1] * vals[1] + ww[2] * vals[2] + ww[3] * vals[3]
        den = ww[0] + ww[1] + ww[2] + ww[3]
        with np.errstate(invalid="ignore", divide="ignore"):
            v = (num / den).astype(F)
        ok[idx] = anyw
        val[idx] = np.where(anyw, v, msg)
    # points that were out of range return msgval directly (no fall-through)
    term = bad
    return val, ok, term


def sixteen_pt_average(t, xx, yy, msg, weighted=False):
    n = xx.shape[0]
    val = np.full(n, msg, F)
    ifx, ify = _floor(xx), _floor(yy)
    inb = (ifx >= t.sx + 1) & (ifx <= t.ex - 2) & (ify >= t.sy + 1) & (ify <= t.ey - 2)
    ok = np.zeros(n, bool)
    idx = np.nonzero(inb)[0]
    if idx.size:
        x, y, a, b = xx[idx], yy[idx], ifx[idx], ify[idx]
        sw = np.zeros(idx.size, F)
        s = np.zeros(idx.size, F)
        ws, vs = [], []
        for i in range(1, 5):
            for j in range(1, 5):
                px, py = a + 3 - i, b + 3 - j
                v = t.get(px, py)
                if weighted:
                    w = np.maximum(F(0.0), F(2.0) - np.sqrt((x - px.astype(F)) ** 2 + (y - py.astype(F)) ** 2))
                else:
                    w = np.ones(idx.size, F)
                w = np.where(v == msg, F(0.0), w).astype(F)
                sw = (sw + w).astype(F)
                ws.append(w)
                vs.append(v)
        for w, v in zip(ws, vs):
            s = (s + w * v).astype(F)
        good = sw != 0
        with np.errstate(invalid="ignore", divide="ignore"):
            val[idx] = np.where(good, s / sw, msg)
        ok[idx] = good
    return val, ok


def _oned(x, a, b, c, d):
    one = F(1.0)
    h = F(0.5)
    o = np.zeros_like(x)
    o = np.where(x == 0, b, o)
    o = np.where((x != 0) & (x == one), c, o)
    bc = (b * c) != 0
    ad0 = (a * d) == 0
    lin = b * (one - x) + c * x
    fa = b + x * (h * (c - a) + x * (h * (c + a) - b))
    fd = c + (one - x) * (h * (b - d) + (one - x) * (h * (b + d) - c))
    full = (one - x) * (b + x * (h * (c - a) + x * (h * (c + a) - b))) + \
        x * (c + (one - x) * (h * (b - d) + (one - x) * (h * (b + d) - c)))
    r = np.where(ad0,
                 np.where((a == 0) & (d == 0), lin, np.where(a != 0, fa, np.where(d != 0, fd, o))),
                 full)
    return np.where(bc, r, o).astype(F)


def sixteen_pt(t, xx, yy, msg):
    n = xx.shape[0]
    val = np.full(n, msg, F)
    ok = np.zeros(n, bool)
    ix0 = np.trunc(xx).astype(np.int64)
    iy0 = np.trunc(yy).astype(np.int64)
    inb = (ix0 >= t.sx) & (ix0 <= t.ex) & (iy0 >= t.sy) & (iy0 <= t.ey)
    i = np.trunc(xx + F(0.00001)).astype(np.int64)
    j = np.trunc(yy + F(0.00001)).astype(np.int64)
    x = (xx - i.astype(F)).astype(F)
    y = (yy - j.astype(F)).astype(F)
    off = (np.abs(x) > F(0.0001)) | (np.abs(y) > F(0.0001))
    # interior branch
    idx = np.nonzero(inb & off)[0]
    if idx.size:
        stl = np.empty((4, 4, idx.size), F)
        for k in range(4):
            kk = np.clip(i[idx] + k - 1, t.sx, t.ex)
            for l in range(4):
                ll = np.clip(j[idx] + l - 1, t.sy, t.ey)
                v = t.get(kk, ll)
                if msg != 0:
                    v = np.where(v == 0, TINY, v)
                stl[k, l] = v
        anym = np.any(stl.reshape(16, -1) == msg, axis=0)
        xs, ys = x[idx], y[idx]
        a = _oned(xs, stl[0, 0], stl[1, 0], stl[2, 0], stl[3, 0])
        b = _oned(xs, stl[0, 1], stl[1, 1], stl[2, 1], stl[3, 1])
        c = _oned(xs, stl[0, 2], stl[1, 2], stl[2, 2], stl[3, 2])
        d = _oned(xs, stl[0, 3], stl[1, 3], stl[2, 3], stl[3, 3])
        r = _oned(ys, a, b, c, d)
        r = np.where(r == TINY, F(0.0), r)
        val[idx] = np.where(anym, msg, r)
        ok[idx] = ~anym
    # on-node branch
    idx = np.nonzero(inb & ~off)[0]
    if idx.size:
        ii, jj = i[idx], j[idx]
        inb2 = (ii >= t.sx) & (ii <= t.ex) & (jj >= t.sy) & (jj <= t.ey)
        v = np.full(idx.size, msg, F)
        v[inb2] = t.get(ii[inb2], jj[inb2])
        good = inb2 & (v != msg)
        val[idx] = np.where(good, v, msg)
        ok[idx] = good
    return val, ok


def nearest_neighbor(t, xx, yy, msg):
    n = xx.shape[0]
    val = np.full(n, msg, F)
    from .f32 import nint
    ix, iy = nint(xx), nint(yy)
    inb = (ix >= t.sx) & (ix <= t.ex) & (iy >= t.sy) & (iy <= t.ey)
    val[inb] = t.get(ix[inb], iy[inb])
    return val, val != msg


def search(t, xx, yy, msg, depth=1200):
    """search_extrap: breadth-first search for the nearest valid source point."""
    from .f32 import nint
    n = xx.shape[0]
    val = np.full(n, msg, F)
    ok = np.zeros(n, bool)
    for p in range(n):
        x0, y0 = int(nint(xx[p])), int(nint(yy[p]))
        if x0 < t.sx or x0 > t.ex or y0 < t.sy or y0 > t.ey:
            continue
        seen = {(x0, y0)}
        q = deque([(x0, y0, 0)])
        found = None
        while q:
            i, j, dep = q.popleft()
            if t.get(i, j) != msg:
                found = (i, j)
            # WPS increments qdata%depth in place for every neighbour it queues,
            # so successive neighbours of one point get depth d+1, d+2, ...
            for (a, b) in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if t.sx <= a <= t.ex and t.sy <= b <= t.ey and (a, b) not in seen and dep < depth:
                    dep += 1
                    q.append((a, b, dep))
                    seen.add((a, b))
            if found is not None:
                break
        if found is None:
            continue
        i, j = found
        X, Y = F(xx[p]), F(yy[p])
        dist = (F(i) - X) * (F(i) - X) + (F(j) - Y) * (F(j) - Y)
        best = t.get(i, j)
        for (a, b, _) in q:
            v = t.get(a, b)
            if v != msg:
                dd = (F(a) - X) * (F(a) - X) + (F(b) - Y) * (F(b) - Y)
                if dd < dist:
                    dist, best = dd, v
        val[p] = best
        ok[p] = True
    return val, ok


def parse_chain(s):
    """'average_gcell(4.0)+four_pt+search(10)' -> (gcell_threshold|None, [(name, opt), ...])."""
    gcell = None
    chain = []
    for tok in s.split("+"):
        tok = tok.strip()
        if tok.startswith("average_gcell"):
            gcell = float(tok[tok.index("(") + 1:tok.index(")")]) if "(" in tok else 0.0
        elif tok.startswith("search"):
            depth = int(tok[tok.index("(") + 1:tok.index(")")]) if "(" in tok else 1200
            chain.append(("search", depth))
        else:
            chain.append((tok, None))
    return gcell, chain


def interp_chain(t, xx, yy, msg, chain):
    """Apply a fall-through chain on one tile/level; returns float32 values (msg if all fail)."""
    xx = np.asarray(xx, F)
    yy = np.asarray(yy, F)
    n = xx.shape[0]
    out = np.full(n, msg, F)
    pend = np.arange(n)
    for name, opt in chain:
        if pend.size == 0:
            break
        x, y = xx[pend], yy[pend]
        if name == "four_pt":
            v, ok = four_pt(t, x, y, msg)
            term = np.zeros_like(ok)
        elif name in ("average_4pt", "wt_average_4pt"):
            v, ok, term = four_pt_average(t, x, y, msg, weighted=name.startswith("wt"))
        elif name in ("average_16pt", "wt_average_16pt"):
            v, ok = sixteen_pt_average(t, x, y, msg, weighted=name.startswith("wt"))
            term = np.zeros_like(ok)
        elif name == "sixteen_pt":
            v, ok = sixteen_pt(t, x, y, msg)
            term = np.zeros_like(ok)
        elif name == "nearest_neighbor":
            v, ok = nearest_neighbor(t, x, y, msg)
            term = np.zeros_like(ok)
        elif name == "search":
            v, ok = search(t, x, y, msg, opt)
            term = np.zeros_like(ok)
        else:
            raise ValueError("unknown interpolator %s" % name)
        out[pend[ok]] = v[ok]
        pend = pend[~ok & ~term]
    return out
