"""Readers for static source datasets.

``GeogridTiles`` reads the WPS "geogrid binary" tile format directly from the
bytes, driven only by the dataset's ``index`` file (projection regular_ll,
known_x/known_y/known_lat/known_lon, dx/dy, tile_x/tile_y/tile_z or
tile_z_start/tile_z_end, tile_bdr, wordsize, signed, endian, row_order,
scale_factor, missing_value, units, description, type, category_min/max,
filename_digits).  Tiles are named ``xxxxx-xxxxx.yyyyy-yyyyy`` with 1-based
global source indices (5 or 6 digits).

A source object exposes a small interface used by the rest of the package:

  * ``meta``                       parsed index (dict)
  * ``ll_to_xy(lat, lon)``         fractional 1-based global indices
  * ``xy_to_ll(x, y)``             inverse
  * ``tile_of(x, y)``              tile origin (global start index) for a point
  * ``read_tile(tx0, ty0)``        (array[z, y, x] float32 raw values incl. border,
                                    global index of array[.,0,0])
  * ``window(x0, x1, y0, y1)``     mosaic of raw values over a global index window
                                    (used by the non-WPS methods)

New sources (e.g. a USGS 3DEP GeoTIFF) only need to implement the same
interface; see ``README.md``.
"""
import hashlib
import os
import re

import numpy as np

from . import f32 as m

F = np.float32


def parse_index(path):
    meta = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            meta[k.strip().lower()] = v.strip().strip('"')
    return meta


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class GeogridTiles:
    def __init__(self, path):
        self.path = path.rstrip("/") + "/"
        self.index_path = os.path.join(self.path, "index")
        self.meta = md = parse_index(self.index_path)
        self.index_sha256 = sha256(self.index_path)
        if md.get("projection", "regular_ll") != "regular_ll":
            raise NotImplementedError("only regular_ll sources are implemented: %s" % path)
        # Fortran list-read values are REAL*4
        self.dx = m.f32(md["dx"])
        self.dy = m.f32(md["dy"])
        self.known_x = m.f32(md.get("known_x", 1.0))
        self.known_y = m.f32(md.get("known_y", 1.0))
        self.known_lat = m.f32(md["known_lat"])
        lon1 = float(md["known_lon"])
        while abs(lon1) > 180.0:
            lon1 += 360.0 if lon1 < -180.0 else -360.0
        self.known_lon = m.f32(lon1)
        self.nxmax = int(m.nint(F(360.0) / self.dx))
        self.tile_x = int(md["tile_x"])
        self.tile_y = int(md["tile_y"])
        if "tile_z_start" in md and "tile_z_end" in md:
            self.z0, self.z1 = int(md["tile_z_start"]), int(md["tile_z_end"])
        else:
            self.z0, self.z1 = 1, int(md.get("tile_z", 1))
        self.nz = self.z1 - self.z0 + 1
        self.bdr = int(md.get("tile_bdr", 0))
        self.wordsize = int(md["wordsize"])
        self.signed = md.get("signed", "no").lower() == "yes"
        self.big_endian = md.get("endian", "big").lower() != "little"
        self.top_bottom = md.get("row_order", "bottom_top").lower() == "top_bottom"
        self.scale_factor = m.f32(md["scale_factor"]) if "scale_factor" in md else None
        self.missing_value = m.f32(md["missing_value"]) if "missing_value" in md else F(1.0e20)
        self.type = md.get("type", "continuous").lower()
        self.cat_min = int(md["category_min"]) if "category_min" in md else None
        self.cat_max = int(md["category_max"]) if "category_max" in md else None
        self.units = md.get("units", "")
        self.description = md.get("description", "")
        self.digits = int(md.get("filename_digits", 5))
        self._cache = {}

    # ------------------------------------------------------------------ geometry
    def ll_to_xy(self, lat, lon, f32=True):
        """llij_latlon (module_map_utils.F); geogrid passes lon-360 if lon>=180."""
        if f32:
            lat = np.asarray(lat, F)
            lon = np.asarray(lon, F)
            lon = np.where(lon >= F(180.0), lon - F(360.0), lon).astype(F)
            i = (lon - self.known_lon) / self.dx
            j = (lat - self.known_lat) / self.dy
            i = i + self.known_x
            j = j + self.known_y
            per = F(self.nxmax)
            i = np.where(i < F(0.5), i + per, i).astype(F)
            i = np.where(i >= F(self.nxmax) + F(0.5), i - per, i).astype(F)
            return i, j
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        dx, dy, lat1, lon1 = self.exact_georef()
        i = (lon - lon1) / dx + float(self.known_x)
        n = float(self.nxmax)
        i = np.where(i < 0.5, i + n, i)
        i = np.where(i >= n + 0.5, i - n, i)
        j = (lat - lat1) / dy + float(self.known_y)
        return i, j

    def exact_georef(self):
        """Double-precision georeferencing with the index's rounded decimals
        snapped to the exact arc-second grid they denote (e.g. dx=0.00833333 ->
        1/120 deg, known_lon=-179.99583 -> -180 + 1/240).  Snapping is applied
        only when the change is < 1e-4 of a cell; otherwise the index values
        are used as written."""
        dx, dy = float(self.meta["dx"]), float(self.meta["dy"])
        lat1, lon1 = float(self.meta["known_lat"]), float(self.meta["known_lon"])

        def snap_res(d):
            r = round(1.0 / abs(d))
            if r > 0 and abs(abs(d) - 1.0 / r) < 1e-4 * abs(d):
                return np.sign(d) / r
            return d

        def snap_pos(p, d):
            h = abs(d) / 2.0
            q = round(p / h) * h
            return q if abs(q - p) < 1e-4 * abs(d) else p
        dx, dy = snap_res(dx), snap_res(dy)
        return dx, dy, snap_pos(lat1, dy), snap_pos(lon1, dx)

    def xy_to_ll(self, x, y, f32=True):
        """ijll_latlon."""
        if f32:
            x = np.asarray(x, F)
            y = np.asarray(y, F)
            per = F(self.nxmax)
            iw = np.where(x < F(0.5), x + per, x).astype(F)
            iw = np.where(x >= F(self.nxmax) + F(0.5), x - per, iw).astype(F)
            iw = iw - self.known_x
            jw = y - self.known_y
            lat = self.known_lat + jw * self.dy
            lon = self.known_lon + iw * self.dx
            return lat.astype(F), lon.astype(F)
        dx, dy, lat1, lon1 = self.exact_georef()
        lat = lat1 + (np.asarray(y, float) - float(self.known_y)) * dy
        lon = lon1 + (np.asarray(x, float) - float(self.known_x)) * dx
        lon = np.mod(lon + 180.0, 360.0) - 180.0
        return lat, lon

    def tile_of(self, x, y):
        """Global start indices of the tile containing (x, y) (get_tile_dimensions)."""
        x = np.asarray(x, F)
        y = np.asarray(y, F)
        tx = self.tile_x * np.floor((x - F(0.5)) / F(self.tile_x)).astype(np.int64) + 1
        ty = self.tile_y * np.floor((y - F(0.5)) / F(self.tile_y)).astype(np.int64) + 1
        return tx, ty

    # ------------------------------------------------------------------ I/O
    def tile_name(self, tx0, ty0):
        d = self.digits
        return os.path.join(self.path, "%0*d-%0*d.%0*d-%0*d" % (
            d, tx0, d, tx0 + self.tile_x - 1, d, ty0, d, ty0 + self.tile_y - 1))

    def read_tile(self, tx0, ty0, xr=None, yr=None):
        """Raw tile incl. border as float32 [z, y, x]; returns (arr, x_start, y_start)
        where x_start/y_start are the global indices of arr[:, 0, 0].

        ``xr``/``yr`` = (first, last) global indices restrict decoding to a
        sub-window (memory-mapped; used for very large tiles such as NLCD 9s).
        Decoding follows read_geogrid.c: big-endian unless endian=little; signed
        words become negative only when strictly greater than 2^(8*ws-1).
        Missing tile -> None.
        """
        key = (tx0, ty0, xr, yr)
        if key in self._cache:
            return self._cache[key]
        fn = self.tile_name(tx0, ty0)
        if not os.path.exists(fn):
            self._cache[key] = None
            return None
        b = self.bdr
        nx, ny, nz = self.tile_x + 2 * b, self.tile_y + 2 * b, self.nz
        ws = self.wordsize
        mm = np.memmap(fn, dtype=np.uint8, mode="r", shape=(nz, ny, nx * ws))
        gx0, gy0 = tx0 - b, ty0 - b
        lx0, lx1 = (0, nx - 1) if xr is None else (max(xr[0] - gx0, 0), min(xr[1] - gx0, nx - 1))
        ly0, ly1 = (0, ny - 1) if yr is None else (max(yr[0] - gy0, 0), min(yr[1] - gy0, ny - 1))
        if self.top_bottom:      # file row r holds global row (ny-1-r)
            fr0, fr1 = ny - 1 - ly1, ny - 1 - ly0
        else:
            fr0, fr1 = ly0, ly1
        raw = np.asarray(mm[:, fr0:fr1 + 1, lx0 * ws:(lx1 + 1) * ws])
        raw = raw.reshape(nz, fr1 - fr0 + 1, lx1 - lx0 + 1, ws).astype(np.int64)
        if not self.big_endian:
            raw = raw[..., ::-1]
        ival = np.zeros(raw.shape[:-1], np.int64)
        for k in range(ws):
            ival = (ival << 8) | raw[..., k]
        if self.signed:
            half = 1 << (8 * ws - 1)
            ival = np.where(ival > half, ival - (1 << (8 * ws)), ival)
        arr = ival.astype(F)
        if self.top_bottom:
            arr = arr[:, ::-1, :].copy()
        del mm
        out = (arr, gx0 + lx0, gy0 + ly0)
        if xr is None and yr is None:
            self._cache[key] = out
        return out

    def window(self, x0, x1, y0, y1):
        """Mosaic raw values for global indices x0..x1, y0..y1 (inclusive), [z, y, x].

        Missing tiles / areas are filled with the dataset missing value (or NaN if
        none is declared).  x wraps periodically.
        """
        fill = self.missing_value if "missing_value" in self.meta else F(np.nan)
        out = np.full((self.nz, y1 - y0 + 1, x1 - x0 + 1), fill, F)
        ys = np.arange(y0, y1 + 1)
        xs = np.arange(x0, x1 + 1)
        xw = ((xs - 1) % self.nxmax) + 1
        txs = self.tile_x * ((xw - 1) // self.tile_x) + 1
        tys = self.tile_y * ((ys - 1) // self.tile_y) + 1
        for ty0 in np.unique(tys):
            yy = np.nonzero(tys == ty0)[0]
            for tx0 in np.unique(txs):
                xx = np.nonzero(txs == tx0)[0]
                t = self.read_tile(int(tx0), int(ty0), xr=(int(xw[xx].min()), int(xw[xx].max())),
                                   yr=(int(ys[yy].min()), int(ys[yy].max())))
                if t is None:
                    continue
                arr, ax0, ay0 = t
                out[:, yy[0]:yy[-1] + 1, :][:, :, xx] = arr[:, ys[yy] - ay0][:, :, xw[xx] - ax0]
        return out

    def provenance(self):
        return {"path": self.path, "index_sha256": self.index_sha256}


_REGISTRY = {}


def open_source(path):
    """Factory: pick a reader for a dataset directory (extend for new formats)."""
    if path not in _REGISTRY:
        if os.path.exists(os.path.join(path, "index")):
            _REGISTRY[path] = GeogridTiles(path)
        elif any(f.lower().endswith((".tif", ".tiff")) for f in os.listdir(path)):
            from .geotiff import GeoTiffMosaic
            _REGISTRY[path] = GeoTiffMosaic(path)
        else:
            raise ValueError("no reader for source %s" % path)
    return _REGISTRY[path]
