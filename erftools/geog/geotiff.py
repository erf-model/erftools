"""Reader for directories of geographic (lat/lon) GeoTIFF tiles, e.g. USGS 3DEP.

``GeoTiffMosaic`` exposes the source interface used by the non-WPS methods in
``erftools.geog.methods`` (see ``dataset.py``): ``meta``, ``type``,
``cat_min/cat_max``, ``z0/z1/nz``, ``missing_value``, ``scale_factor``,
``units``, ``description``, ``index_sha256``, ``ll_to_xy``, ``xy_to_ll`` and
``window``.  The WPS method (``tile_of``/``read_tile``) is not supported.

All tiles in the directory must share one regular lat/lon grid (same pixel
size, pixel centres on one lattice).  The global index follows the geogrid
convention: 1-based, x increasing east, y increasing north, with x = 1 at the
first pixel centre east of 180 W and y = 1 at the first pixel centre north of
90 S.  Overlapping tile edges (3DEP tiles carry a 6-pixel overlap) must agree;
``check_overlaps()`` reports the largest disagreement.

Decoding uses Pillow (mode "F", LZW/deflate, floating-point predictor) and
falls back to ``tifffile`` when it is installed.  Only single-band float32
tiles in a geographic CRS are accepted.
"""
import glob
import hashlib
import os

import numpy as np

from .dataset import sha256

TAG_PIXEL_SCALE = 33550
TAG_TIEPOINT = 33922
TAG_GEOKEYS = 34735
TAG_NODATA = 42113
GK_MODEL_TYPE = 1024      # 2 = geographic
GK_RASTER_TYPE = 1025     # 1 = PixelIsArea, 2 = PixelIsPoint
GK_GEOG_CRS = 2048


def _geokeys(entries):
    """GeoKeyDirectory -> {key: value} for keys stored inline (location 0)."""
    e = list(entries)
    out = {}
    for k in range(4, 4 + 4 * e[3], 4):
        key, loc, _cnt, val = e[k:k + 4]
        if loc == 0:
            out[key] = val
    return out


def _read_tiff(path):
    """(array float32 [rows, cols], tags dict {tag: value})."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as im:
            if im.mode != "F":
                raise ValueError("%s: expected a float32 single-band TIFF, got mode %s" % (path, im.mode))
            tags = {k: im.tag_v2[k] for k in (TAG_PIXEL_SCALE, TAG_TIEPOINT, TAG_GEOKEYS, TAG_NODATA)
                    if k in im.tag_v2}
            return np.asarray(im, dtype=np.float32), tags
    except ImportError:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            tags = {t.code: t.value for t in page.tags.values()
                    if t.code in (TAG_PIXEL_SCALE, TAG_TIEPOINT, TAG_GEOKEYS, TAG_NODATA)}
            return page.asarray().astype(np.float32), tags


class GeoTiffMosaic:
    def __init__(self, path, pattern="*.tif", units="meters", description=None):
        self.path = path.rstrip("/") + "/"
        self.files = sorted(glob.glob(os.path.join(self.path, pattern)))
        if not self.files:
            raise ValueError("no %s files in %s" % (pattern, path))
        self.file_sha256 = {os.path.basename(f): sha256(f) for f in self.files}
        h = hashlib.sha256()
        for k in sorted(self.file_sha256):
            h.update(("%s  %s\n" % (self.file_sha256[k], k)).encode())
        self.index_sha256 = h.hexdigest()
        self.type = "continuous"
        self.cat_min = self.cat_max = None
        self.z0 = self.z1 = self.nz = 1
        self.scale_factor = None
        self.units = units
        self.description = description or "GeoTIFF mosaic %s" % self.path
        self._tiles = {}          # file -> (x_start, y_start, ncols, nrows), y_start = global y of top row
        self._cache = {}
        nodata = set()
        for f in self.files:
            nodata.add(self._register(f))
        if len(nodata) != 1:
            raise ValueError("tiles disagree on nodata: %s" % sorted(nodata))
        self.missing_value = np.float32(nodata.pop())
        self.meta = {"projection": "regular_ll", "dx": self.dx, "dy": self.dy,
                     "known_x": 1.0, "known_y": 1.0, "known_lat": self.lat1, "known_lon": self.lon1,
                     "missing_value": float(self.missing_value), "units": units,
                     "type": "continuous", "files": list(self.file_sha256)}

    # ------------------------------------------------------------------ setup
    def _register(self, f):
        """Read only the header of ``f`` and place it on the global lattice."""
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(f) as im:
                ncols, nrows = im.size
                tags = {k: im.tag_v2[k] for k in (TAG_PIXEL_SCALE, TAG_TIEPOINT, TAG_GEOKEYS, TAG_NODATA)
                        if k in im.tag_v2}
        except ImportError:
            arr, tags = _read_tiff(f)
            nrows, ncols = arr.shape
        gk = _geokeys(tags[TAG_GEOKEYS])
        if gk.get(GK_MODEL_TYPE) != 2:
            raise NotImplementedError("%s: only geographic (lat/lon) GeoTIFFs are supported" % f)
        sx, sy = float(tags[TAG_PIXEL_SCALE][0]), float(tags[TAG_PIXEL_SCALE][1])
        r = round(1.0 / sx)
        if abs(sx - 1.0 / r) > 1e-6 * sx or abs(sy - sx) > 1e-6 * sx:
            raise ValueError("%s: pixel scale %r is not a square 1/N degree grid" % (f, (sx, sy)))
        d = 1.0 / r
        if not hasattr(self, "dx"):
            self.dx = self.dy = d
            self.lon1 = -180.0 + d / 2.0
            self.lat1 = -90.0 + d / 2.0
            self.nxmax = int(round(360.0 / d))
            self.crs_epsg = gk.get(GK_GEOG_CRS)
        elif d != self.dx:
            raise ValueError("%s: pixel size %g differs from %g" % (f, d, self.dx))
        tp = tags[TAG_TIEPOINT]
        i0, j0, lon0, lat0 = float(tp[0]), float(tp[1]), float(tp[3]), float(tp[4])
        if gk.get(GK_RASTER_TYPE, 1) == 2:          # PixelIsPoint: tiepoint is a pixel centre
            i0, j0 = i0 + 0.5, j0 + 0.5
        # centre of pixel (col 0, row 0)
        lonc = lon0 + (0.5 - i0) * d
        latc = lat0 - (0.5 - j0) * d
        xs = (lonc - self.lon1) / d + 1.0
        ys = (latc - self.lat1) / d + 1.0
        if abs(xs - round(xs)) > 1e-3 or abs(ys - round(ys)) > 1e-3:
            raise ValueError("%s: pixel centres are off the common lattice (%g, %g)" % (f, xs, ys))
        self._tiles[f] = (int(round(xs)), int(round(ys)), ncols, nrows)
        return float(tags.get(TAG_NODATA, "nan"))

    # ------------------------------------------------------------------ geometry
    def ll_to_xy(self, lat, lon, f32=True):
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        lon = np.mod(lon + 180.0, 360.0) - 180.0
        i = (lon - self.lon1) / self.dx + 1.0
        j = (lat - self.lat1) / self.dy + 1.0
        if f32:
            return i.astype(np.float32), j.astype(np.float32)
        return i, j

    def xy_to_ll(self, x, y, f32=True):
        lat = self.lat1 + (np.asarray(y, float) - 1.0) * self.dy
        lon = self.lon1 + (np.asarray(x, float) - 1.0) * self.dx
        lon = np.mod(lon + 180.0, 360.0) - 180.0
        if f32:
            return lat.astype(np.float32), lon.astype(np.float32)
        return lat, lon

    def tile_of(self, x, y):
        raise NotImplementedError("GeoTIFF sources support the non-WPS methods only")

    def read_tile(self, *a, **k):
        raise NotImplementedError("GeoTIFF sources support the non-WPS methods only")

    # ------------------------------------------------------------------ I/O
    def _array(self, f):
        if f not in self._cache:
            self._cache[f] = _read_tiff(f)[0]
        return self._cache[f]

    def _extent(self, f):
        xs, ytop, nc, nr = self._tiles[f]
        return xs, xs + nc - 1, ytop - nr + 1, ytop

    def window(self, x0, x1, y0, y1):
        """Raw values for global indices x0..x1, y0..y1 (inclusive) as [1, y, x];
        areas no tile covers hold NaN."""
        out = np.full((1, y1 - y0 + 1, x1 - x0 + 1), np.nan, np.float32)
        for f in self.files:
            fx0, fx1, fy0, fy1 = self._extent(f)
            ax0, ax1, ay0, ay1 = max(x0, fx0), min(x1, fx1), max(y0, fy0), min(y1, fy1)
            if ax0 > ax1 or ay0 > ay1:
                continue
            a = self._array(f)
            # file row r holds global y = fy1 - r
            sub = a[fy1 - ay1:fy1 - ay0 + 1, ax0 - fx0:ax1 - fx0 + 1][::-1]
            dst = out[0, ay0 - y0:ay1 - y0 + 1, ax0 - x0:ax1 - x0 + 1]
            take = np.isnan(dst) | (dst == self.missing_value)
            dst[take] = sub[take]
        return out

    def check_overlaps(self):
        """Largest |difference| between tiles where they overlap (valid pixels only)."""
        worst = {}
        fs = self.files
        for a in range(len(fs)):
            for b in range(a + 1, len(fs)):
                ea, eb = self._extent(fs[a]), self._extent(fs[b])
                x0, x1 = max(ea[0], eb[0]), min(ea[1], eb[1])
                y0, y1 = max(ea[2], eb[2]), min(ea[3], eb[3])
                if x0 > x1 or y0 > y1:
                    continue
                va, vb = [self._array(f)[e[3] - y1:e[3] - y0 + 1, x0 - e[0]:x1 - e[0] + 1]
                          for f, e in ((fs[a], ea), (fs[b], eb))]
                ok = (va != self.missing_value) & (vb != self.missing_value) & ~np.isnan(va) & ~np.isnan(vb)
                d = float(np.abs(va - vb)[ok].max()) if ok.any() else float("nan")
                worst["%s|%s" % (os.path.basename(fs[a]), os.path.basename(fs[b]))] = (d, int(ok.sum()))
        return worst

    def provenance(self):
        return {"path": self.path, "index_sha256": self.index_sha256, "files": self.file_sha256}
