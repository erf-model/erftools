"""Model grid: Lambert conformal C-grid with mass / U / V / corner points.

Index convention (as in geogrid): mass point (i, j), 1-based, i = 1..nx,
j = 1..ny, nx = e_we - 1.  The reference point (ref_lat, ref_lon) sits at
(known_x, known_y) = (e_we/2, e_sn/2) unless given.  U points are at i - 0.5
(i = 1..nx+1), V points at j - 0.5, corners at both.  All 2-D arrays are
returned numpy-style as [j, i].

Two projection back ends (option ``grid.method``):

* ``"wps_f32"``  -- Python transcription of WPS module_map_utils.F
  (set_lc / ijll_lc / llij_lc, get_map_factor, get_rotang, Coriolis) evaluated
  in IEEE single precision with glibc float functions; bit-reproduces geogrid.
* ``"pyproj"``   -- pyproj LCC on a sphere in double precision; map factors from
  pyproj ``get_factors``; rotation from the cone convergence n*(lon-lon0).

Arrays for interpolation are produced on a "memory" region with a halo of
``halo`` points around the domain (geogrid uses HALO_WIDTH = 3; its smoothers
read those halo points).
"""
from dataclasses import dataclass, field

import numpy as np

from . import f32 as m

F = np.float32


@dataclass
class GridSpec:
    e_we: int = 202
    e_sn: int = 202
    dx: float = 500.0
    dy: float = 500.0
    map_proj: str = "lambert"
    ref_lat: float = 38.9562
    ref_lon: float = -106.9880
    truelat1: float = 38.5
    truelat2: float = 38.5
    stand_lon: float = -97.5
    ref_x: float = None
    ref_y: float = None
    earth_radius: float = 6370000.0
    halo: int = 3

    @property
    def nx(self):
        return self.e_we - 1

    @property
    def ny(self):
        return self.e_sn - 1

    @property
    def known_x(self):
        return self.e_we / 2.0 if self.ref_x is None else self.ref_x

    @property
    def known_y(self):
        return self.e_sn / 2.0 if self.ref_y is None else self.ref_y

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# --------------------------------------------------------------------------
# WPS single-precision Lambert conformal (module_map_utils.F)
# --------------------------------------------------------------------------
class WpsLambertF32:
    def __init__(self, g: GridSpec):
        if g.map_proj != "lambert":
            raise NotImplementedError("only lambert is implemented")
        self.truelat1 = F(g.truelat1)
        self.truelat2 = F(g.truelat2)
        self.stdlon = F(g.stand_lon)
        self.lat1 = F(g.ref_lat)
        self.lon1 = F(g.ref_lon)
        self.knowni = F(g.known_x)
        self.knownj = F(g.known_y)
        self.dx = F(g.dx)
        self.re_m = F(g.earth_radius)
        self.hemi = F(-1.0) if self.truelat1 < 0 else F(1.0)
        self.rebydx = self.re_m / self.dx
        R = m.RAD_PER_DEG
        # lc_cone
        if abs(self.truelat1 - self.truelat2) > F(0.1):
            c = m.log10(m.cos(self.truelat1 * R)) - m.log10(m.cos(self.truelat2 * R))
            c = c / (m.log10(m.tan((F(45.0) - abs(self.truelat1) / F(2.0)) * R))
                     - m.log10(m.tan((F(45.0) - abs(self.truelat2) / F(2.0)) * R)))
            self.cone = F(c)
        else:
            self.cone = m.sin(abs(self.truelat1) * R)
        # set_lc
        deltalon1 = self.lon1 - self.stdlon
        if deltalon1 > F(180.0):
            deltalon1 = deltalon1 - F(360.0)
        if deltalon1 < F(-180.0):
            deltalon1 = deltalon1 + F(360.0)
        tl1r = self.truelat1 * R
        ctl1r = m.cos(tl1r)
        ratio = (m.tan((F(90.0) * self.hemi - self.lat1) * R / F(2.0))
                 / m.tan((F(90.0) * self.hemi - self.truelat1) * R / F(2.0)))
        self.rsw = self.rebydx * ctl1r / self.cone * m.power(ratio, self.cone)
        arg = self.cone * (deltalon1 * R)
        self.polei = self.hemi * self.knowni - self.hemi * self.rsw * m.sin(arg)
        self.polej = self.hemi * self.knownj + self.rsw * m.cos(arg)

    def ij_to_ll(self, i, j):
        """ijll_lc: float32 grid coordinates -> (lat, lon) float32."""
        i = np.asarray(i, F)
        j = np.asarray(j, F)
        R = m.RAD_PER_DEG
        chi1 = (F(90.0) - self.hemi * self.truelat1) * R
        chi2 = (F(90.0) - self.hemi * self.truelat2) * R
        inew = self.hemi * i
        jnew = self.hemi * j
        xx = inew - self.polei
        yy = self.polej - jnew
        r2 = xx * xx + yy * yy
        r = np.sqrt(r2) / self.rebydx
        lon = self.stdlon + m.DEG_PER_RAD * m.atan2(self.hemi * xx, yy) / self.cone
        lon = np.fmod(lon + F(360.0), F(360.0)).astype(F)
        if chi1 == chi2:
            chi = F(2.0) * m.atan(m.power(r / m.tan(chi1), F(1.0) / self.cone) * m.tan(chi1 * F(0.5)))
        else:
            chi = F(2.0) * m.atan(m.power(r * self.cone / m.sin(chi1), F(1.0) / self.cone)
                                  * m.tan(chi1 * F(0.5)))
        lat = (F(90.0) - chi * m.DEG_PER_RAD) * self.hemi
        lon = np.where(lon > F(180.0), lon - F(360.0), lon).astype(F)
        lon = np.where(lon < F(-180.0), lon + F(360.0), lon).astype(F)
        lat = np.where(r2 == 0, self.hemi * F(90.0), lat).astype(F)
        lon = np.where(r2 == 0, self.stdlon, lon).astype(F)
        return lat, lon

    def ll_to_ij(self, lat, lon):
        """llij_lc: (lat, lon) float32 -> float32 grid coordinates (mass)."""
        lat = np.asarray(lat, F)
        lon = np.asarray(lon, F)
        R = m.RAD_PER_DEG
        deltalon = lon - self.stdlon
        deltalon = np.where(deltalon > F(180.0), deltalon - F(360.0), deltalon).astype(F)
        deltalon = np.where(deltalon < F(-180.0), deltalon + F(360.0), deltalon).astype(F)
        tl1r = self.truelat1 * R
        ctl1r = m.cos(tl1r)
        ratio = (m.tan((F(90.0) * self.hemi - lat) * R / F(2.0))
                 / m.tan((F(90.0) * self.hemi - self.truelat1) * R / F(2.0)))
        rm = self.rebydx * ctl1r / self.cone * m.power(ratio, self.cone)
        arg = self.cone * (deltalon * R)
        i = self.polei + self.hemi * rm * m.sin(arg)
        j = self.polej - rm * m.cos(arg)
        return (self.hemi * i).astype(F), (self.hemi * j).astype(F)

    def mapfac(self, lat):
        """get_map_factor (process_tile_module.F), Lambert branch."""
        lat = np.asarray(lat, F)
        R = m.RAD_PER_DEG
        if self.truelat1 != self.truelat2:
            colat1 = R * (F(90.0) - self.truelat1)
            colat2 = R * (F(90.0) - self.truelat2)
            n = ((np.log(m.sin(colat1)) - np.log(m.sin(colat2)))
                 / (np.log(m.tan(colat1 / F(2.0))) - np.log(m.tan(colat2 / F(2.0)))))
            colat = R * (F(90.0) - lat)
            return (m.sin(colat2) / m.sin(colat)
                    * m.power(m.tan(colat / F(2.0)) / m.tan(colat2 / F(2.0)), F(n))).astype(F)
        colat0 = R * (F(90.0) - self.truelat1)
        colat = R * (F(90.0) - lat)
        return (m.sin(colat0) / m.sin(colat)
                * m.power(m.tan(colat / F(2.0)) / m.tan(colat0 / F(2.0)), m.cos(colat0))).astype(F)

    @staticmethod
    def coriolis(lat):
        lat = np.asarray(lat, F)
        f = F(2.0) * m.OMEGA_E * m.sin(m.RAD_PER_DEG * lat)
        e = F(2.0) * m.OMEGA_E * m.cos(m.RAD_PER_DEG * lat)
        return f.astype(F), e.astype(F)

    @staticmethod
    def rotang(xlat, xlon):
        """get_rotang on a full memory array [j, i] (centred in j, one-sided at the ends)."""
        xlat = np.asarray(xlat, F)
        xlon = np.asarray(xlon, F)
        R = m.RAD_PER_DEG
        sina = np.empty_like(xlat)
        cosa = np.empty_like(xlat)

        def _a(lat_c, lon_p, lon_m, lat_p, lat_m):
            d_lon = (lon_p - lon_m).astype(F)
            d_lon = np.where(d_lon > F(180.0), d_lon - F(360.0), d_lon).astype(F)
            d_lon = np.where(d_lon < F(-180.0), d_lon + F(360.0), d_lon).astype(F)
            return m.atan2(-m.cos(lat_c * R) * (d_lon * R), (lat_p - lat_m) * R)

        a = _a(xlat[1:-1], xlon[2:], xlon[:-2], xlat[2:], xlat[:-2])
        sina[1:-1] = m.sin(a)
        cosa[1:-1] = m.cos(a)
        a = _a(xlat[0], xlon[1], xlon[0], xlat[1], xlat[0])
        sina[0] = m.sin(a)
        cosa[0] = m.cos(a)
        a = _a(xlat[-1], xlon[-1], xlon[-2], xlat[-1], xlat[-2])
        sina[-1] = m.sin(a)
        cosa[-1] = m.cos(a)
        return sina, cosa


# --------------------------------------------------------------------------
# Independent double-precision Lambert conformal (pyproj)
# --------------------------------------------------------------------------
class PyprojLambert:
    def __init__(self, g: GridSpec):
        import pyproj
        self.g = g
        self.crs = pyproj.CRS.from_proj4(
            f"+proj=lcc +lat_1={g.truelat1} +lat_2={g.truelat2} +lat_0={g.truelat1} "
            f"+lon_0={g.stand_lon} +x_0=0 +y_0=0 +R={g.earth_radius} +units=m +no_defs")
        self.proj = pyproj.Proj(self.crs)
        self.x0, self.y0 = self.proj(g.ref_lon, g.ref_lat)
        self.n = (np.sin(np.radians(g.truelat1)) if abs(g.truelat1 - g.truelat2) < 1e-9 else
                  (np.log(np.cos(np.radians(g.truelat1))) - np.log(np.cos(np.radians(g.truelat2))))
                  / (np.log(np.tan(np.radians(45 + g.truelat2 / 2)))
                     - np.log(np.tan(np.radians(45 + g.truelat1 / 2)))))

    def ij_to_xy(self, i, j):
        return (self.x0 + (np.asarray(i, float) - self.g.known_x) * self.g.dx,
                self.y0 + (np.asarray(j, float) - self.g.known_y) * self.g.dy)

    def xy_to_ij(self, x, y):
        return ((np.asarray(x, float) - self.x0) / self.g.dx + self.g.known_x,
                (np.asarray(y, float) - self.y0) / self.g.dy + self.g.known_y)

    def ij_to_ll(self, i, j):
        x, y = self.ij_to_xy(i, j)
        lon, lat = self.proj(x, y, inverse=True)
        return np.asarray(lat), np.asarray(lon)

    def ll_to_ij(self, lat, lon):
        x, y = self.proj(np.asarray(lon, float), np.asarray(lat, float))
        return self.xy_to_ij(x, y)

    def mapfac(self, lat, lon):
        fac = self.proj.get_factors(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(fac.parallel_scale)

    @staticmethod
    def coriolis(lat):
        om = 7.292e-5
        return 2 * om * np.sin(np.radians(lat)), 2 * om * np.cos(np.radians(lat))

    def rotang(self, lat, lon):
        # Grid +y is rotated from local north by gamma = n (lon - lon0) (cone
        # convergence); WRF's alpha = -gamma (SINALPHA < 0 west of stand_lon).
        dlon = (np.asarray(lon, float) - self.g.stand_lon + 180.0) % 360.0 - 180.0
        a = -self.n * np.radians(dlon)
        return np.sin(a), np.cos(a)


# --------------------------------------------------------------------------
@dataclass
class ModelGrid:
    spec: GridSpec
    method: str = "wps_f32"
    fields: dict = field(default_factory=dict)

    def __post_init__(self):
        g = self.spec
        h = g.halo
        self.proj = WpsLambertF32(g) if self.method == "wps_f32" else PyprojLambert(g)
        # memory index ranges (1-based Fortran indices)
        self.i_m = np.arange(1 - h, g.nx + h + 1)            # mass i
        self.j_m = np.arange(1 - h, g.ny + h + 1)
        self.i_s = np.arange(1 - h, g.nx + 1 + h + 1)        # staggered i (U)
        self.j_s = np.arange(1 - h, g.ny + 1 + h + 1)
        self._compute()

    # slices that cut the domain (patch) out of memory arrays
    @property
    def sl_m(self):
        h = self.spec.halo
        return (slice(h, h + self.spec.ny), slice(h, h + self.spec.nx))

    def _ll(self, ii, jj, shift_x=0.0, shift_y=0.0):
        I, J = np.meshgrid(ii, jj)
        if self.method == "wps_f32":
            # xytoll: rx = x - 0.5 for U / corner, ry = y - 0.5 for V / corner
            x = I.astype(F) - F(shift_x)
            y = J.astype(F) - F(shift_y)
            return self.proj.ij_to_ll(x, y)
        return self.proj.ij_to_ll(I - shift_x, J - shift_y)

    def _compute(self):
        p = self.proj
        f = self.fields
        f["XLAT_M"], f["XLONG_M"] = self._ll(self.i_m, self.j_m)
        f["XLAT_U"], f["XLONG_U"] = self._ll(self.i_s, self.j_m, 0.5, 0.0)
        f["XLAT_V"], f["XLONG_V"] = self._ll(self.i_m, self.j_s, 0.0, 0.5)
        f["XLAT_C"], f["XLONG_C"] = self._ll(self.i_s, self.j_s, 0.5, 0.5)
        f["CLAT"], f["CLONG"] = f["XLAT_M"].copy(), f["XLONG_M"].copy()
        for s in ("M", "U", "V"):
            lat, lon = f["XLAT_" + s], f["XLONG_" + s]
            mf = p.mapfac(lat) if self.method == "wps_f32" else p.mapfac(lat, lon)
            f["MAPFAC_" + s] = mf
            f["MAPFAC_%sX" % s] = mf
            f["MAPFAC_%sY" % s] = mf
        f["F"], f["E"] = p.coriolis(f["XLAT_M"])
        for s, suf in (("M", ""), ("U", "_U"), ("V", "_V")):
            if self.method == "wps_f32":
                sa, ca = p.rotang(f["XLAT_" + s], f["XLONG_" + s])
            else:
                sa, ca = p.rotang(f["XLAT_" + s], f["XLONG_" + s])
            f["SINALPHA" + suf], f["COSALPHA" + suf] = sa, ca

    def patch(self, name):
        """Cut a memory array to the domain (mass, U, V or corner shaped)."""
        a = self.fields[name] if isinstance(name, str) else name
        g, h = self.spec, self.spec.halo
        ny = g.ny + (1 if a.shape[-2] == len(self.j_s) else 0)
        nx = g.nx + (1 if a.shape[-1] == len(self.i_s) else 0)
        return a[..., h:h + ny, h:h + nx]

    def latlon_to_ij(self, lat, lon):
        """Fractional 1-based mass-grid indices of arbitrary points (float64)."""
        if self.method == "wps_f32":
            i, j = self.proj.ll_to_ij(np.float32(lat), np.float32(lon))
            return np.float64(i), np.float64(j)
        return self.proj.ll_to_ij(lat, lon)

    def corner_attrs(self):
        """16-element corner_lats/corner_lons global attributes (process_tile)."""
        f = self.fields
        P = self.patch
        lm, om = P("XLAT_M"), P("XLONG_M")
        lu, ou = P("XLAT_U"), P("XLONG_U")
        lv, ov = P("XLAT_V"), P("XLONG_V")
        g = self.spec
        lats, lons = [], []
        for a, b in ((lm, om), (lu, ou), (lv, ov)):
            for (jj, ii) in ((0, 0), (-1, 0), (-1, -1), (0, -1)):
                lats.append(a[jj, ii])
                lons.append(b[jj, ii])
        pts = [(0.5, 0.5), (0.5, g.ny + 0.5), (g.nx + 0.5, g.ny + 0.5), (g.nx + 0.5, 0.5)]
        for (x, y) in pts:
            if self.method == "wps_f32":
                la, lo = self.proj.ij_to_ll(F(x), F(y))
            else:
                la, lo = self.proj.ij_to_ll(x, y)
            lats.append(la)
            lons.append(lo)
        return np.array(lats, F), np.array(lons, F)
