"""Opt-in terrain replacement for HRRR -> WRF/ERF inputs.

HRRR's atmosphere sits on HRRR's own (3 km, smooth) terrain. This module rebuilds every
output column over a different terrain (e.g. WPS geogrid HGT_M at the output mass points)
and lets the unchanged NativeHRRR writers (to_wrfinput / to_wrfbdy) produce the files:

1. Horizontal: the HRRR state on its hybrid levels is bilinearly interpolated to the output
   mass / U / V points with NativeHRRR.interpxy, exactly as to_wrfinput does.
2. Surface pressure on the new ground (hypsometric_psfc): the old column is anchored at
   (HGT_old, PSFC_old) and integrated hypsometrically with virtual temperature varying
   linearly in ln p between levels; below the old ground the column is extended with a
   constant virtual-temperature lapse rate (default 6.5 K/km).
3. Target levels: p = C3H (PSFC_new - p_top) + C4H + p_top (same eta, same p_top), the
   WRF-style hybrid-level placement given by C3 and C4.
4. Vertical: every field is interpolated linearly in ln p (interp_lnp). Below the lowest
   source level temperature follows the same lapse rate in T(p) form,
   T = T_0 (p/p_0)^(R_d lapse / g) (WRF real.exe t_extrap_type = 2); every other field
   (qv, hydrometeors, u, v, w) is held constant (real.exe extrap_type = 2).
5. erftools' own derive_fields / calc_real / calc_perts then rebuild theta, theta_m, rho_d,
   the base state (RealInit for the NEW terrain) and the perturbations, and PH is integrated
   hydrostatically as WRF real.exe does (hydrostatic_ph).

The result is a RemappedHRRR whose dataset is already on the output grid; its interpxy picks
grid points instead of interpolating, so to_wrfinput / to_wrfbdy (with their boundary
coupling and orientation conventions) are reused unchanged.

Static land fields from any geo_em-style NetCDF: read_geogrid, check_geogrid_grid,
map_static_fields (the one geo_em -> wrfinput mapping; with lsm_fields also VEGFRA, SHDMIN,
SHDMAX, LAI, ALBBCK, SNOALB and TMN), add_geogrid_land_fields.
The soil state (TSLB, SMOIS, SH2O, ZS, DZS) comes from the HRRR GRIB file itself, see
erftools.preprocessing.hrrr_land; noahmp_input.add_noahmp_land_state writes all of it.

Nothing in this module is used by NativeHRRR's own code paths.
"""
import numpy as np
import pandas as pd
import xarray as xr

from ..constants import R_d, R_v, Cp_d, CONST_GRAV, p_0
from .hrrr import NativeHRRR

STD_LAPSE = 0.0065  # K/m, standard atmosphere / WRF real.exe
HYDROMETEORS = ('QCLOUD', 'QRAIN', 'QICE', 'QSNOW', 'QGRAUP')
BELOW_TOL = 1e-9    # in ln p: a target this close to the lowest source level is not extrapolated


# ---------------------------------------------------------------------------------------
# column operations (all arrays: vertical axis first, ordered bottom -> top)
# ---------------------------------------------------------------------------------------

def virtual_temperature(T, qv):
    """T_v from temperature and water-vapour mixing ratio."""
    return T * (1.0 + qv * R_v / R_d) / (1.0 + qv)


def _bracket(lnp_src, lnp_dst):
    """Lower bracketing index and linear weight in ln p for every target point.

    lnp_src: (ns, ...) strictly decreasing along axis 0; lnp_dst: (nd, ...).
    Returns k_lo (nd, ...), w (nd, ...) unclipped, below mask, above mask."""
    ns = lnp_src.shape[0]
    if not np.all(np.diff(lnp_src, axis=0) < 0):
        raise ValueError('source pressure must decrease strictly with k')
    cnt = np.empty(lnp_dst.shape, dtype=np.int64)
    for k in range(lnp_dst.shape[0]):
        cnt[k] = (lnp_src > lnp_dst[k][None]).sum(axis=0)
    k_lo = np.clip(cnt - 1, 0, ns - 2)
    lo = np.take_along_axis(lnp_src, k_lo, axis=0)
    hi = np.take_along_axis(lnp_src, k_lo + 1, axis=0)
    w = (lnp_dst - lo) / (hi - lo)
    return k_lo, w, cnt == 0, cnt == ns


def interp_lnp(lnp_src, f_src, lnp_dst, bracket=None):
    """Linear interpolation in ln p, column by column; constant beyond the source column."""
    k_lo, w, below, above = bracket if bracket is not None else _bracket(lnp_src, lnp_dst)
    w = np.clip(w, 0.0, 1.0)
    f_lo = np.take_along_axis(f_src, k_lo, axis=0)
    f_hi = np.take_along_axis(f_src, k_lo + 1, axis=0)
    return f_lo + w * (f_hi - f_lo)


def interp_lnp_temperature(lnp_src, T_src, lnp_dst, lapse_rate=STD_LAPSE, bracket=None):
    """As interp_lnp, but below the lowest source level T follows a constant lapse rate,
    T = T_0 (p/p_0)^(R_d lapse / g) (hydrostatic form of dT/dz = -lapse)."""
    br = bracket if bracket is not None else _bracket(lnp_src, lnp_dst)
    out = interp_lnp(lnp_src, T_src, lnp_dst, br)
    below = lnp_dst > lnp_src[0][None] + BELOW_TOL
    ext = T_src[0][None] * np.exp(R_d * lapse_rate / CONST_GRAV * (lnp_dst - lnp_src[0][None]))
    return np.where(below, ext, out)


def _augmented_column(hgt_old, psfc_old, p_src, tv_src, lapse_rate):
    """Old column with its ground as level 0: ln p, T_v and hypsometric heights (bottom->top)."""
    lnp = np.log(p_src)
    lnps = np.log(psfc_old)
    if np.any(lnps <= lnp[0]):
        raise ValueError('PSFC must exceed the lowest-level pressure in every column')
    tv_s = tv_src[0] * np.exp(R_d * lapse_rate / CONST_GRAV * (lnps - lnp[0]))
    lnp_a = np.concatenate([lnps[None], lnp], axis=0)
    tv_a = np.concatenate([tv_s[None], tv_src], axis=0)
    dz = R_d / CONST_GRAV * 0.5 * (tv_a[:-1] + tv_a[1:]) * (lnp_a[:-1] - lnp_a[1:])
    z_a = np.concatenate([hgt_old[None], hgt_old[None] + np.cumsum(dz, axis=0)], axis=0)
    return lnp_a, tv_a, z_a


def hypsometric_psfc(hgt_new, hgt_old, psfc_old, p_src, tv_src, lapse_rate=STD_LAPSE):
    """Surface pressure at height hgt_new from the old column.

    Above the old ground: hypsometric integration through the old column, T_v linear in ln p
    within each layer (exact inverse of the layer thickness used for the heights).
    Below the old ground: T_v = T_v,s + lapse (hgt_old - z), p = PSFC_old (T_v/T_v,s)^(g/(R_d lapse)).
    Returns (psfc_new, heights of the augmented old column)."""
    lnp_a, tv_a, z_a = _augmented_column(hgt_old, psfc_old, p_src, tv_src, lapse_rate)
    na = z_a.shape[0]
    j = np.clip((z_a <= hgt_new[None]).sum(axis=0) - 1, 0, na - 2)[None]
    z_j = np.take_along_axis(z_a, j, 0)[0]
    tv_j = np.take_along_axis(tv_a, j, 0)[0]
    tv_k = np.take_along_axis(tv_a, j + 1, 0)[0]
    big_x = np.take_along_axis(lnp_a, j, 0)[0] - np.take_along_axis(lnp_a, j + 1, 0)[0]
    a = (tv_k - tv_j) / big_x
    c = CONST_GRAV * (hgt_new - z_j) / R_d
    x = 2.0 * c / (tv_j + np.sqrt(np.maximum(tv_j * tv_j + 2.0 * a * c, 0.0)))
    p_above = np.exp(np.take_along_axis(lnp_a, j, 0)[0] - x)
    tv_s = tv_a[0]
    p_below = psfc_old * ((tv_s + lapse_rate * (hgt_old - hgt_new)) / tv_s) ** (
        CONST_GRAV / (R_d * lapse_rate))
    return np.where(hgt_new >= hgt_old, p_above, p_below), z_a


def heights_at_pressure(lnp_a, tv_a, z_a, lnp_dst, lapse_rate=STD_LAPSE):
    """Hypsometric height of pressures lnp_dst in the augmented old column (inverse of
    hypsometric_psfc), lapse-rate extension below the old ground."""
    br = _bracket(lnp_a, lnp_dst)
    k_lo = br[0]
    x = np.take_along_axis(lnp_a, k_lo, 0) - lnp_dst
    big_x = np.take_along_axis(lnp_a, k_lo, 0) - np.take_along_axis(lnp_a, k_lo + 1, 0)
    tv_j = np.take_along_axis(tv_a, k_lo, 0)
    a = (np.take_along_axis(tv_a, k_lo + 1, 0) - tv_j) / big_x
    z = np.take_along_axis(z_a, k_lo, 0) + R_d / CONST_GRAV * (tv_j * x + 0.5 * a * x * x)
    below = lnp_dst > lnp_a[0][None]
    tv_s, ps, zs = tv_a[0][None], np.exp(lnp_a[0])[None], z_a[0][None]
    z_below = zs - tv_s / lapse_rate * ((np.exp(lnp_dst) / ps) ** (R_d * lapse_rate / CONST_GRAV) - 1.0)
    return np.where(below, z_below, z)


def target_pressures(c3, c4, psfc, p_top):
    """Hybrid-level pressures, p = C3 (PSFC - p_top) + C4 + p_top (WRF-style hybrid-level placement)."""
    return c3[:, None, None] * (psfc[None] - p_top) + c4[:, None, None] + p_top


def stagger_psfc(psfc, axis):
    """Average a mass-point 2-D field to U (axis=1) or V (axis=0) points; edge faces take
    the adjacent column."""
    a = np.moveaxis(psfc, axis, 0)
    out = np.concatenate([a[:1], 0.5 * (a[1:] + a[:-1]), a[-1:]], axis=0)
    return np.moveaxis(out, 0, axis)


def hydrostatic_ph(ds):
    """WRF real.exe hydrostatic geopotential, (hydrostatic_ph):
    phi(k+1) = phi(k) - alpha_d(k) (C1H(k) mu_d + C2H(k)) dnw(k), phi(0) = PHB(0) + PH(0) = g HGT,
    alpha_d = ALB + AL, mu_d = MUB + MU. ERF rebuilds the wrfbdy density as
    rho = -(C1H mu + C2H) / (d(phi) RDNW), which this makes equal to 1/(ALB + AL)."""
    alpha = (ds["ALB"] + ds["AL"]).transpose("bottom_top", "south_north", "west_east").values
    mud = (ds["MUB"] + ds["MU"]).transpose("south_north", "west_east").values
    c1h, c2h = ds["C1H"].values, ds["C2H"].values
    dnw = 1.0 / ds["RDNW"].values  # negative
    phb = ds["PHB"].transpose("bottom_top_stag", "south_north", "west_east").values
    ph_old = ds["PH"].transpose("bottom_top_stag", "south_north", "west_east").values
    phi = np.empty_like(phb)
    phi[0] = phb[0] + ph_old[0]
    for k in range(alpha.shape[0]):
        phi[k + 1] = phi[k] - alpha[k] * (c1h[k] * mud + c2h[k]) * dnw[k]
    dz_change = (phi - (phb + ph_old)) / CONST_GRAV
    ds["PH"] = (("bottom_top_stag", "south_north", "west_east"), phi - phb)
    return {"hydrostatic_ph_top_height_change_m": [float(dz_change[-1].min()), float(dz_change[-1].max())],
            "hydrostatic_ph_k1_height_change_m": [float(dz_change[1].min()), float(dz_change[1].max())]}


def smooth_121(field, passes=1):
    """Optional 2-D 1-2-1 smoother (edges held), for terrain; not applied by default."""
    f = np.array(field, dtype=np.float64)
    for _ in range(passes):
        g = f.copy()
        g[1:-1, :] = 0.25 * f[:-2, :] + 0.5 * f[1:-1, :] + 0.25 * f[2:, :]
        h = g.copy()
        h[:, 1:-1] = 0.25 * g[:, :-2] + 0.5 * g[:, 1:-1] + 0.25 * g[:, 2:]
        f = h
    return f


# ---------------------------------------------------------------------------------------
# the remapped state, already on the output grid
# ---------------------------------------------------------------------------------------

class RemappedHRRR(NativeHRRR):
    """NativeHRRR whose dataset already lives on the output grid (mass fields on
    (south_north, west_east), U on west_east_stag, V on south_north_stag).

    interpxy returns the values AT the requested output points (which must be output grid
    points: mass, U or V, chosen by field name) rather than interpolating, so to_wrfinput
    and to_wrfbdy work unchanged. Built by remap_snapshot."""

    def __init__(self, *args, **kwargs):
        raise TypeError('build a RemappedHRRR with remap_snapshot()')

    def _point_indices(self, name, xi, yi):
        if name == 'U':
            gx, gy = self.xg_u, self.yg_u
        elif name == 'V':
            gx, gy = self.xg_v, self.yg_v
        else:
            gx, gy = self.xg, self.yg
        dx = gx[1, 0] - gx[0, 0]
        dy = gy[0, 1] - gy[0, 0]
        ii = np.rint((xi - gx[0, 0]) / dx).astype(int)
        jj = np.rint((yi - gy[0, 0]) / dy).astype(int)
        if (ii.min() < 0 or jj.min() < 0 or ii.max() >= gx.shape[0] or jj.max() >= gx.shape[1]
                or not np.allclose(gx[ii, jj], xi, rtol=0, atol=1e-3)
                or not np.allclose(gy[ii, jj], yi, rtol=0, atol=1e-3)):
            raise ValueError(f'{name}: requested points are not output grid points')
        return ii, jj

    def interpxy(self, name, xi, yi, dtype=float):
        da = self.ds[name].astype(dtype)
        xdim = [d for d in da.dims if d.startswith('west_east')][0]
        ydim = [d for d in da.dims if d.startswith('south_north')][0]
        zdim = [d for d in da.dims if d.startswith('bottom_top')]
        ii, jj = self._point_indices(name, xi, yi)
        if zdim:
            vals = da.transpose(zdim[0], ydim, xdim).values[:, jj.T, ii.T]
            dims = (zdim[0], 'south_north', 'west_east')
        else:
            vals = da.transpose(ydim, xdim).values[jj.T, ii.T]
            dims = ('south_north', 'west_east')
        # unstaggered horizontal names, as NativeHRRR.interpxy returns (callers rename)
        return xr.DataArray(vals, dims=dims)


def remap_snapshot(snap, hgt_new, eta, p_top=10000.0, lapse_rate=STD_LAPSE, check='warn'):
    """Rebuild a NativeHRRR snapshot (on its hybrid levels, set_output_grid already called)
    over the terrain hgt_new (output mass points, (south_north, west_east) array, metres).

    eta: full (staggered) eta levels of the output column (same as the snapshot's);
    p_top: model top pressure [Pa]. Returns (RemappedHRRR, stats dict, diag Dataset)."""
    ds = snap.ds
    if 'bottom_top' not in ds.dims or snap.xg is None:
        raise ValueError('snapshot must be on its hybrid levels with set_output_grid called')
    hgt_new = np.asarray(hgt_new, dtype=np.float64)
    ny, nx = snap.xg.shape[1], snap.xg.shape[0]
    if hgt_new.shape != (ny, nx):
        raise ValueError(f'hgt_new shape {hgt_new.shape} != output mass grid {(ny, nx)}')
    hyd = [v for v in HYDROMETEORS if v in ds]

    # total pressure on the HRRR grid; dry theta (T) is interpolated horizontally, as
    # to_wrfinput does, and turned into temperature with the interpolated pressure
    ptot = ds['P'] + ds['PB']
    shell = NativeHRRR.__new__(NativeHRRR)
    shell.ds = ds.assign(_PTOT=ptot)
    shell.verbose = False

    def hz(name, kind='m'):
        xi, yi = {'m': (snap.xg, snap.yg), 'u': (snap.xg_u, snap.yg_u),
                  'v': (snap.xg_v, snap.yg_v)}[kind]
        return shell.interpxy(name, xi, yi).values.astype(np.float64)

    # 1. horizontal interpolation of the old (HRRR-terrain) state
    p_src = hz('_PTOT')
    t_src = (hz('T') + 300.0) * (p_src / p_0) ** (R_d / Cp_d)
    qv_src = hz('QVAPOR')
    hgt_old = hz('HGT')
    psfc_old = hz('PSFC')

    # 2. surface pressure on the new ground
    tv_src = virtual_temperature(t_src, qv_src)
    psfc_new, _ = hypsometric_psfc(hgt_new, hgt_old, psfc_old, p_src, tv_src, lapse_rate)
    lnp_a, tv_a, z_a = _augmented_column(hgt_old, psfc_old, p_src, tv_src, lapse_rate)

    # 3. target levels
    c3h, c4h = ds['C3H'].values, ds['C4H'].values
    c3f, c4f = ds['C3F'].values, ds['C4F'].values
    p_dst = target_pressures(c3h, c4h, psfc_new, p_top)
    lnp_src, lnp_dst = np.log(p_src), np.log(p_dst)
    br = _bracket(lnp_src, lnp_dst)

    # 4. vertical interpolation, mass points
    t_new = interp_lnp_temperature(lnp_src, t_src, lnp_dst, lapse_rate, br)
    qv_new = interp_lnp(lnp_src, qv_src, lnp_dst, br)
    hyd_new = {v: interp_lnp(lnp_src, hz(v), lnp_dst, br) for v in hyd}
    z_new = heights_at_pressure(lnp_a, tv_a, z_a, lnp_dst, lapse_rate)

    # W on the full levels, face pressures from the same column functions
    lnpf_src = np.log(target_pressures(c3f, c4f, psfc_old, p_top))
    lnpf_dst = np.log(target_pressures(c3f, c4f, psfc_new, p_top))
    w_new = interp_lnp(lnpf_src, hz('W'), lnpf_dst)

    # U, V at their own points: source pressure and PSFC interpolated there (as to_wrfinput
    # places them); the target PSFC adds the mass-point PSFC change averaged to the faces
    # (one-sided at the domain edges, as ERF staggers mu), so an unchanged terrain is an identity
    dpsfc = psfc_new - psfc_old
    u_new = interp_lnp(np.log(hz('_PTOT', 'u')), hz('U', 'u'),
                       np.log(target_pressures(c3h, c4h, hz('PSFC', 'u') + stagger_psfc(dpsfc, 1), p_top)))
    v_new = interp_lnp(np.log(hz('_PTOT', 'v')), hz('V', 'v'),
                       np.log(target_pressures(c3h, c4h, hz('PSFC', 'v') + stagger_psfc(dpsfc, 0), p_top)))

    # 5. erftools derive_fields / calc_real / calc_perts on the new columns
    m3 = ('bottom_top', 'south_north', 'west_east')
    m2 = ('south_north', 'west_east')
    new = xr.Dataset({
        'pres': (m3, p_dst), 't': (m3, t_new), 'q': (m3, qv_new / (1.0 + qv_new)),
        'gh': (m3, z_new), 'w': (m3, np.zeros_like(t_new)),  # omega unused: W set below
        'U': (('bottom_top', 'south_north', 'west_east_stag'), u_new),
        'V': (('bottom_top', 'south_north_stag', 'west_east'), v_new),
        'HGT': (m2, hgt_new), 'PSFC': (m2, psfc_new),
        'SST': (m2, hz('SST')), 'LANDMASK': (m2, hz('LANDMASK')),
    })
    for v in hyd:
        new[v] = (m3, hyd_new[v])

    rem = RemappedHRRR.__new__(RemappedHRRR)
    rem.ds = new
    rem.verbose = False
    for attr in ('datetime', 'grid', 'xg', 'yg', 'xg_u', 'yg_u', 'xg_v', 'yg_v'):
        setattr(rem, attr, getattr(snap, attr))
    rem.derive_fields(check, inplace=True)
    rem.ds['W'] = (('bottom_top_stag', 'south_north', 'west_east'), w_new)
    rem.ds['P_TOP'] = float(p_top)  # the levels were placed with this p_top exactly
    eta_da = xr.DataArray(np.asarray(eta, dtype=np.float64), dims='bottom_top_stag', name='eta')
    rem.calc_real(eta=eta_da, inplace=True)
    rem.calc_perts(None, inplace=True)  # PH from gh is replaced by the hydrostatic PH next
    stats = hydrostatic_ph(rem.ds)

    dh = hgt_new - hgt_old
    dp = psfc_new - psfc_old
    nbelow = (lnp_dst > lnp_src[0][None] + BELOW_TOL).sum(axis=0)
    below = dh < 0
    sel = np.abs(dh) > 20.0
    # least-squares dPSFC/dHGT in Pa/m, which is numerically hPa per 100 m
    slope = float(np.polyfit(dh[sel], dp[sel], 1)[0]) if sel.sum() > 10 else float('nan')
    stats.update({
        'hgt_old_range_m': [float(hgt_old.min()), float(hgt_old.max())],
        'hgt_new_range_m': [float(hgt_new.min()), float(hgt_new.max())],
        'dhgt_range_m': [float(dh.min()), float(dh.max())],
        'dpsfc_range_hPa': [float(dp.min()) / 100.0, float(dp.max()) / 100.0],
        'dpsfc_per_100m_hPa_fit': slope,
        'columns_new_ground_below_old': int(below.sum()),
        'target_levels_below_old_lowest_level': int(nbelow.sum()),
        'max_levels_extrapolated_in_a_column': int(nbelow.max()),
        'lapse_rate_K_per_m': lapse_rate,
    })
    diag = xr.Dataset({
        'HGT_OLD': (m2, hgt_old), 'HGT_NEW': (m2, hgt_new),
        'PSFC_OLD': (m2, psfc_old), 'PSFC_NEW': (m2, psfc_new),
        'NLEV_EXTRAP': (m2, nbelow.astype(np.int32)),
    })
    return rem, stats, diag


# ---------------------------------------------------------------------------------------
# static land fields from any geo_em-style NetCDF (WPS geogrid, or another producer)
# ---------------------------------------------------------------------------------------
#
# Inputs relied on (names as geo_em.dNN.nc; Time dimension optional):
#   grid check : XLAT_M/XLONG_M, XLAT_U/XLONG_U, XLAT_V/XLONG_V; MAPFAC_M optional
#   terrain    : HGT_M (mass points)
#   land use   : LU_INDEX, or LANDUSEF(land_cat) -> dominant category; global attrs MMINLU and
#                NUM_LAND_CAT (or passed explicitly); ISWATER/ISLAKE/ISICE/ISURBAN/ISOILWATER
#                optional (defaults from the MMINLU table below)
#   soil       : SCT_DOM, or SOILCTOP(soil_cat) -> dominant category
#   LANDMASK   : optional; derived as LU != ISWATER/ISLAKE if absent
#   LSM only   : SOILTEMP [K, sea level], GREENFRAC(month=12) [fraction], LAI12M(month=12),
#                ALBEDO12M(month=12) [percent], SNOALB [percent]; months January first
# Nothing here depends on WPS file names or WPS-only attributes.

# category indices by land-use table (WRF LANDUSE.TBL / VEGPARM.TBL conventions)
LANDUSE_TABLES = {
    'MODIFIED_IGBP_MODIS_NOAH': {'ISWATER': 17, 'ISLAKE': 21, 'ISICE': 15, 'ISURBAN': 13},
    'USGS': {'ISWATER': 16, 'ISLAKE': -1, 'ISICE': 24, 'ISURBAN': 1},
    'NLCD40': {'ISWATER': 17, 'ISLAKE': -1, 'ISICE': 15, 'ISURBAN': 13},
}
ISOILWATER_DEFAULT = 14  # STAS / STAS-RUC soil tables
LSM_GEO_FIELDS = ('SOILTEMP', 'GREENFRAC', 'LAI12M', 'ALBEDO12M', 'SNOALB')
WATER_ALBEDO = 0.08      # real.exe ALBBCK / SNOALB over water


def read_geogrid(path):
    """A geo_em-style NetCDF as an xarray Dataset, Time dimension dropped if present."""
    geo = xr.open_dataset(path).load()
    if 'Time' in geo.dims:
        geo = geo.isel(Time=0)
    return geo


def check_geogrid_grid(geo, grid, tol_deg=1e-4, tol_msf=1e-5):
    """Compare a geo_em-style dataset with the erftools output grid: shapes and lat/lon at
    mass, U and V points, and MAPFAC_M if present. Raises ValueError if out of tolerance.
    Projection attributes are only recorded, not required."""
    lat, lon = [a[0] for a in grid.calc_lat_lon()]
    lat_u, lon_u = [a[0] for a in grid.calc_lat_lon('U')]
    lat_v, lon_v = [a[0] for a in grid.calc_lat_lon('V')]
    rec = {'dims': {k: int(v) for k, v in geo.sizes.items()}}
    pairs = {'M': ('XLAT_M', 'XLONG_M', lat, lon), 'U': ('XLAT_U', 'XLONG_U', lat_u, lon_u),
             'V': ('XLAT_V', 'XLONG_V', lat_v, lon_v)}
    ok = True
    for key, (vla, vlo, la, lo) in pairs.items():
        gla, glo = geo[vla].values.astype(np.float64), geo[vlo].values.astype(np.float64)
        if gla.shape != la.shape:
            rec[f'{key}_shape'] = [list(gla.shape), list(la.shape)]
            ok = False
            continue
        dla, dlo = float(np.abs(gla - la).max()), float(np.abs(glo - lo).max())
        rec[f'{key}_max_abs_dlat_deg'], rec[f'{key}_max_abs_dlon_deg'] = dla, dlo
        ok &= dla < tol_deg and dlo < tol_deg
    if 'MAPFAC_M' in geo:
        dm = float(np.abs(geo['MAPFAC_M'].values - grid.calc_msf(lat)).max())
        rec['MAPFAC_M_max_abs_diff'] = dm
        ok &= dm < tol_msf
    rec['attrs'] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in geo.attrs.items()
                    if k in ('MAP_PROJ', 'DX', 'DY', 'TRUELAT1', 'TRUELAT2', 'STAND_LON',
                             'CEN_LAT', 'CEN_LON')}
    rec['pass'] = bool(ok)
    if not ok:
        raise ValueError(f'geo_em-style file does not match the output grid: {rec}')
    return rec


def monthly_interp_to_date(field12, date):
    """WRF real.exe monthly_interp_to_date (dyn_em/module_initialize_real.F): monthly values
    valid on the 15th, linear in (integer) Julian day; Dec/Jan wrap with 31-day spacing.
    field12: (12, ...) January first. 2021-12-24: (22 x Dec + 9 x Jan) / 31."""
    date = pd.Timestamp(date)
    middle = np.array([pd.Timestamp(date.year, m, 15).dayofyear for m in range(1, 13)])
    middle = np.concatenate([[middle[0] - 31], middle, [middle[11] + 31]])
    target = date.dayofyear
    for l in range(13):
        if middle[l] < target <= middle[l + 1]:
            m1, m2 = (12, 1) if l in (0, 12) else (l, l + 1)
            return ((field12[m2 - 1] * (target - middle[l]) + field12[m1 - 1] * (middle[l + 1] - target))
                    / (middle[l + 1] - middle[l]))
    raise ValueError(f'no bracketing months for {date}')


def _monthly(geo, name):
    """A 12-month geo_em field as (12, south_north, west_east) float64, January first. The
    month axis is found by name ('month', as geogrid writes it), not by position."""
    da = geo[name]
    mdim = 'month' if 'month' in da.dims else None
    if mdim is None or da.sizes[mdim] != 12:
        raise ValueError(f'{name} must have a 12-long month dimension, got {dict(da.sizes)}')
    arr = da.transpose(mdim, ...).values.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} has non-finite values')
    return arr


def _dominant(frac):
    """1-based index of the largest fraction along axis 0 (geogrid's dominant category;
    ties -> lowest index)."""
    return (np.argmax(frac, axis=0) + 1).astype(np.float64)


def map_static_fields(geo, date, lsm_fields=False, sst=None, mminlu=None, num_land_cat=None):
    """THE geo_em -> wrfinput static-field mapping (reusable by any producer of geo_em-style
    fields), following WRF real.exe with surface_input_source = 3:

      IVGTYP   = nint(LU_INDEX) [or dominant LANDUSEF], inland lake (ISLAKE) -> ISWATER
      ISLTYP   = nint(SCT_DOM)  [or dominant SOILCTOP]; land with ISOILWATER soil -> 8,
                 water -> ISOILWATER
      LANDMASK = LANDMASK [or IVGTYP not water/lake]
      with lsm_fields (inputs: LSM_GEO_FIELDS):
      VEGFRA   = 100 x GREENFRAC interpolated to `date` (monthly_interp_to_date), [0, 100];
                 0 on water
      SHDMIN/SHDMAX = 100 x min/max of GREENFRAC over the 12 months
      LAI      = LAI12M interpolated to `date`, >= 0
      ALBBCK   = ALBEDO12M interpolated to `date` / 100 (percent -> fraction), on water 0.08
      SNOALB   = SNOALB / 100 (maximum snow albedo), on water 0.08
      TMN      = SOILTEMP - 0.0065 HGT_M on land (share/module_soil_pre.F adjust_soil_temp_new,
                 Noah / Noah-MP), = sst on water if sst is given
    The water values and the percent conversions follow WRF real.exe as documented
    (module_initialize_real.F); there is no WRF source in this repository to check them against.
    GREENFRAC must be a fraction (geogrid writes it so): a percent field raises, since 100 x
    percent would pass every later check. The Noah-MP reader of ERF reads VEGFRA, LAI, SHDMIN,
    SHDMAX and TMN but not ALBBCK or SNOALB, which are written for WRF parity.

    Returns (fields: name -> 2-D float64 array, attrs: MMINLU, NUM_LAND_CAT, IS* for wrfinput)."""
    def attr(name, override=None):
        if override is not None:
            return override
        v = geo.attrs.get(name)
        return v.item() if hasattr(v, 'item') else v

    mminlu = attr('MMINLU', mminlu)
    if mminlu is None:
        raise ValueError('MMINLU (land-use table name) is required, as an attribute or argument')
    mminlu = str(mminlu).strip()
    table = LANDUSE_TABLES.get(mminlu, {})
    if 'LU_INDEX' in geo:
        ivgtyp = np.rint(geo['LU_INDEX'].values).astype(np.float64)
    else:
        ivgtyp = _dominant(geo['LANDUSEF'].values)
    num_land_cat = attr('NUM_LAND_CAT', num_land_cat)
    if num_land_cat is None:
        num_land_cat = geo['LANDUSEF'].shape[0] if 'LANDUSEF' in geo else int(ivgtyp.max())
    cats = {}
    for k in ('ISWATER', 'ISLAKE', 'ISICE', 'ISURBAN'):
        v = attr(k)
        cats[k] = int(v) if v is not None else table.get(k)
        if cats[k] is None:
            raise ValueError(f'{k} unknown for land-use table {mminlu!r}; pass it as an attribute')
    v = attr('ISOILWATER')
    cats['ISOILWATER'] = int(v) if v is not None else ISOILWATER_DEFAULT
    iswater, islake, isoilwater = cats['ISWATER'], cats['ISLAKE'], cats['ISOILWATER']

    if 'LANDMASK' in geo:
        landmask = geo['LANDMASK'].values.astype(np.float64)
    else:
        landmask = ((ivgtyp != iswater) & (ivgtyp != islake)).astype(np.float64)
    if islake > 0:
        ivgtyp = np.where(ivgtyp == islake, float(iswater), ivgtyp)
    if 'SCT_DOM' in geo:
        isltyp = np.rint(geo['SCT_DOM'].values).astype(np.float64)
    else:
        isltyp = _dominant(geo['SOILCTOP'].values)
    land = landmask > 0.5
    isltyp = np.where(land & (isltyp == isoilwater), 8.0, isltyp)
    isltyp = np.where(~land & (isltyp != isoilwater), float(isoilwater), isltyp)
    out = {'IVGTYP': ivgtyp, 'ISLTYP': isltyp, 'LANDMASK': landmask}
    if lsm_fields:
        missing = [v for v in LSM_GEO_FIELDS if v not in geo]
        if missing:
            raise KeyError(f'lsm_fields needs {missing} in the geo_em-style dataset')
        gf = _monthly(geo, 'GREENFRAC')
        if gf.min() < 0.0 or gf.max() > 1.0 + 1e-6:
            raise ValueError(f'GREENFRAC must be a fraction in [0, 1], range '
                             f'[{gf.min():g}, {gf.max():g}]')
        out['VEGFRA'] = np.where(land, np.clip(100.0 * monthly_interp_to_date(gf, date), 0.0, 100.0), 0.0)
        out['SHDMIN'] = 100.0 * gf.min(axis=0)
        out['SHDMAX'] = 100.0 * gf.max(axis=0)
        out['LAI'] = np.maximum(monthly_interp_to_date(_monthly(geo, 'LAI12M'), date), 0.0)
        albbck = monthly_interp_to_date(_monthly(geo, 'ALBEDO12M'), date) / 100.0
        snoalb = geo['SNOALB'].values.astype(np.float64) / 100.0
        for name, alb in (('ALBBCK', albbck), ('SNOALB', snoalb)):
            if not np.all(np.isfinite(alb)) or np.any(alb < 0.0) or np.any(alb > 1.0):
                raise ValueError(f'{name} outside [0, 1] after the percent conversion')
            out[name] = np.where(land, alb, WATER_ALBEDO)
        tmn = geo['SOILTEMP'].values.astype(np.float64) - STD_LAPSE * geo['HGT_M'].values
        if sst is not None:
            tmn = np.where(land, tmn, np.asarray(sst, dtype=np.float64))
        out['TMN'] = tmn
    attrs = {'MMINLU': mminlu, 'NUM_LAND_CAT': int(num_land_cat), **cats}
    return out, attrs


def add_geogrid_land_fields(inp, geo, date, lsm_fields=False, dtype=np.float32, **kw):
    """Write map_static_fields into a to_wrfinput Dataset (Time, south_north, west_east) and
    carry MMINLU / NUM_LAND_CAT / IS* as global attributes. SST/TSK are left as they are;
    with lsm_fields, TMN over water is the dataset's SST."""
    sst = inp['SST'].values[0] if 'SST' in inp else None
    fields, attrs = map_static_fields(geo, date, lsm_fields=lsm_fields, sst=sst, **kw)
    for name, arr in fields.items():
        inp[name] = (('Time', 'south_north', 'west_east'), arr[None].astype(dtype))
    for k, v in attrs.items():
        inp.attrs[k] = v if isinstance(v, str) else np.int32(v)
    return inp


def soil_state_from_hrrr(*args, **kwargs):
    """Soil state for an LSM run (TSLB, SMOIS, SH2O, ZS, DZS) from HRRR's RUC soil levels:
    see erftools.preprocessing.hrrr_land.soil_state_from_hrrr, which this calls."""
    from .hrrr_land import soil_state_from_hrrr as _soil
    return _soil(*args, **kwargs)
