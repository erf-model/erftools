"""Run a preset on a grid and collect geo_em-style outputs."""
import copy
import json
import os
import time

import numpy as np

from . import filters, methods, wps_engine
from .dataset import open_source
from .grid import GridSpec, ModelGrid
from .presets import DATA_ROOT, LANDUSE_TABLE

F = np.float32
WPS_FILTERS = ("smth_desmth_special", "smth_desmth", "one_two_one")
GRID_VARS = {"XLAT_M": "M", "XLONG_M": "M", "XLAT_V": "V", "XLONG_V": "V", "XLAT_U": "U",
             "XLONG_U": "U", "CLAT": "M", "CLONG": "M", "MAPFAC_M": "M", "MAPFAC_V": "V",
             "MAPFAC_U": "U", "MAPFAC_MX": "M", "MAPFAC_VX": "V", "MAPFAC_UX": "U",
             "MAPFAC_MY": "M", "MAPFAC_VY": "V", "MAPFAC_UY": "U", "E": "M", "F": "M",
             "SINALPHA": "M", "COSALPHA": "M", "XLAT_C": "CORNER", "XLONG_C": "CORNER",
             "SINALPHA_U": "U", "COSALPHA_U": "U", "SINALPHA_V": "V", "COSALPHA_V": "V"}


class Output:
    def __init__(self, name, data, z_dim=None, units="", description="", stagger="M"):
        self.name, self.data, self.z_dim = name, np.asarray(data, F), z_dim
        self.units, self.description, self.stagger = units, description, stagger


def _src(path, root):
    if not path or not (os.path.isabs(path) or root):
        raise ValueError("source %r needs a data root: set WPS_GEOG_PATH (or --data-root) and, for "
                         "slope_limited_3dep, ERFTOOLS_GEOG_3DEP_PATH" % path)
    return open_source(path if os.path.isabs(path) else os.path.join(root, path))


def _cat_range(src, spec):
    if "categories" in spec:
        return tuple(spec["categories"])
    if src.cat_min is not None:
        return src.cat_min, src.cat_max
    return src.z0, src.z1


def run(preset, landuse="modis", grid_spec=None, data_root=DATA_ROOT, log=print):
    t0 = time.time()
    preset = copy.deepcopy(preset)
    gs = grid_spec or GridSpec()
    grid = ModelGrid(gs, method=preset.get("grid", "wps_f32"))
    sl = grid.patch
    outs = {}
    prov = []
    flags = {"FLAG_MF_XY": 1}
    for n in GRID_VARS:
        outs[n] = Output(n, sl(n), stagger=GRID_VARS[n])
    landmask = None
    lu = preset["landuse"][landuse]
    luinfo = LANDUSE_TABLE[landuse]
    specs = [lu] + preset["fields"]
    timings = {}
    for spec in specs:
        ts = time.time()
        name = spec["name"]
        src = _src(spec["source"], data_root)
        prov.append({"field": name, "source": src.path, "index_sha256": src.index_sha256,
                     "method": spec.get("method"), "interp": spec.get("interp"),
                     "opts": spec.get("opts"), "filter": spec.get("filter")})
        meth = spec.get("method", "wps")
        fill = spec.get("fill_missing")
        if spec["kind"] == "categorical":
            c0, c1 = _cat_range(src, spec)
            if meth == "wps":
                w = wps_engine.calc_categorical(grid, src, spec, landmask, (c0, c1), gs.dx)
                frac = wps_engine.fractions(w, wps_engine.NAN_WPS)
                ffill = wps_engine.NAN_WPS
            else:
                frac = methods.sample_categorical(grid, src, meth, spec.get("opts"), (c0, c1))
                ffill = F(np.nan)          # no-data sentinel for landmask_from / dominant
                frac = frac.astype(F)
            if "landmask_water" in spec:
                landmask = wps_engine.landmask_from(frac, spec["landmask_water"], c0, ffill)
                if meth != "wps":
                    landmask = np.where(np.isnan(frac[0]), F(-1.0), landmask).astype(F)
                dom = wps_engine.dominant_with_mask(frac, landmask, spec["landmask_water"], c0)
                outs["LANDMASK"] = Output("LANDMASK", sl(landmask), units="none",
                                          description="Landmask : 1=land, 0=water")
            elif spec.get("dominant_rule", "wps") == "wps":
                dom = wps_engine.dominant_plain(frac, c0, ffill)
            else:
                dom = dominant_landmask_consistent(frac, landmask, c0, water_cat=14)
            if meth != "wps":
                frac = np.where(np.isnan(frac), F(0.0), frac).astype(F)
                dom = np.where(np.isnan(dom), F(0.0), dom).astype(F)
            outs[name] = Output(name, sl(frac), spec.get("z_dim"), "category", src.description)
            if spec.get("dominant"):
                outs[spec["dominant"]] = Output(spec["dominant"], sl(dom), units="category",
                                                description="Dominant category")
        else:
            if meth == "wps":
                fld = wps_engine.calc_continuous(grid, src, spec, landmask, gs.dx)
            else:
                fld = methods.sample_continuous(grid, src, meth, spec.get("opts"))
                f0 = F(0.0) if fill is None else F(fill)
                fld = np.where(np.isnan(fld), f0, fld)
                if spec.get("masked") in ("water", "land") and landmask is not None:
                    mv = 0.0 if spec["masked"] == "water" else 1.0
                    fld = np.where((landmask == mv)[None], f0, fld)
                fld = fld.astype(F) if spec.get("filter") is None else fld
            info = {}
            if spec.get("filter"):
                flt = dict(spec["filter"])
                fname = flt.pop("name")
                raw = fld[0]
                if fname in WPS_FILTERS:
                    filt, info = filters.apply(fname, raw.astype(F), flt, gs.dx, gs.dy)
                    filt_p = sl(filt)
                else:
                    filt_p, info = filters.apply(fname, sl(raw).astype(float), flt, gs.dx, gs.dy,
                                                 mapfac=sl("MAPFAC_M"))
                extras = spec.get("extras", [])
                if "HGT_M_RAW" in extras:
                    outs["HGT_M_RAW"] = Output("HGT_M_RAW", sl(raw), units=src.units,
                                               description="terrain before filtering")
                if "HGT_M_STD" in extras:
                    std = methods.sample_std(grid, src, spec.get("opts", {}).get("supersample", 8))
                    outs["HGT_M_STD"] = Output("HGT_M_STD", sl(std), units="meters",
                                               description="std-dev of source terrain surface within cell")
                if "SLOPE" in extras:
                    outs["SLOPE"] = Output("SLOPE", slope_field(filt_p, gs.dx, gs.dy, sl("MAPFAC_M")),
                                           units="m/m", description="terrain slope magnitude of HGT_M")
                data = np.asarray(filt_p, F)[None]
            else:
                data = sl(fld)
            if spec.get("z_dim") is None:
                data = data[0]
            outs[name] = Output(name, data, spec.get("z_dim"), src.units, src.description)
            if info:
                prov[-1]["filter_info"] = info
        if spec.get("flag"):
            flags[spec["flag"]] = 1
        timings[name] = round(time.time() - ts, 2)
        log("  %-10s %-7s %6.2fs  (%s)" % (name, meth, timings[name], src.path))
    attrs = {"luinfo": luinfo, "flags": flags, "provenance": prov, "timings": timings,
             "runtime_s": round(time.time() - t0, 2), "preset": preset, "landuse": landuse}
    return grid, outs, attrs


def dominant_landmask_consistent(frac, landmask, c0, water_cat=14):
    """Soil dominant category consistent with LANDMASK: on land the largest
    non-water category (water only if nothing else is present), on water the
    water category.  First maximum wins ties (as numpy argmax)."""
    nc = frac.shape[0]
    f = np.array(frac, float)
    k = water_cat - c0
    land_frac = f.copy()
    land_frac[k] = -1.0
    dom_land = np.argmax(land_frac, axis=0) + c0
    none_land = land_frac.max(axis=0) <= 0
    dom_land = np.where(none_land, water_cat, dom_land)
    if landmask is None:
        return dom_land.astype(F)
    return np.where(landmask == 0, water_cat, dom_land).astype(F)


def slope_field(h, dx, dy, mapfac):
    h = np.asarray(h, float)
    gy, gx = np.gradient(h, dy, dx)
    return np.hypot(gx, gy) * np.asarray(mapfac, float)


def summary_json(attrs):
    return json.dumps(attrs, default=str)
