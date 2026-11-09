"""Processing presets.

A preset is a plain dict:

  {"name": ..., "grid": "wps_f32" | "pyproj",
   "landuse": {"modis": {...}, "nlcd": {...}},       # the LANDUSEF entry per choice
   "fields": [ {field spec}, ... ]}                   # processed in this order

Field spec keys
  name         output variable name
  kind         "continuous" | "categorical"
  source       dataset directory relative to the data root (or absolute)
  method       "wps" (geogrid-equivalent; uses ``interp``) or one of the methods
               in erftools.geog.methods (nearest, bilinear, bicubic, cell_bilinear,
               area_mean, area_fraction)
  interp       geogrid interp_option string for method "wps"
  opts         method options (supersample, fill)
  masked       "water" | "land" | None   (WPS semantics: fill value there)
  fill_missing value used where masked / no data
  z_dim        name of the 3rd dimension (land_cat, soil_cat, month, crop)
  dominant     name of the dominant-category output (categorical)
  dominant_rule "wps" | "landmask_consistent"
  landmask_water list of water categories (LANDUSEF only)
  filter       {"name": ..., **options}  (terrain filter, HGT_M)
  flag         global attribute set to 1 when the field is written
  extras       list of extra diagnostic outputs (HGT_M only):
               "HGT_M_RAW", "HGT_M_STD", "SLOPE"

Copy a preset with ``copy.deepcopy`` and edit any entry to build new
variants; the CLI also accepts a JSON file with the same structure.
"""
import copy
import os

# WPS_GEOG_V4 root: the WPS_GEOG_PATH environment variable, or the ``data_root``
# argument of pipeline.run / ``--data-root`` of the CLI (no default)
DATA_ROOT = os.environ.get("WPS_GEOG_PATH", "")

LANDUSE_TABLE = {
    "modis": {"mminlu": "MODIFIED_IGBP_MODIS_NOAH", "water": [17, 21], "iswater": 17, "islake": 21,
              "isice": 15, "isurban": 13, "isoilwater": 14},
    "nlcd": {"mminlu": "NLCD40", "water": [17], "iswater": 17, "islake": -1,
             "isice": 15, "isurban": 13, "isoilwater": 14},
}


def _gwd(names, rel, masked="water"):
    return [dict(name=n, kind="continuous", source=rel + n.lower() + "/", method="wps",
                 interp="average_4pt", masked=masked, fill_missing=0.0) for n in names]


_GWD10 = ["CON", "VAR", "OA1", "OA2", "OA3", "OA4", "OL1", "OL2", "OL3", "OL4"]
_GWD3 = [p + s for s in ("LS", "SS") for p in
         ("CON", "VAR", "OA1", "OA2", "OA3", "OA4", "OL1", "OL2", "OL3", "OL4")]

# --------------------------------------------------------------------------
# wps_equivalent: GEOGRID.TBL.ARW.noahmp 'default' entries (WPS v4.4), transcribed
# --------------------------------------------------------------------------
WPS_EQUIVALENT = {
    "name": "wps_equivalent",
    "description": "geogrid v4.4 algorithms as specified by GEOGRID.TBL.ARW.noahmp (res 'default', "
                   "and 'nlcd2011_9s+default' for LANDUSEF), REAL*4 arithmetic",
    "grid": "wps_f32",
    "landuse": {
        "modis": dict(name="LANDUSEF", kind="categorical", source="modis_landuse_20class_30s_with_lakes/",
                      method="wps", interp="nearest_neighbor", z_dim="land_cat", dominant="LU_INDEX",
                      dominant_rule="wps", landmask_water=[17, 21]),
        "nlcd": dict(name="LANDUSEF", kind="categorical", source="nlcd2011_ll_9s/",
                     method="wps", interp="average_gcell(0.0)", z_dim="land_cat", dominant="LU_INDEX",
                     dominant_rule="wps", landmask_water=[17]),
    },
    "fields": [
        dict(name="HGT_M", kind="continuous", source="topo_gmted2010_30s/", method="wps",
             interp="average_gcell(4.0)+four_pt+average_4pt", fill_missing=0.0,
             filter={"name": "smth_desmth_special", "passes": 1}),
        dict(name="SOILTEMP", kind="continuous", source="soiltemp_1deg/", method="wps",
             interp="sixteen_pt+four_pt+average_4pt+average_16pt+search", masked="water", fill_missing=0.0),
        dict(name="SOILCTOP", kind="categorical", source="soiltype_top_30s/", method="wps",
             interp="nearest_neighbor", z_dim="soil_cat", dominant="SCT_DOM", dominant_rule="wps"),
        dict(name="SOILCBOT", kind="categorical", source="soiltype_bot_30s/", method="wps",
             interp="nearest_neighbor", z_dim="soil_cat", dominant="SCB_DOM", dominant_rule="wps"),
        dict(name="ALBEDO12M", kind="continuous", source="albedo_modis/", method="wps",
             interp="four_pt+average_4pt+average_16pt+search", z_dim="month", masked="water",
             fill_missing=8.0),
        dict(name="GREENFRAC", kind="continuous", source="greenfrac_fpar_modis/", method="wps",
             interp="average_gcell(4.0)+four_pt+average_4pt+average_16pt+search", z_dim="month",
             masked="water", fill_missing=0.0),
        dict(name="LAI12M", kind="continuous", source="lai_modis_10m/", method="wps",
             interp="average_gcell(4.0)+four_pt+average_4pt+average_16pt+search", z_dim="month",
             masked="water", fill_missing=0.0, flag="FLAG_LAI12M"),
        dict(name="SNOALB", kind="continuous", source="maxsnowalb_modis/", method="wps",
             interp="four_pt+average_4pt+average_16pt+search", masked="water", fill_missing=0.0),
    ] + _gwd(_GWD10, "orogwd_10m/") + [
        dict(name="VAR_SSO", kind="continuous", source="varsso_10m/", method="wps",
             interp="average_gcell(4.0)+four_pt+average_4pt", fill_missing=0.0, flag="FLAG_VAR_SSO"),
    ] + _gwd(_GWD3, "orogwd3_10m/") + [
        dict(name="CROPTYPE", kind="continuous", source="crop/croptype/", method="wps",
             interp="average_gcell(4.0)+four_pt+average_4pt+average_16pt+search", z_dim="crop",
             masked="water", fill_missing=0.0),
        dict(name="SEASON_GDD", kind="continuous", source="crop/crop_gdd/", method="wps",
             interp="average_gcell(4.0)+four_pt+average_4pt+average_16pt+search", masked="water",
             fill_missing=1605.0),
        dict(name="PLANTING", kind="continuous", source="crop/crop_planting/", method="wps",
             interp="nearest_neighbor", masked="water", fill_missing=126.0),
        dict(name="HARVEST", kind="continuous", source="crop/crop_harvest/", method="wps",
             interp="nearest_neighbor", masked="water", fill_missing=290.0),
        dict(name="FDEPTH", kind="continuous", source="groundwater/f/", method="wps",
             interp="average_gcell(1.0)+average_4pt+search", masked="water", fill_missing=100.0),
        dict(name="RECHCLIM", kind="continuous", source="groundwater/recharge/", method="wps",
             interp="average_gcell(1.0)+average_4pt+search", masked="water", fill_missing=-1.0),
        dict(name="EQZWT", kind="continuous", source="groundwater/wtd/", method="wps",
             interp="average_gcell(1.0)+average_4pt+search", masked="water", fill_missing=0.0),
        dict(name="RIVERBED", kind="continuous", source="groundwater/riverbed/", method="wps",
             interp="average_gcell(1.0)+average_4pt+search", masked="water", fill_missing=0.0),
    ],
}

# --------------------------------------------------------------------------
# slope_limited: area-weighted sampling plus a slope-limited terrain filter (see README)
# --------------------------------------------------------------------------
_C8 = {"supersample": 8, "fill": "nearest_valid"}
SLOPE_LIMITED = {
    "name": "slope_limited",
    "description": "independent double-precision pipeline: pyproj LCC grid; terrain = cell average of "
                   "the bilinear GMTED2010 surface + conservative slope-limited filter (max 0.5 m/m); "
                   "land use / soil = exact-area category fractions; fine vegetation sources "
                   "area-averaged, coarse ones cell-averaged bilinear; no water masking",
    "grid": "pyproj",
    "landuse": {
        "modis": dict(name="LANDUSEF", kind="categorical", source="modis_landuse_20class_30s_with_lakes/",
                      method="area_fraction", opts={"supersample": 8}, z_dim="land_cat",
                      dominant="LU_INDEX", dominant_rule="wps", landmask_water=[17, 21]),
        "nlcd": dict(name="LANDUSEF", kind="categorical", source="nlcd2011_ll_9s/",
                     method="area_fraction", opts={"supersample": 8}, z_dim="land_cat",
                     dominant="LU_INDEX", dominant_rule="wps", landmask_water=[17]),
    },
    "fields": [
        dict(name="HGT_M", kind="continuous", source="topo_gmted2010_30s/", method="cell_bilinear",
             opts={"supersample": 8}, fill_missing=0.0,
             filter={"name": "slope_limited", "max_slope": 0.5, "kappa": 0.2},
             extras=["HGT_M_RAW", "HGT_M_STD", "SLOPE"]),
        dict(name="SOILTEMP", kind="continuous", source="soiltemp_1deg/", method="bilinear",
             opts={"fill": "nearest_valid"}, fill_missing=0.0),
        dict(name="SOILCTOP", kind="categorical", source="soiltype_top_30s/", method="area_fraction",
             opts={"supersample": 8}, z_dim="soil_cat", dominant="SCT_DOM",
             dominant_rule="landmask_consistent"),
        dict(name="SOILCBOT", kind="categorical", source="soiltype_bot_30s/", method="area_fraction",
             opts={"supersample": 8}, z_dim="soil_cat", dominant="SCB_DOM",
             dominant_rule="landmask_consistent"),
        dict(name="ALBEDO12M", kind="continuous", source="albedo_modis/", method="cell_bilinear",
             opts=_C8, z_dim="month", fill_missing=8.0),
        dict(name="GREENFRAC", kind="continuous", source="greenfrac_fpar_modis/", method="area_mean",
             opts=_C8, z_dim="month", fill_missing=0.0),
        dict(name="LAI12M", kind="continuous", source="lai_modis_30s/", method="area_mean",
             opts=_C8, z_dim="month", fill_missing=0.0, flag="FLAG_LAI12M"),
        dict(name="SNOALB", kind="continuous", source="maxsnowalb_modis/", method="cell_bilinear",
             opts=_C8, fill_missing=0.0),
        dict(name="VAR_SSO", kind="continuous", source="varsso/", method="area_mean",
             opts=_C8, fill_missing=0.0, flag="FLAG_VAR_SSO"),
    ],
}

PRESETS = {"wps_equivalent": WPS_EQUIVALENT, "slope_limited": SLOPE_LIMITED}


def get(name):
    return copy.deepcopy(PRESETS[name])
