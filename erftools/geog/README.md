# erftools.geog — Python geogrid replacement ("option B") for ERF static fields

`erftools.geog` builds a **geo_em-compatible NetCDF** (terrain, land use, soil, vegetation and
the other static fields) for an ERF/WRF-style Lambert-conformal C-grid, directly from the
raw WRF-SIG `WPS_GEOG_V4` tiles. It does not call WPS, WRF or any Fortran at run time, and
the only WPS-related files it reads are the dataset tiles and their `index` files.
Every processing step is a selectable option, set per field. There are two presets:

| preset | intent | example output files |
|---|---|---|
| `wps_equivalent` | Python re-implementation of geogrid v4.4, with the algorithms exactly as `GEOGRID.TBL.ARW.noahmp` specifies them. It is the correctness test of the implementations. | `geo_em.d01.optionB_wps_modis.nc`, `geo_em.d01.optionB_wps_nlcd.nc` |
| `slope_limited` | Area-weighted sampling with a slope-limited terrain filter, chosen for fine ERF grids (justified below). | `geo_em.d01.optionB_modis.nc`, `geo_em.d01.optionB_nlcd.nc` |

`modis` means MODIS 20-class + lakes (geogrid res `default`). `nlcd` means NLCD 2011 9"
(`nlcd2011_9s+default`). Only LANDUSEF/LU_INDEX/LANDMASK change between the two.

## Quick start (serial, about 15 s per file)

From the erftools checkout (or with erftools installed):

```bash
python -m erftools.geog --preset slope_limited --landuse nlcd --out /path/geo_em.d01.nc
python -m erftools.geog --preset wps_equivalent --grid /path/namelist.wps --out ...   # another grid
python tests/geog/validate.py OUTDIR       # acceptance + cross-check tables (needs the 4 files in OUTDIR)
python tests/geog/terrain_filters.py       # terrain method x filter study
pytest tests/geog                          # or: python tests/geog/test_units.py / test_presets_data.py
```

The WPS_GEOG_V4 root comes from the environment variable `WPS_GEOG_PATH` or from
`--data-root` (there is no default). The tests that need the data or the geogrid
reference files (`ERFTOOLS_GEOG_REF_DIR`, holding `geo_em.d01.modis_default.nc` and
`geo_em.d01.nlcd2011_9s.nc`) skip when these are absent.

Dependencies: numpy, scipy (ndimage), pyproj and netCDF4 (pyproj is in erftools' `grib` extra).

## Running for another grid
The grid uses namelist.wps names. Pass them with `--grid namelist.wps` (or `--namelist`), or
`--grid spec.json` with a GridSpec dict such as an earlier output's `ERFTOOLS_GEOG_GRID_SPEC` (it reads e_we,
e_sn, dx, dy, ref_lat, ref_lon, truelat1, truelat2, stand_lon, ref_x, ref_y and map_proj) or
with flags such as `--e_we 402 --dx 250 ...`. The default grid is:
- lambert projection, truelat1 = truelat2 = 38.5, stand_lon = -97.5, sphere R = 6 370 000 m
- ref (centre) point 38.9562, -106.9880
- dx = dy = 500 m, e_we = e_sn = 202 (201 x 201 mass points)

In Python:

```python
from erftools.geog import presets; from erftools.geog.grid import GridSpec
from erftools.geog.pipeline import run; from erftools.geog.writer import write_geo_em
p = presets.get("slope_limited")
p["fields"][0]["filter"]["max_slope"] = 0.4          # any option can be edited
grid, outs, attrs = run(p, "modis", GridSpec(e_we=302, e_sn=302, dx=333.3))
write_geo_em("geo_em.d01.nc", grid, outs, attrs)
```

`--preset` also takes a JSON file with the same structure as `erftools/geog/presets.py`. Only
Lambert conformal is implemented. Other `map_proj` values raise `NotImplementedError`.

## Design

| module | role |
|---|---|
| `grid.py` | Grid and projection. Two back ends: `wps_f32` is a transcription of WPS `module_map_utils` (set_lc/ijll_lc/llij_lc), get_map_factor, get_rotang and Coriolis, evaluated in REAL*4. `pyproj` is pyproj LCC on the sphere in float64, with map factors from `get_factors` and the rotation from the cone convergence n(λ-λ0). Fields cover the domain plus a 3-point halo (geogrid's memory region). |
| `dataset.py` | `GeogridTiles` reads the tile bytes directly, driven by `index` keys: projection, known_*, dx/dy, tile_x/y/z, tile_z_start/end, tile_bdr, wordsize, signed, endian, row_order, scale_factor, missing_value, type, category_min/max, units and description. It provides memory-mapped sub-window reads, so the 1.8 GB NLCD tile is never read whole, and an index sha256 for provenance. `open_source()` is the factory for adding new readers (see below). |
| `f32.py` | Single-precision helpers. numpy float32 gives + - * / sqrt. glibc's `sinf/cosf/tanf/atanf/atan2f/powf/log10f` are called through ctypes, which are the same functions gfortran's REAL*4 intrinsics call. |
| `wps_interp.py` | Vectorised `four_pt`, `sixteen_pt`, `average_4pt`, `average_16pt`, `wt_average_4pt/16pt`, `nearest_neighbor` and `search(depth)`, with geogrid's fall-through chain semantics and edge clamping. |
| `wps_engine.py` | Ports of geogrid's `calc_field`: `average_gcell` / categorical accumulation (each source pixel goes to its nearest model point), the interpolator chain for points with no pixel, `masked=water/land` with `fill_missing`, scale factors, fractions, LANDMASK and the dominant-category rules. |
| `methods.py` | Our methods in float64: `nearest`, `bilinear` (missing-aware), `bicubic`, `cell_bilinear` (cell average of the bilinear surface on an N x N sub-grid), `area_mean` (area-weighted mean over the overlapping source cells), `area_fraction` (area fraction of each category) and `nearest_valid` gap fill. |
| `filters.py` | WPS `smth_desmth_special`, `smth_desmth` and `one_two_one`, plus `none`, `gaussian(sigma_m)`, `shapiro(order, passes)` and `slope_limited(max_slope, metric, max_curvature)`. It also has the slope and curvature diagnostics. |
| `presets.py` | The two presets as plain dicts. Each field has its source, method, interp chain, opts, mask, fill, filter and extras. |
| `pipeline.py` / `writer.py` / `cli.py` | `pipeline.py` runs a preset (LANDUSEF first, because it defines LANDMASK). `writer.py` writes NETCDF3 64-bit-offset files with geogrid's dims, variable attributes and global attributes, plus provenance. `cli.py` is the command line. |

**Provenance**, as global attributes on every output:
- `ERFTOOLS_GEOG_VERSION` (0.1.0) and `ERFTOOLS_GEOG_CODE_SHA256` (sha256 over the package `.py` files)
- `ERFTOOLS_GEOG_PRESET` and `ERFTOOLS_GEOG_PRESET_JSON` (the full option set)
- `ERFTOOLS_GEOG_LANDUSE`, `ERFTOOLS_GEOG_GRID_METHOD` and `ERFTOOLS_GEOG_GRID_SPEC`
- `ERFTOOLS_GEOG_SOURCES_JSON`: per field, the dataset path, index sha256, method, interp, opts, filter, and filter iterations
- `ERFTOOLS_GEOG_RUNTIME_S`

**Adding a source** (for example a USGS 3DEP GeoTIFF): implement a class with `meta`, `type`,
`cat_min/cat_max`, `z0/z1/nz`, `missing_value`, `scale_factor`, `units`, `description`,
`index_sha256`, `ll_to_xy(lat, lon, f32=False)`, `xy_to_ll` and `window(x0, x1, y0, y1)`.
Then dispatch to it in `dataset.open_source()`. The non-WPS methods use only this interface.
The `wps` method also needs `tile_of`, `read_tile` and the REAL*4 `ll_to_xy`.

**GeoTIFF sources** (`geotiff.GeoTiffMosaic`): `open_source()` picks this reader for a directory
with no `index` file that holds `*.tif` tiles.
- **Accepted tiles:** single-band float32 GeoTIFFs in a geographic CRS whose pixels lie on one 1/N-degree
  lattice, for example USGS 3DEP 1″ `USGS_1_n39w107.tif`.
- **Decoding:** Pillow, which is already in the environment, with `tifffile` as the fallback.
- **Overlaps:** tile overlaps must agree; `check_overlaps()` reports any disagreement.
- **Methods:** non-WPS only.
- **Preset:** `slope_limited_3dep` is `slope_limited` with HGT_M = `area_mean` (16 × 16 sub-grid) of 3DEP 1″, and the
  same slope-limited filter. Tiles live in the directory named by `ERFTOOLS_GEOG_3DEP_PATH`
  (no default).
- **Datums:** 3DEP's horizontal datum is NAD83 and its vertical datum NAVD88. NAD83 is used as WGS84
  (< 2 m apart here).

## Per-field algorithms

### `wps_equivalent` (transcribed from GEOGRID.TBL `default` entries, run in REAL*4)

| field | source | geogrid processing as reproduced |
|---|---|---|
| LANDUSEF / LU_INDEX / LANDMASK (modis) | modis_landuse_20class_30s_with_lakes | Categorical accumulation: each 30" pixel is assigned to its nearest model point (nint of the REAL*4 LCC coordinates). Cells that get no pixel take `nearest_neighbor`. Then fractions. LANDMASK = 1 if water (17+21) < 0.5. LU_INDEX is the largest non-water category on land and the largest water category on water. |
| LANDUSEF (nlcd) | nlcd2011_ll_9s (40 fraction levels) | `average_gcell(0.0)`: the threshold 0 x src_dx x 111 <= dx_km is met, so each 9" pixel's 40 values are summed into its nearest model point, averaged, scaled by 0.1 and normalised. Water is category 17. |
| HGT_M | topo_gmted2010_30s | `average_gcell(4.0)` is inactive because 4 x 0.00833° x 111 = 3.7 km > 0.5 km, so the chain is `four_pt` then `average_4pt`. Then `smth_desmth_special`, 1 pass, on the haloed memory array. |
| SOILCTOP/SCT_DOM, SOILCBOT/SCB_DOM | soiltype_top/bot_30s | As for MODIS land use. Dominant = first maximum fraction. |
| SOILTEMP | soiltemp_1deg | `sixteen_pt` → `four_pt` → `average_4pt` → `average_16pt` → `search`. Masked over water (fill 0). Missing value 0, scale 0.01. |
| ALBEDO12M, SNOALB | albedo_modis, maxsnowalb_modis (0.05°, dy < 0) | `four_pt` → `average_4pt` → `average_16pt` → `search`. Masked over water (fill 8 / 0). |
| GREENFRAC | greenfrac_fpar_modis (30") | `average_gcell(4.0)` inactive, then the same chain as ALBEDO12M. Masked over water. |
| LAI12M | lai_modis_10m (the TBL default; 30" needs res `modis_lai`) | Same chain as GREENFRAC. |
| VAR_SSO | varsso_10m | `four_pt` → `average_4pt`. |
| CON, VAR, OA1-4, OL1-4; *LS, *SS | orogwd_10m, orogwd3_10m | `average_4pt`, masked over water. |
| CROPTYPE, SEASON_GDD, PLANTING, HARVEST, FDEPTH, RECHCLIM, EQZWT, RIVERBED | crop/*, groundwater/* | As in the TBL. |

geogrid details that are reproduced:
- Its float32 operation order and the 3-point halo, which the smoother reads.
- The signed-word rule `ival > 2^(n-1)` from `read_geogrid.c`.
- Scaling is skipped for values that equal the fill value.
- `search` increments the BFS depth once per queued neighbour.
- The `wt_average_4pt` y-clamp typo (`ifx < start_y`).
- `sixteen_pt` replaces zeros with 1e-20.
- The halo-edge `rotang` is one-sided.

There is one documented deviation, for domains that straddle a source-tile boundary.
geogrid's result for a few cells next to that boundary depends on the order it visits tiles.
Here all tiles are accumulated first.

### `slope_limited` (our choice, float64, no geogrid code paths)

| field | method | why |
|---|---|---|
| grid | pyproj LCC on the 6370 km sphere | The same projection definition, independently implemented. Map factors are analytic, from pyproj. SINALPHA/COSALPHA come analytically from the cone convergence, -n(λ-λ0). |
| HGT_M | `cell_bilinear` (8x8 sub-grid) of GMTED2010 30", then `slope_limited(max_slope=0.5)` | The 30" source (~720 x 925 m) is coarser than the 500 m cells. Conservative aggregation (`area_mean`) of a coarser source gives stair-steps (max slope 0.77, see the table below). The cell average of the bilinear surface is the conservative remap of a continuous reconstruction. The filter (next paragraph) is conservative and selective, and is designed for ERF's terrain-following mesh. |
| LANDUSEF / LU_INDEX / LANDMASK | `area_fraction` (8x8 sub-grid per cell; for NLCD, the area-weighted mean of the 40 stored fractions) | These are true cell-area fractions, kept in LANDUSEF. LANDMASK = 1 if water < 0.5. LU_INDEX uses the WPS land/water-consistent rule. |
| SOILCTOP/BOT, SCT/SCB_DOM | `area_fraction` | Dominant category consistent with LANDMASK: on land, the largest non-water class (14 only if nothing else is present); on water, 14. |
| SOILTEMP | `bilinear` with nearest-valid fill | The 1° source is far coarser than the grid, so smooth interpolation is used. It is not masked: lake cells get valid values. |
| ALBEDO12M, SNOALB | `cell_bilinear` with nearest-valid fill | The 0.05° (5.5 km) source is more than 2x the cell size, so smooth interpolation is used. |
| GREENFRAC, LAI12M | `area_mean` with nearest-valid fill. LAI uses **lai_modis_30s**, not the 10' default. | The source is <= 2x the cell, so conservative aggregation is used. The finest LAI on disk is 30". |
| VAR_SSO | `area_mean` of **varsso (30")** | This is the sub-grid variance at the scale closest to the 500 m grid. The 10' product describes 18 km cells. |
| extras | `HGT_M_RAW` (before the filter), `HGT_M_STD` (std of the source surface within the cell; a lower bound, because the source is coarser than the cell), `SLOPE` (\|∇h\|, m/m) | |

The GWD, crop and groundwater fields are not in `slope_limited`; add entries with any method if they are needed.
No water masking is applied, so every cell has a physical value. Set `masked: "water"` on a
field to get geogrid-style fill values instead.

**Terrain filter design (`slope_limited`).** This is a flux-form nonlinear diffusion.
- A cell is flagged when its centred, map-factor-scaled |∇h| exceeds `max_slope`. Optionally a
  cell is also flagged when its grid-scale curvature |δ²h|/Δx exceeds `max_curvature`.
- Every face that touches a flagged cell carries the flux κ(h_R - h_L), with κ = 0.2 (stable and monotone).
- No flux crosses the domain boundary. The filter iterates until no cell is flagged.

Its properties:
- The domain-mean elevation is conserved to round-off.
- Terrain that already satisfies the criterion is left bit-for-bit unchanged.
- The maximum slope of the terrain-following coordinate is set directly. ERF's metric terms
  and time step depend on that maximum slope, not on the rms smoothness.

The default of 0.5 (26.6°) changes HGT_M by 2.9 m rms and reaches the limit exactly. The
largest local changes, up to 112 m, are on a few ridge and cliff cells. Note that geogrid's
desmoother *raises* the maximum face slope (0.64 → 0.66) while it lowers |∇h| (0.639 → 0.617).

## Validation (`tests/geog/validate.py`; full output in `optionB/validation_output.txt`)

### 1. `wps_equivalent` against the geogrid reference: **PASS, bit-identical**

| land use | variables compared | exact match | max \|diff\| |
|---|---|---|---|
| modis (`geo_em.d01.modis_default.nc`) | 79 | 79/79 at 100.000 % | 0 |
| nlcd (`geo_em.d01.nlcd2011_9s.nc`) | 79 | 79/79 at 100.000 % | 0 |

The 79 variables are every variable geogrid wrote:
- grid: XLAT/XLONG_M/U/V/C, CLAT/CLONG, MAPFAC_*, E, F, SINALPHA/COSALPHA (M/U/V)
- land and soil: LANDMASK, LANDUSEF, LU_INDEX, SOILCTOP/SCT_DOM, SOILCBOT/SCB_DOM
- surface: HGT_M, SOILTEMP, ALBEDO12M, GREENFRAC, LAI12M, SNOALB, VAR_SSO
- GWD: the 10 orogwd fields and the 20 orogwd3 LS/SS fields
- CROPTYPE, SEASON_GDD, PLANTING, HARVEST, FDEPTH, RECHCLIM, EQZWT, RIVERBED

The key global attributes also agree exactly: CEN_LAT/LON, corner_lats/lons, MMINLU,
NUM_LAND_CAT, ISWATER, ISLAKE, ISICE, ISURBAN, ISOILWATER and the FLAGs. The target was
categorical 100 %, continuous within float32 rounding, and HGT_M within 0.01 m; every field
reaches max |diff| = 0.

### 2. `slope_limited` against geogrid: a cross-check, with differences explained (no tuning)

| quantity | modis | nlcd | explanation |
|---|---|---|---|
| XLAT/XLONG (M, U, V, C) | max 1.5e-5° (1 ulp) | same | float64 projection rounded once vs. a REAL*4 chain |
| MAPFAC M/U/V | 2.4e-7 | same | float rounding |
| F, E | rel 3.2e-7 | same | float rounding |
| SINALPHA / COSALPHA | 1.9e-3 / 2.1e-4 | same | geogrid's are centred differences of REAL*4 lat/lon over 1 km (Δλ ≈ 0.009°, ulp 7.6e-6°), so they carry ~1 % noise. The same finite-difference formula on float64 lat/lon agrees with our analytic value to 5e-10. |
| HGT_M mine − ref | mean −0.02, rms 7.09, max 119.5 m | same | Before our filter: rms 6.5, max 39 m, from cell averaging vs. point bilinear plus the WPS smoother. The filter adds rms 2.9 m, at the steepest cells only. |
| max slope | 0.500 (26.5°) vs 0.617 (31.7°) | same | by design |
| LANDMASK agreement | 99.97 % (13 cells) | 99.98 % (9 cells) | lake-edge cells near 50 % water |
| LU_INDEX agreement | 97.8 % (906 cells) | 95.2 % (1940 cells) | geogrid samples a few nearest pixels per cell (30": 0–2 pixels, the rest take a nearest neighbour; 9": ~4–6 pixels); ours is the exact area overlap. In the differing cells our top fraction has median 0.50 and the second has 0.44, so these cells are genuinely mixed (mostly 7↔10, 10→1, 5→1). Domain-mean class fractions agree to 0.001 / 0.0001. |
| SCT_DOM / SCB_DOM | 99.47 % / 99.11 % | 99.57 % / 99.21 % | Area weighting, plus our LANDMASK-consistent rule. The reference has 61 water cells with non-water soil and 21 land cells with soil 14; ours has 0 and 7 (7 = land cells whose soil map is entirely water). |
| SOILTEMP (land) | mean +0.047 K, rms 0.050 K | same | bilinear vs. sixteen_pt on 1° data |
| ALBEDO12M / SNOALB (land) | rms 0.006 / 0.04 % | same | cell-averaged bilinear vs. four_pt |
| GREENFRAC (land) | rms 0.018 (4 %) | same | 30" area mean vs. four_pt |
| LAI12M (land) | mean +0.06, rms 0.50 (57 %) | same | **Different source on purpose:** 30" LAI vs. the TBL default 10' (18 km) product |
| VAR_SSO | ours is ~6 % of ref (mean 4.6e3 vs 8.2e4 m²) | same | **Different source on purpose:** variance at the 30" scale vs. the 10' scale |

Water cells (about 0.2 % of the domain) are excluded from the continuous statistics,
because geogrid writes fill values there.

### Terrain options (`tests/geog/terrain_filters.py`; full table in `optionB/terrain_options.txt`)

Slope is the centred |∇h|·m (m/m). "chg" columns compare with the unfiltered field. "vs geogrid"
is the rms difference from the reference HGT_M.

| horizontal | filter | max slope (deg) | rms / max chg (m) | rms vs geogrid |
|---|---|---|---|---|
| wps four_pt | none | 0.639 (32.6) | 0 / 0 | 9.5 |
| wps four_pt | smth_desmth_special x1 (= geogrid) | 0.617 (31.7) | 9.5 / 64 | 0 |
| wps four_pt | smth_desmth_special x3 | 0.584 (30.3) | 15.6 / 101 | 7.4 |
| wps four_pt | shapiro(4) x1 | 0.614 (31.6) | 10.1 / 76 | 2.6 |
| wps four_pt | gaussian 500 m | 0.478 (25.5) | 31.5 / 162 | 26.7 |
| wps four_pt | slope_limited 0.5 | 0.499 (26.5) | 3.6 / 126 | 10.0 |
| bicubic | none | 0.766 (37.4) | 0 / 0 | 23.0 |
| area_mean (conservative) | none | 0.767 (37.5) | 0 / 0 | 26.2 |
| cell_bilinear | none | 0.628 (32.1) | 0 / 0 | 6.5 |
| cell_bilinear | smth_desmth_special x1 | 0.607 (31.3) | 7.2 / 48 | 4.0 |
| **cell_bilinear** | **slope_limited 0.5 (slope_limited)** | **0.500 (26.5)** | **2.9 / 112** | 7.1 |
| cell_bilinear | slope_limited 0.4 | 0.400 (21.8) | 8.7 / 174 | 10.8 |
| cell_bilinear | slope_limited 0.3 | 0.300 (16.7) | 20.9 / 358 | 21.8 |
| cell_bilinear | slope_limited 0.5 + curvature 0.4 | 0.500 (26.5) | 7.7 / 130 | 9.0 |


## Runtime (login node, serial)
- `slope_limited`: 12 s (modis) and 14 s (nlcd)
- `wps_equivalent`: 4–6 s
- all four files plus both validation scripts: about 50 s

Files are built in memory and written to CFS with a single write. With netCDF's small writes
CFS took about 4 min per file, and `diskless/persist` does not persist with netcdf-c 4.10.1.

## Limitations
- Only Lambert conformal is implemented. The other WPS projections are not ported.
- Only regular_ll sources are implemented. That covers all of WPS_GEOG_V4 used here.
- Single priority level per field. geogrid's multi-priority fallback (with the "half-area"
  rule) is not implemented; no field in this TBL uses it.
- The multi-tile ordering deviation in the `wps` engine described above.
- No subgrid (sr_x/sr_y > 1) fields.
- No HGT_U/HGT_V. geogrid's ARW TBL does not produce them, and neither does the reference.
  ERF derives nodal and face terrain from HGT_M.
- Bit-identity depends on glibc's float libm (ctypes). On another platform, results are expected to agree to ~1 ulp.
- `HGT_M_STD` cannot see sub-30" variability. A finer DEM is needed for a real sub-grid statistic.
- The `slope_limited` choices are defensible but not tuned. In particular, max_slope = 0.5 should
  be confirmed against ERF stability tests at the planned dt.
- LANDUSEF from NLCD 2011 9" is itself a pre-aggregated 40-class fraction product. It is not the native 30 m NLCD.

## Better original-source datasets (not downloaded; network is needed)

| field | on disk | better source | what downloading needs |
|---|---|---|---|
| terrain | GMTED2010 30" (~0.9 km; coarser than a 500 m grid) | USGS 3DEP 1" (~30 m) or 1/3" (~10 m) DEM: 1°x1° GeoTIFF tiles covering the domain (~50 MB each at 1", ~450 MB at 1/3"). Copernicus GLO-30 is a global alternative. | Public HTTPS/S3 (USGS TNM `prd-tnm` bucket; Copernicus on AWS open data). Transfer on a NERSC DTN. A GeoTIFF reader (needs `rasterio` or `tifffile` in the environment; neither is installed now) plugged into `open_source()`. Then `area_mean` becomes the right terrain method, and `HGT_M_STD` becomes meaningful. |
| land use | NLCD 2011 9" fractions; MODIS 20-class 30" | Native 30 m NLCD 2021 / Annual NLCD (2023) or the 2019 land cover (MRLC, CONUS Albers); MODIS MCD12Q1 v061 at 500 m. | MRLC is a public download. MCD12Q1 needs a NASA Earthdata login (LP DAAC). An Albers / sinusoidal reader via pyproj. A 30 m→40-class (NLCD40) category map. |
| soil | STATSGO-derived 16-class 30" | gNATSGO / gSSURGO (10–30 m, USDA NRCS), POLARIS 30 m, or SoilGrids 250 m (sand/clay → USDA texture class). | USDA Box / NRCS download, or the ISRIC WCS; then a texture-class mapping. |
| LAI / FPAR / albedo | MODIS climatologies, 30" / 0.05° | MCD15A2H v061 (500 m, 8-day) and MCD43A3 (500 m, daily) climatologies for the years needed; VIIRS VNP15/VNP43. | Earthdata login; build a monthly climatology. |
| deep soil temperature | 1° annual-mean (no elevation adjustment) | A climatological deep soil or annual 2 m temperature from ERA5-Land (9 km), or PRISM 800 m annual Tmean, lapse-rate-adjusted to HGT_M. | CDS API account (ERA5-Land); PRISM is public. |

## Open items
1. Decide which preset the ERF workflow uses (`slope_limited` is recommended) and which
   `max_slope` to use, after an ERF stability and dt check.
2. The output feeds the existing ingestion in `erftools.preprocessing.terrain_remap`
   (`read_geogrid`, `remap_snapshot`, `map_static_fields`), which accepts any geo_em-format file.
3. Optionally: a GeoTIFF reader for 3DEP / NLCD-30 m, and YAML preset files in place of Python dicts.
