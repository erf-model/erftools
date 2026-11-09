# ERF Tools
A collection of Python-based modules and scripts for facilitating the usage of
the Energy Research and Forecasting (ERF) model.

## Examples

Some short snippets are provided below with commonly used workflows also having
command line analogues to facilitate execution. These console scripts are immediately
available in your environment after you [install](#sec-installation) the erftools
package. Note that each script includes a `--help` option, e.g.,
```shell
wrf_namelist_to_erf --help
```

Please see the notebooks folder for more detailed examples.

### Converting a WRF namelist into ERF inputs
```python
from erftools.preprocessing import WRFInputDeck

wrf = WRFInputDeck('namelist.input')
wrf.write_inputfile('inputs')

# optional: extract data from wrfinput
wrf.process_initial_conditions('wrfinput_d01',
                               write_hgt='terrain_height.txt',
                               write_z0='roughness_height.txt')
```

The namelist conversion may also be accomplished from the command line:
```shell
wrf_namelist_to_erf namelist.input inputs
```

### Plotting a Nested Grid Configuration
```python
from erftools.grids import LambertConformalGrid

now23 = LambertConformalGrid(
    ref_lat=37.100163,
    ref_lon=-122.49992,
    truelat1=37.100163, # standard parallel
    stand_lon=-122.49992, # standard longitude
    dx=[6000, 2000],
    dy=[6000, 2000],
    nx=[243, 552], # WRF staggered dim - 1
    ny=[243, 579],
    ll_ij=[(30,27)], # parent grid index
)
fig, ax = now23.plot_grids()
```

From the command line, the grid geometry can be read from an ERF input file and plotted:
```shell
plotgrids inputs --latlon0 35.85 -123.72 --truelat1 36.05 --standlon -65.0
```

### Static fields without WPS (geo_em from WPS_GEOG tiles)
`erftools.geog` builds a geo_em-compatible file (terrain, land use, soil, vegetation)
for a Lambert-conformal grid directly from the WPS_GEOG_V4 tiles. The preset
`wps_equivalent` reproduces geogrid; `slope_limited` is a conservative, slope-limited
alternative. See `erftools/geog/README.md`.
```shell
python -m erftools.geog --preset slope_limited --landuse modis --grid namelist.wps --out geo_em.d01.nc
```
Tests: `pytest tests/geog` (data-dependent tests skip unless `WPS_GEOG_PATH` exists).

### Land and snow initial state for Noah-MP (HRRR + geo_em)
ERF's Noah-MP driver reads its initial land state itself, from the file that
`ERF_SETUP_FILE_01` names in `namelist.erf`; ERF passes it nothing. An atmosphere-only
wrfinput (HRRR `to_wrfinput`, `terrain_remap`) cannot start it.
`erftools.preprocessing.noahmp_input.add_noahmp_land_state` adds what WRF real.exe would:
```python
from erftools.preprocessing import hrrr_land, noahmp_input, terrain_remap
geo  = terrain_remap.read_geogrid('geo_em.d01.nc')
# valid at the start time: surface and snow fields, and HRRR's RUC soil levels
land = hrrr_land.read_hrrr_land(['hrrr.t00z.wrfnatf00.grib2', 'hrrr.t00z.wrfprsf00.grib2'])
inp, record = noahmp_input.add_noahmp_land_state(inp, geo, land)  # inp: wrfinput Dataset
# options for a target terrain much finer than HRRR's (off by default):
#   adjust_tsk=True, adjust_soil_temp=True, snow_method='elevation' (+ radius, max_gradient, cap)
print(noahmp_input.check_noahmp_inputs(inp))                     # {'ok': True, ...}
```

| Field | Source | Method |
|---|---|---|
| VEGFRA, LAI, ALBBCK | geo_em GREENFRAC, LAI12M, ALBEDO12M | monthly values valid on the 15th, linear in day to the start date (real.exe `monthly_interp_to_date`); VEGFRA in %, ALBBCK percent / 100 |
| SHDMIN, SHDMAX | GREENFRAC | 100 x min / max over the 12 months |
| SNOALB | SNOALB | percent / 100; water 0.08 (as ALBBCK) |
| TMN | SOILTEMP, HGT_M | SOILTEMP - 0.0065 HGT_M on land, SST on water |
| TSLB, SMOIS, SH2O, ZS, DZS | HRRR RUC soil levels (TSOIL, SOILW, SOILL) | land-only bilinear, linear in depth to the layer centres (default 0-10, 10-40, 40-100, 100-200 cm); option `adjust_soil_temp=True`: TSLB + 0.0065 (HGT_HRRR - HGT); water: TSLB = SST, SMOIS = SH2O = 1; SH2O = SMOIS when the file has no SOILL |
| SNOW, SNOWH | HRRR WEASD, SNOD | land-only bilinear (default; option `snow_method='elevation'`, below); SNOWH = SNOW / HRRR's local density (200 kg m-3 if HRRR has no depth) |
| SNOWC | HRRR SNOWC | HRRR's cover fraction, land-only bilinear, 0 where SNOW = 0 (default); `snowc='threshold'`: 1 where SNOW >= 10 kg m-2 (real.exe) |
| CANWAT | HRRR CNWAT | land-only bilinear (Noah-MP reads it, then starts from 0) |
| TSK | as given | option `adjust_tsk=True`: land-only HRRR skin temperature + 0.0065 (HGT_HRRR - HGT) on land |
| XLAND, SEAICE, GRID_ID | LANDMASK, HRRR ICEC | XLAND = 2 - LANDMASK (without it Noah-MP treats water as land) |

"Land-only" means HRRR water points never enter a land value (and the reverse): their
bilinear weights are dropped and the rest renormalized, and a point with no land corner
takes the nearest HRRR land point.

Defaults and options. The defaults are plain: HRRR's soil temperature and the given TSK
are not changed, snow is interpolated (land-only), and SNOWC is HRRR's own cover. Over a
target terrain much finer than HRRR's 3 km one these can put HRRR's values at the wrong
heights; the options below address that, are off by default, and are recorded with every
other option in the `LAND_STATE_OPTIONS` global attribute (JSON).
- `adjust_tsk=True`, `adjust_soil_temp=True`: TSK (on land) and TSLB + 0.0065 (HGT_HRRR -
  HGT), as WRF real.exe adjust_soil_temp_new (per WRF documentation, not verified against
  source here). `lapse_rate` sets the 0.0065 K/m.
- `snow_method='elevation'`: fits, at every HRRR land point, the weighted least-squares
  slope of HRRR's own SWE against HRRR's terrain over a neighbourhood and applies it to the
  height difference, SWE = SWE_i + b (HGT - HGT_i), clipped to >= 0. Where HRRR's SWE is
  linear in its terrain this is exact; across a snowline the linear fit smears snow a
  little below it. Its parameters are tunable and case-dependent: `radius` (neighbourhood,
  m; Gaussian weights), `min_points` and `min_relief` (no slope below them), `max_gradient`
  (slope clip, kg m-2 per m) and `cap='local_max'` (SWE bounded by the largest HRRR SWE
  within the radius; `cap=None` for none). The defaults (9000 m, 6, 50 m, 1 kg m-2 per m,
  'local_max') are example values for a 3 km source and a 500 m target, not
  recommendations.
- `snowc='threshold'`: real.exe's SNOWC = 1 where SNOW >= 10 kg m-2, instead of HRRR's cover.

Which HRRR file. In 2021 HRRR the full RUC soil column (TSOIL and SOILW at 0, 1, 4, 10, 30,
60, 100, 160 and 300 cm; no SOILL) is in `wrfprsf00`, and `wrfsfcf00` has only MSTAV.
`read_hrrr_land` takes a list of files: surface fields from the first file that has them,
each soil variable (TSOIL / SOILW / SOILL, by GRIB name) from the file with the most levels.
`wrfnatf00` has the snow fields (WEASD, SNOD, SNOWC), canopy water and skin temperature,
but soil moisture only at 0 and 1 cm and no soil temperature. Without a soil file `soil_state_from_hrrr` raises, unless
`soil_fallback='skin_tmn'` is passed (TSLB linear between the lapse-adjusted skin temperature
and TMN at 8 m; the 1 cm moisture held below). That fallback is for testing only.

The required list in `check_noahmp_inputs` is that of the Noah-MP reader
(`Submodules/Noah-MP/drivers/erf/NoahmpReadLandMod.F90`): the attributes WEST-EAST /
SOUTH-NORTH_GRID_DIMENSION, DX, DY, TRUELAT1/2, STAND_LON, MAP_PROJ, GRID_ID, ISWATER, ISURBAN,
ISICE, MMINLU, SIMULATION_START_DATE, I/J_PARENT_START, PARENT_GRID_RATIO; the fields XLAT,
XLONG, HGT, TMN, IVGTYP, ISLTYP, CANWAT, TSK, SNOW, SNOWC, SHDMIN, SHDMAX, TSLB, SMOIS and DZS.
The namelist's `soil_thick_input` should match DZS. The real.exe behaviour above is as
documented for WRF; there is no WRF source in this repository to check it against.
Tests: `python tests/preprocessing/test_*.py` (`ERFTOOLS_HRRR_GRIB`, `ERFTOOLS_HRRR_SOIL_GRIB`
and `ERFTOOLS_WRFINPUT` enable the tests on real files).

### Postprocessing Data Logs
Data logs are output with the `erf.data_log` param and can include time
histories of surface conditions and planar averaged profiles (e.g., for
idealized LES simulations)

```python
from erftools.postprocessing import DataLog

log = DataLog(f'{simdir}/surf_hist.dat',
              f'{simdir}/mean_profiles.dat',
              f'{simdir}/flux_profiles.dat',
              f'{simdir}/sfs_profiles.dat')

log.calc_stress()
log.est_abl_height('max_theta_grad')

print(log.ds) # data are stored in an xarray dataset
```


<a name="sec-installation"></a>
## Installation

Clone and create a conda (or mamba) environment:
```shell
git clone https://github.com/erf-model/erftools.git
cd erftools
conda create -n erftools python=3.9
conda activate erftools
```

Then, install with:
```shell
pip install -e . # editable install
```
or install with all optional dependencies:
```shell
pip install -e .[all]
```


## Contributing

Some notes and recommendations:

* An aspirational goal is to contribute code that can be used as in the examples above, with clear, intuitive naming.
* To avoid duplication, model constants are defined in `erftools.constants`, which should replicate `ERF/Source/ERF_Constants.H`.
* In the same vein, equation of state evaluations are defined in `erftools.utils.EOS`, which should replicate `ERF/Source/Utils/ERF_EOS.H`.
* Other utilities for calculating/deriving/diagnosing quantities of interest are also in `erftools.utils.*`
* Please follow PEP-8 style--as a set of guidelines rather than gospel--to facilitate code usage and maintenance by the community.
