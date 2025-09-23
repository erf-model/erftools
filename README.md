# ERF Tools
A collection of Python-based modules and scripts for facilitating the usage of
the Energy Research and Forecasting (ERF) model.

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

## Examples

Some short snippets are provided below.
Please see the notebooks folder for more detailed examples.

### Converting a WRF namelist into ERF inputs
```python
from erftools.preprocessing import WRFInputDeck

wrf = WRFInputDeck('namelist.input')
wrf.write_inputfile('inputs')

# optional: extract data from wrfinput
wrf.process_initial_conditions('wrfinput_d01',
                               landuse_table_path='/Users/equon/WRF/run/LANDUSE.TBL',
                               write_hgt='terrain_height.txt',
                               write_z0='roughness_height.txt')
```

The namelist conversion may also be accomplished from the command line:
```shell
wrf_namelist_to_erf namelist.input inputs
```

### Plotting a Domain Configuration
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

Grid geometry can be read from an ERF input file and plotting may be
done from the command line as well.
```shell
plotgrids inputs --latlon0 35.85 -123.72 --truelat1 36.05 --standlon -65.0
```


### Postprocessing data logs
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

## Contributing

Some notes and recommendations:

* An aspirational goal is to contribute code that can be used as in the examples above, with clear, intuitive naming.
* To avoid duplication, model constants are defined in `erftools.constants`, which should replicate `ERF/Source/ERF_Constants.H`.
* In the same vein, equation of state evaluations are defined in `erftools.utils.EOS`, which should replicate `ERF/Source/Utils/ERF_EOS.H`.
* Other utilities for calculating/deriving/diagnosing quantities of interest are also in `erftools.utils.*`
* Please follow PEP-8 style--as a set of guidelines rather than gospel--to facilitate code usage and maintenance by the community.
