# ERF Tools
A collection of Python-based modules and scripts for facilitating the usage of ERF.

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

### Converting a WRF namelist into ERF inputs
```python
from erftools.preprocessing import WRFInputDeck
wrf = WRFInputDeck('namelist.input')
wrf.process_initial_conditions('wrfinput_d01',
                               landuse_table_path='/Users/equon/WRF/run/LANDUSE.TBL',
                               write_hgt='terrain_height.txt',
                               write_z0='roughness_height.txt')
wrf.write_inputfile('inputs')
```

### Postprocessing data logs
Data logs are output with the `erf.data_log` param and can include time histories of surface conditions and planar averaged profiles (e.g., for idealized LES simulations)
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
