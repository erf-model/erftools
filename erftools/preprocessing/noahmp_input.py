"""A wrfinput that ERF's Noah-MP driver can start from: the static LSM fields
(terrain_remap.add_geogrid_land_fields), the soil state and the snow from HRRR (hrrr_land),
XLAND / SEAICE and the GRID_ID attribute; and a checker for the fields Noah-MP reads.

Noah-MP reads its initial state itself from the file ERF_SETUP_FILE_01 of namelist.erf names
(Source/LandSurfaceModel/Noah-MP/ERF_NOAHMP_Init.cpp); ERF passes it nothing. The lists below
are those of the reader, Submodules/Noah-MP/drivers/erf/NoahmpReadLandMod.F90 (NRL) and the
initialization NoahmpInitMainMod.F90 (NIM), at Noah-MP fbaa38f.
"""
import inspect
import json

import numpy as np
import pandas as pd
import xarray as xr

from . import hrrr_land as HL
from . import terrain_remap as TR

# global attributes read with error_handler (abort if absent), NRL:43-95 and :159-166;
# GRID_ID may also be spelled grid_id (NRL:69-73); ISLAKE is optional (NRL:78-82)
NOAHMP_REQUIRED_ATTRS = ('WEST-EAST_GRID_DIMENSION', 'SOUTH-NORTH_GRID_DIMENSION', 'DX', 'DY',
                         'TRUELAT1', 'TRUELAT2', 'STAND_LON', 'MAP_PROJ', 'GRID_ID', 'ISWATER',
                         'ISURBAN', 'ISICE', 'MMINLU', 'SIMULATION_START_DATE',
                         'I_PARENT_START', 'J_PARENT_START', 'PARENT_GRID_RATIO')
# 2-D fields read FATAL: NRL:216-242 (XLAT, XLONG, HGT, TMN, IVGTYP, ISLTYP), :264-267, :302-304
NOAHMP_REQUIRED_2D = ('XLAT', 'XLONG', 'HGT', 'TMN', 'IVGTYP', 'ISLTYP', 'CANWAT', 'TSK',
                      'SNOW', 'SNOWC', 'SHDMIN', 'SHDMAX')
NOAHMP_REQUIRED_SOIL = ('TSLB', 'SMOIS')  # (Time, soil_layers_stag, sn, we), NRL:286-294
# read NOT_FATAL: XLAND (absent -> every point is land, NoahmpIOVarInitMod default 1.0),
# SEAICE, MAPFAC_MX/MY, SNOWH (absent -> 0.005 SNOW, NIM:46-53), VEGFRA, LAI
NOAHMP_OPTIONAL = ('XLAND', 'SEAICE', 'MAPFAC_MX', 'MAPFAC_MY', 'SNOWH', 'VEGFRA', 'LAI')


def _2d(inp, name):
    a = np.asarray(inp[name].values, np.float64)
    return a[0] if a.ndim == 3 else a


def add_noahmp_land_state(inp, geo, land, date=None, grid_id=1, dzs=HL.NOAHMP_DZS,
                          lapse_rate=TR.STD_LAPSE, adjust_tsk=False, adjust_soil_temp=False,
                          soil_fallback=None, snow_method='interp', dtype=np.float32, **snow_kw):
    """Write the Noah-MP land state into a wrfinput Dataset (to_wrfinput output with HGT, TSK,
    SST and XLAT/XLONG, or a wrfinput read back with xarray) and return (inp, record).

    geo: geo_em-style Dataset (read_geogrid) on the same grid; land: read_hrrr_land(...) of
    the HRRR file(s) valid at the start time (surface and snow fields, and the RUC soil
    levels); date: start time (default: SIMULATION_START_DATE).
    Writes LANDMASK / IVGTYP / ISLTYP and VEGFRA, SHDMIN, SHDMAX, LAI, ALBBCK, SNOALB, TMN
    (map_static_fields), XLAND = 2 - LANDMASK, SEAICE, TSLB / SMOIS / SH2O / ZS / DZS
    (soil_state_from_hrrr), SNOW / SNOWH / SNOWC / CANWAT (snow_from_hrrr with snow_method
    and snow_kw, e.g. snowc, or radius / max_gradient / cap for snow_method='elevation'), the
    GRID_ID attribute, and LAND_STATE_OPTIONS, a JSON attribute with every option used
    (snow_kw with the defaults filled in).
    The defaults are plain: TSK and the soil temperature as HRRR has them, snow by land-only
    interpolation, SNOWC from HRRR's snow cover. Options for a target terrain much finer
    than HRRR's:
    adjust_tsk=True: on land, TSK = HRRR skin temperature (land points only) + lapse_rate
      (HGT_hrrr - HGT); SST is left unchanged. adjust_soil_temp=True: the same adjustment of
      TSLB. Both as WRF real.exe adjust_soil_temp_new (per WRF documentation, not verified
      against source here).
    snow_method='elevation': see hrrr_land.snow_from_hrrr."""
    if date is None:
        date = pd.to_datetime(str(inp.attrs['SIMULATION_START_DATE']).replace('_', ' '))
    inp = TR.add_geogrid_land_fields(inp, geo, date, lsm_fields=True, dtype=dtype)
    lat, lon = _2d(inp, 'XLAT'), _2d(inp, 'XLONG')
    hgt = _2d(inp, 'HGT')
    landmask = _2d(inp, 'LANDMASK')
    is_land = landmask > 0.5
    xo, yo = HL.hrrr_xy(lat, lon)
    sub = HL.subset_to_points(land, xo, yo)
    regrid = HL.LandRegridder.from_dataset(sub, xo, yo)
    record = {'hrrr_file': str(land.attrs.get('source', '')), 'date': str(date),
              'hrrr_subset_shape': [int(sub.sizes['y']), int(sub.sizes['x'])]}

    tsk = _2d(inp, 'TSK')
    if adjust_tsk:
        valid = sub['LANDMASK'].values > 0.5
        skin, _ = regrid(sub['SKINT'].values, valid)
        hgt_src, _ = regrid(sub['HGT'].values, valid)
        new = np.where(is_land, skin + lapse_rate * (hgt_src - hgt), tsk)
        record['tsk_change_land_K'] = ([float((new - tsk)[is_land].min()),
                                        float((new - tsk)[is_land].max())] if is_land.any() else [])
        tsk = new
        inp['TSK'] = (('Time', 'south_north', 'west_east'), tsk[None].astype(dtype))
    soil, record['soil'] = HL.soil_state_from_hrrr(
        sub, regrid, hgt, landmask, tmn=_2d(inp, 'TMN'), tsk=tsk, sst=_2d(inp, 'SST'),
        dzs=dzs, lapse_rate=lapse_rate, adjust_soil_temp=adjust_soil_temp, fallback=soil_fallback)
    snow, record['snow'] = HL.snow_from_hrrr(sub, regrid, hgt, landmask, method=snow_method,
                                             **snow_kw)
    bound = inspect.signature(HL.snow_from_hrrr).bind_partial(**snow_kw)
    bound.apply_defaults()
    snow_opts = {k: v for k, v in bound.arguments.items()
                 if k not in ('land', 'regrid', 'hgt_new', 'landmask', 'method')}
    record['options'] = {
        'adjust_tsk': bool(adjust_tsk), 'adjust_soil_temp': bool(adjust_soil_temp),
        'lapse_rate_K_per_m': float(lapse_rate), 'soil_fallback': soil_fallback,
        'snow_method': snow_method, 'snow': snow_opts, 'dzs_m': [float(d) for d in dzs],
        'grid_id': int(grid_id),
        'hrrr_files': [s.strip().rsplit('/', 1)[-1] for s in record['hrrr_file'].split(',') if s.strip()]}

    m2 = ('Time', 'south_north', 'west_east')
    m3 = ('Time', 'soil_layers_stag', 'south_north', 'west_east')
    for name in ('TSLB', 'SMOIS', 'SH2O'):
        inp[name] = (m3, soil[name][None].astype(dtype))
    for name in ('ZS', 'DZS'):
        inp[name] = (('Time', 'soil_layers_stag'), soil[name][None].astype(dtype))
    for name, arr in snow.items():
        inp[name] = (m2, arr[None].astype(dtype))
    inp['XLAND'] = (m2, (2.0 - (is_land.astype(np.float64)))[None].astype(dtype))
    seaice = np.zeros(hgt.shape)
    if 'SEAICE' in sub and (sub['LANDMASK'].values < 0.5).any():
        ice, _ = regrid(np.clip(np.nan_to_num(sub['SEAICE'].values), 0.0, 1.0),
                        sub['LANDMASK'].values < 0.5)
        seaice = np.where(is_land, 0.0, ice)
    inp['SEAICE'] = (m2, seaice[None].astype(dtype))
    units = {'TSLB': 'K', 'SMOIS': 'm3 m-3', 'SH2O': 'm3 m-3', 'ZS': 'm', 'DZS': 'm',
             'SNOW': 'kg m-2', 'SNOWH': 'm', 'SNOWC': '1', 'CANWAT': 'kg m-2', 'TMN': 'K',
             'VEGFRA': '%', 'SHDMIN': '%', 'SHDMAX': '%', 'ALBBCK': '1', 'SNOALB': '1'}
    for name, u in units.items():
        inp[name].attrs['units'] = u
    inp.attrs['GRID_ID'] = np.int32(grid_id)
    inp.attrs['LAND_STATE_OPTIONS'] = json.dumps(record['options'], default=str)
    inp.attrs['NUM_SOIL_LAYERS'] = np.int32(len(dzs))
    return inp, record


def check_noahmp_inputs(wrfinput, nsoil=4):
    """The fields and attributes ERF's Noah-MP reader needs, checked on a wrfinput (path or
    Dataset). Returns {'missing': [...], 'invalid': [...], 'notes': [...], 'ok': bool};
    'missing' and 'invalid' would abort or mislead Noah-MP, 'notes' are what Noah-MP does
    by itself (defaults, caps)."""
    ds = xr.open_dataset(wrfinput) if isinstance(wrfinput, str) else wrfinput
    missing, invalid, notes = [], [], []
    for a in NOAHMP_REQUIRED_ATTRS:
        if a not in ds.attrs and not (a == 'GRID_ID' and 'grid_id' in ds.attrs):
            missing.append(f'attribute {a}')
    if 'SIMULATION_START_DATE' in ds.attrs:
        try:  # NRL:97-121: YYYY-MM-DD_HH:MM[:SS] or with a space
            pd.to_datetime(str(ds.attrs['SIMULATION_START_DATE']).replace('_', ' ')[:16],
                           format='%Y-%m-%d %H:%M')
        except ValueError:
            invalid.append(f"SIMULATION_START_DATE {ds.attrs['SIMULATION_START_DATE']!r}")
    shape = None
    if 'WEST-EAST_GRID_DIMENSION' in ds.attrs and 'SOUTH-NORTH_GRID_DIMENSION' in ds.attrs:
        shape = (int(ds.attrs['SOUTH-NORTH_GRID_DIMENSION']) - 1,
                 int(ds.attrs['WEST-EAST_GRID_DIMENSION']) - 1)
    for name in NOAHMP_REQUIRED_2D + NOAHMP_REQUIRED_SOIL + ('DZS',):
        if name not in ds.variables:
            missing.append(name)
    for name in NOAHMP_OPTIONAL:
        if name not in ds.variables:
            notes.append(f'{name} absent (optional)' + {
                'XLAND': ': Noah-MP then treats every point as land',
                'SNOWH': ': Noah-MP sets SNOWH = 0.005 SNOW'}.get(name, ''))

    def arr(name):
        a = np.asarray(ds[name].values, np.float64)
        return a[0] if a.ndim >= 3 and a.shape[0] == 1 else a

    for name in NOAHMP_REQUIRED_2D + NOAHMP_OPTIONAL:
        if name in ds.variables:
            a = arr(name)
            if shape is not None and a.shape != shape:
                invalid.append(f'{name} shape {a.shape} != {shape}')
            elif not np.all(np.isfinite(a)):
                invalid.append(f'{name} has non-finite values')
    land = arr('LANDMASK') > 0.5 if 'LANDMASK' in ds.variables else None
    if 'DZS' in ds.variables:
        dzs = np.asarray(ds['DZS'].values, np.float64).ravel()
        if dzs.size < nsoil or np.any(dzs[:nsoil] <= 0):
            invalid.append(f'DZS {list(dzs)}: needs {nsoil} positive thicknesses (NRL:274-284)')
    for name in NOAHMP_REQUIRED_SOIL:
        if name in ds.variables:
            a = arr(name)
            if a.ndim != 3 or a.shape[0] < nsoil:
                invalid.append(f'{name} shape {a.shape}: needs {nsoil} soil layers first')
                continue
            lo, hi = {'TSLB': (200.0, 330.0), 'SMOIS': (0.0, 1.0)}[name]
            v = a[:nsoil][:, land] if land is not None else a[:nsoil]
            if not np.all(np.isfinite(a)):
                invalid.append(f'{name} has non-finite values')
            elif v.size and (v.min() < lo or v.max() > hi):
                invalid.append(f'{name} on land outside [{lo:g}, {hi:g}]: [{v.min():g}, {v.max():g}]')
    ranges = {'SNOW': (0.0, None), 'SNOWH': (0.0, None), 'SNOWC': (0.0, 1.0),
              'CANWAT': (0.0, None), 'TMN': (200.0, 330.0), 'TSK': (180.0, 340.0),
              'SHDMIN': (0.0, 100.0), 'SHDMAX': (0.0, 100.0), 'VEGFRA': (0.0, 100.0),
              'LAI': (0.0, 100.0), 'XLAND': (1.0, 2.0), 'SEAICE': (0.0, 1.0)}
    for name, (lo, hi) in ranges.items():
        if name in ds.variables:
            a = arr(name)
            if np.isfinite(a).all() and (a.min() < lo or (hi is not None and a.max() > hi)):
                invalid.append(f'{name} outside [{lo:g}, {hi}]: [{a.min():g}, {a.max():g}]')
    if 'ISLTYP' in ds.variables and np.nanmin(arr('ISLTYP')) < 1:
        invalid.append('ISLTYP < 1 somewhere (Noah-MP stops, NIM:80-90)')
    if 'SHDMIN' in ds.variables and 'SHDMAX' in ds.variables and np.any(arr('SHDMIN') > arr('SHDMAX')):
        invalid.append('SHDMIN > SHDMAX somewhere')
    if 'SNOW' in ds.variables and np.nanmax(arr('SNOW')) > 2000.0:
        notes.append('SNOW > 2000 kg m-2 somewhere: Noah-MP caps it, keeping density (NIM:66-69)')
    if land is not None and 'XLAND' in ds.variables and np.any((arr('XLAND') > 1.5) == land):
        invalid.append('XLAND disagrees with LANDMASK')
    if land is not None and 'SNOW' in ds.variables and np.any(arr('SNOW')[~land] != 0.0):
        notes.append('SNOW nonzero on water points')
    notes.append('SH2O and ZS are not read by Noah-MP: SH2O is recomputed from SMOIS and TSLB '
                 '(NIM:92-130); CANWAT is read but reset to 0 (NIM:141)')
    return {'missing': missing, 'invalid': invalid, 'notes': notes,
            'ok': not missing and not invalid}
