from datetime import datetime
from importlib import resources

import numpy as np
import xarray as xr
import click

from erftools.utils import get_zcc, est_p
from erftools.inputs import ERFInputs


# wrap Jan/Dec for a 365 day year
my_date_oz = np.array(
    [-15,
     16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350,
     381]
)

def monthly_interp_weights_from_date(dt):
    """Get interpolation weights to be applied to monthly data centered
    on specific days of the year (hardcoded).
    """
    # get decimal day of year (1-based)
    jan1 = datetime(dt.year, 1, 1)
    decimal_day = (dt - jan1).total_seconds() / 86400. + 1.0

    # find months to interpolate between
    idx_r = np.where(my_date_oz > decimal_day)[0][0]
    idx_l = idx_r - 1

    # calculate interpolation weights
    tdelta = my_date_oz[idx_r] - my_date_oz[idx_l]
    fac_l = (my_date_oz[idx_r] - decimal_day) / tdelta
    fac_r = (decimal_day - my_date_oz[idx_l]) / tdelta

    # fill full array of monthly weights
    weights = np.zeros(12)
    if idx_l == 0 or idx_r == 13:
        weights[-1] = fac_l
        weights[0] = fac_r
    else:
        weights[idx_l-1] = fac_l
        weights[idx_r-1] = fac_r

    return xr.DataArray(weights, dims='month')

def interp_from_monthly_to_date(xdata,dt):
    """Interpolate from monthly data `xdata` (an xarray Dataset or
    DataArray) to a specific datetime `dt`

    The data should have the dimension 'month'
    """
    weightedvals = monthly_interp_weights_from_date(dt) * xdata
    return weightedvals.sum('month')

def interp_from_monthly_to_date_heights(xdata,dt,heights,**kwargs):
    """Interpolate from monthly data `xdata` (an xarray Dataset or
    DataArray) to a specific datetime `dt` and an array of model heights

    The data should have the dimensions 'month' and 'plev' and the
    pressure levels are in hPa.
    """
    # get pressure levels from a standard atmosphere
    plev = est_p(heights,**kwargs) / 100 # hPa

    # interpolate to pressure levels
    plev_data = interp_from_monthly_to_date(xdata, dt)
    interp_data = plev_data.interp(plev=plev)

    # create new dimension coordinate
    interp_data = interp_data.rename(plev='height')
    interp_data = interp_data.assign_coords(height=heights,
                                            plev=('height',plev))

    # fill in near-surface values
    interp_data = interp_data.bfill('height')

    return interp_data


def interp_ozone(inp, latitude, dataset='CAM', verbose=False):
    """Interpolate from climatological O3 data to the starting datetime
    and height levels from the ERF inputfile.

    `inp` may be an input file or an ERFInputs object

    Ozone datasets are stored in erftools/data/ozone_*.nc
    """
    dataname = f'ozone_{dataset}.nc'
    with resources.files('erftools.data').joinpath(dataname) as fpath:
        ds = xr.load_dataset(fpath)

    date = inp.start_datetime
    if date is None:
        raise ValueError(f'start_date not found in {inputfile}')
    if verbose:
        print('Interpolating to',date)

    heights = get_zcc(inp,verbose=verbose)
    if verbose:
        print('Model heights:',heights)

    plevels = est_p(heights)
    if verbose:
        print('Standard pressure levels:',plevels)

    ds = ds.interp(lat=latitude)
    interpdata = interp_from_monthly_to_date_heights(ds, date, heights)
    if verbose:
        print('O3 [ppbv]: ', interpdata['o3vmr'].values / 1e-9)

    return interpdata['o3vmr'].values


@click.command()
@click.argument('inputfile', type=click.Path(writable=True))
@click.option('--latitude', type=click.FloatRange(-90,90),
              required=True)
@click.option('--dataset', default='CAM',
              help='Name of O3 dataset [default=CAM]')
def generate_ozone_profile(inputfile, latitude, dataset):
    """Interpolate from climatological O3 data to the starting datetime
    and height levels from the ERF inputfile.

    Ozone datasets are stored in erftools/data/ozone_*.nc
    """
    inp = ERFInputs(inputfile)
    o3vmr = interp_ozone(inp, latitude, dataset, verbose=True)
    erfstr = str(o3vmr).lstrip('[').rstrip(']')
    print('erf.o3vmr = ', erfstr)



'''
For verifying interpolator behavior

date_oz = [16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350]
daysperyear = 365

def ozn_time_int(julday, julian, ozmixm=None):
    """Interpolate ozone concentration profile

    Port of ozn_time_int subroutine from WRF
    phys/module_radiation_driver.F

    Parameters
    ----------
    julday: int
        Day of year
    julian: float
        Day of year, 0.0 at 0Z on 1 Jan
    ozmixm: np.array
        Ozone concentration (nlon, nlev, nlat, nmonth)
    """
    ozncyc = True
    intjulian = julian + 1.0
    # Jan 1st 00Z is julian=1.0 here
    ijul = int(intjulian)
    intjulian = intjulian - ijul
    ijul = ijul % 365
    if ijul==0:
        ijul = 365
    intjulian = intjulian + ijul

    np1 = 0
    datefound = False
    for m in range(12):
        if (date_oz[m] > intjulian) and not datefound:
            np1 = m
            datefound = True
    cdayozp = date_oz[np1]

    if np1 > 0:
        cdayozm = date_oz[np1-1]
        np = np1
        nm = np - 1
    else:
        cdayozm = date_oz[11]
        np = np1 # ==0, jan
        nm = 11 # dec

    if ozncyc and np1 == 0:
        # Dec-Jan interpolation
        deltat = cdayozp + daysperyear - cdayozm
        if intjulian > cdayozp:
            print('interp in dec',cdayozm,cdayozp)
            # We are in December
            fact1 = (cdayozp + daysperyear - intjulian) / deltat
            fact2 = (intjulian - cdayozm) / deltat
        else:
            print('interp in jan',cdayozm,cdayozp)
            # We are in January
            fact1 = (cdayozp - intjulian) / deltat
            fact2 = (intjulian + daysperyear - cdayozm) / deltat
    else:
        print('interp in general',cdayozm,cdayozp)
        deltat = cdayozp - cdayozm
        fact1 = (cdayozp - intjulian) / deltat
        fact2 = (intjulian - cdayozm) / deltat

    if ozmixm is not None:
        # WRF source has indices nm+1, np+1, which should be to account for
        # module_ra_cam_suopport:oznini filling ozmixm starting from m=2...
        return ozmixm[:,:,:,nm]*fact1 + ozmixm[:,:,:,np]*fact2
    else:
        return fact1, fact2
'''
