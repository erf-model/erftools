import click
import yt
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from erftools.grids import LambertConformalGrid

def slice_and_contour_latlon(
    plotfile, field,
    slice_axis='z', slice_coord=0.0,
    outfile_data=None, outfile_plot=None,
    xres=512, yres=512, y_scale=1.05,
    lon_min=None, lon_max=None,
    lat_min=None, lat_max=None,
    cen_lat=None, cen_lon=None,
    truelat1=None, truelat2=None,
    standlon=None,
    cmap='viridis',
    feature_res='50m',
    ref_lat=None, ref_lon=None, ref_name=None
):
    """Extract a 2D slice from an AMReX plotfile and plot in lat/lon
    coordinates with Cartopy.

    If lat/lon min/max are provided, then the ERF grid is assumed to
    have a Plate-Carree projection. Otherwise, a Lambert Conformal
    projection may be defined by the central lat/lon, standard
    parallel(s), and standard longitude.

    Parameters
    ----------
    plotfile : str
        Path to AMReX plotfile
    field : str
        Name of field in plotfile
    slice_axis : 'x', 'y', or 'z'
        Slice normal direction
    slice_coord : float
        Slice location along slice_axis
    outfile_data : str, optional
        If not None, write out numpy array data to this file
    outfile_plot : str, optional
        If not None, write out this image file; otherwise show on screen
    xres, yres : int, optional
        Slice resampling resolution
    y_scale : float, optional
        Scaling factor for latitude axis
    lon_min, lon_max : float, optional
        Longitude range
    lat_min, lat_max : float, optional
        Latitude range
    cen_lat, cen_lon : float, optional
        Central latitude, longitude
    truelat1, truelat2 : float, optional
        Standard parallel(s)
    standlon : float, optional
        Standard longitude
    cmap : str, optional
        Name of colormap
    ref_lat, ref_lon : float, optional
        Lat, lon of reference location
    ref_name : str, optional
        Name of reference location
    """
    # Load dataset
    ds = yt.load(plotfile)

    # Make slice
    assert slice_axis not in ('x','y'), 'x- and y-slice plotting not set up yet'
    data_source = ds.slice(slice_axis, slice_coord)

    # Get FRB (Fixed Resolution Buffer)
    frb = data_source.to_frb((1.0, 'unitary'), (xres, yres))
    arr = np.array(frb[field])

    # Save raw data
    if outfile_data is not None:
        np.save(outfile_data, arr)
        print(f'Slice data saved to {outfile_data}')

    latlon_coords = False
    lcc_coords = False
    if ((lon_min is not None) and (lon_max is not None) and
        (lat_min is not None) and (lat_max is not None)):
        latlon_coords = True
    elif ((cen_lat is not None) and
          (cen_lon is not None) and
          (truelat1 is not None)):
        lcc_coords = True

    # Normalized coordinates -- not sure why the width=1 isn't respected
    x_unit = np.linspace(float(frb.bounds[0]), float(frb.bounds[1]), xres)
    y_unit = np.linspace(float(frb.bounds[2]), float(frb.bounds[3]), yres)

    if latlon_coords:
        # Map to lon/lat with adjustable y scaling
        lon = lon_min \
            + (lon_max - lon_min) * (x_unit - x_unit.min()) / (x_unit.max() - x_unit.min())
        lat = lat_min \
            + (lat_max - lat_min) * (y_unit - y_unit.min()) / (y_unit.max() - y_unit.min())
        lat = lat_min \
            + (lat - lat_min) * y_scale  # apply scaling

        x2d, y2d = np.meshgrid(lon, lat)

        myproj = ccrs.PlateCarree()

    elif lcc_coords:
        # get grid info from pltfile
        nx, ny, _ = ds.domain_dimensions
        Lx = ds.domain_width[0].value
        Ly = ds.domain_width[1].value

        # create grid projection
        lcc = LambertConformalGrid(
            ref_lat=cen_lat,
            ref_lon=cen_lon,
            truelat1=truelat1,
            truelat2=truelat2,
            stand_lon=standlon,
            dx=Lx/nx,
            dy=Ly/ny,
            nx=nx,
            ny=ny)

        # The extent of x_unit, y_unit are >= the domain widths in x,y,
        # respectively, and are padded with nans. For the projection to work
        # properly, the grid must be translated so that the lower left corner
        # is coincident with the original projected coordinate.
        x2d, y2d = np.meshgrid(lcc.x[0] + x_unit, lcc.y[0] + y_unit)

        myproj = lcc.proj

    # Set up Cartopy plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    if latlon_coords:
        ax.set_extent([lon_min, lon_max, lat_min, lat_min + (lat_max - lat_min) * y_scale],
                      crs=ccrs.PlateCarree())

    # Filled contour
    cs = ax.contourf(x2d, y2d, arr, levels=11,
                     cmap=cmap, transform=myproj)
    cbar = plt.colorbar(cs, ax=ax, orientation='vertical', shrink=0.95)
    cbar.set_label(field)

    # Add map features
    ax.add_feature(cfeature.STATES.with_scale(feature_res), linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS.with_scale(feature_res), linewidth=0.8, edgecolor='black')
    ax.coastlines(feature_res, linewidth=0.8)

    # Add gridlines and lat/lon ticks
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'fontsize': 'x-large'}
    gl.ylabel_style = {'fontsize': 'x-large'}

    # Add location marker if provided
    if ref_lat is not None and ref_lon is not None:
        ax.plot(ref_lon, ref_lat, color='black', marker='*', markersize=8,
                transform=ccrs.Geodetic())
        if ref_name is not None:
            ax.text(ref_lon + 0.3, ref_lat + 0.3, ref_name,
                    fontsize='small', color='black', ha='left', transform=ccrs.Geodetic())

    plt.axis('tight')
    ax.set_ylim(top=lat_max)

    if outfile_plot is None:
        plt.show()
    else:
        plt.savefig(outfile_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Contour plot saved to {outfile_plot}')

    return arr


@click.command()
@click.argument('plotfile', type=click.Path(exists=True, readable=True))
@click.argument('imagefile', type=click.Path(writable=True), required=False)
@click.option('--var', required=True, help='Variable name to plot')
@click.option('-x', 'slice_axis', flag_value='x', help='Slice in x direction')
@click.option('-y', 'slice_axis', flag_value='y', help='Slice in y direction')
@click.option('-z', 'slice_axis', flag_value='z', help='Slice in z direction (default)')
@click.option('--slice_coord', default=0,
              type=click.FLOAT,
              help='Slice location along the specified axis -- '
                   'Note: this corresponds to grid coordinates without any '
                   'terrain deformation')
@click.option('--area', nargs=4, required=False,
              type=(click.FloatRange((-180,180)), click.FloatRange((-180,180)),
                    click.FloatRange(( -90, 90)), click.FloatRange(( -90, 90))),
              help='Plot region bounds: lon_min lon_max lat_min lat_max')
@click.option('--latlon0', nargs=2, required=False,
              type=(click.FloatRange(-90,90), click.FloatRange(-180,180)),
              help='Center latitude, longitude')
@click.option('--truelat1', required=False,
              type=click.FloatRange(-90,90),
              help='Standard parallel')
@click.option('--truelat2', required=False,
              type=click.FloatRange(-90,90),
              help='Second standard parallel (optional)')
@click.option('--standlon', required=False,
              type=click.FloatRange(-180,180),
              help='Standard longitude (optional0')
@click.option('--reflatlon', nargs=2, required=False,
              type=(click.FloatRange(-90,90), click.FloatRange(-180,180)),
              help='Reference latitude, longitude to add to plot (optional)')
@click.option('--output', '-o', required=False,
              type=click.Path(writable=True),
              help='Save slice data to this numpy array file (optional)')
def plot_slice_latlon(
    plotfile, imagefile,
    slice_axis, slice_coord,
    var,
    area,
    latlon0, truelat1, truelat2, standlon,
    reflatlon,
    output
):
    """Extract slice from ERF plotfile and plot contours on a lat/lon
    grid. The slice data are also saved to a numpy array file.

    If lat/lon min/max are provided, then the ERF grid is assumed to
    have a Plate-Carree projection. Otherwise, a Lambert Conformal
    projection may be defined by the central lat/lon, standard
    parallel(s), and standard longitude.

    \b
    Examples
    --------
    plot_slice_latlon /path/to/pltfile --var x_velocity --area -103.5 -80.5 29.8 46.18
    \b
    plot_slice_latlon /path/to/pltfile --var y_velocity --latlon0 35.85 -123.72 --truelat1 36.05 --standlon -65.0
    """
    # Default to z slice
    if slice_axis is None:
        slice_axis = 'z'

    # Parse area
    if area is not None:
        print('ERF grid maps to lat/lon')
        lon_min, lon_max, lat_min, lat_max = area
        assert (lon_max > lon_min) and (lat_max > lat_min)
    else:
        print('ERF grid has Lambert Conformal projection') 
        lon_min = lon_max = lat_min = lat_max = None
        assert latlon0 is not None, 'Need to specify central lat/lon'
        assert truelat1 is not None, 'Need to specify at least 1 standard parallel'
        cen_lat, cen_lon = latlon0

    # Filenames based on var
    if imagefile is None:
        imagefile = f'{plotfile}_{var}_{slice_axis}slice{slice_coord:g}.png'
    if output is None:
        output = f'{plotfile}_{var}_{slice_axis}slice{slice_coord:g}.npy'

    # Annotate reference location
    if reflatlon is not None:
        ref_lat, ref_lon = reflatlon
    else:
        ref_lat = ref_lon = None

    slice_and_contour_latlon(
        plotfile=plotfile,
        field=var,
        slice_axis=slice_axis,
        slice_coord=slice_coord,
        outfile_data=output,
        outfile_plot=imagefile,
        lon_min=lon_min, lon_max=lon_max,
        lat_min=lat_min, lat_max=lat_max,
        cen_lat=cen_lat, cen_lon=cen_lon,
        truelat1=truelat1, truelat2=truelat2,
        standlon=standlon,
        ref_lat=ref_lat, ref_lon=ref_lon,
    )

