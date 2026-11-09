"""erftools.geog: Python geogrid replacement for ERF static fields (no WPS/WRF code at run time).

Builds a geo_em-compatible NetCDF (terrain, land use, soil, vegetation, ...) for a
Lambert-conformal C-grid directly from the WPS_GEOG_V4 tiles. Presets ``wps_equivalent``
(geogrid v4.4 re-implementation, bit-identical to geogrid) and ``slope_limited`` (our own
choice for ERF at 500 m); see README.md in this directory.

  python -m erftools.geog --preset slope_limited --landuse modis --out geo_em.d01.nc

The provenance global attributes use the ``ERFTOOLS_GEOG_*`` names.
"""
__version__ = "0.1.0"
