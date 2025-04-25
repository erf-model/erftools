import numpy as np
import xarray as xr
import yt
yt.set_log_level('error')


xyz = ['x','y','z']

class Plotfile(object):
    """Process cell-centered volume data"""

    def __init__(self,fpath,verbose=False,*args,**kwargs):
        self.verbose = verbose
        if verbose:
            yt.set_log_level('info')
        else:
            yt.set_log_level('error')

        if verbose:
            print('Loading',fpath)
        self.pf = yt.load(fpath, *args, **kwargs)
        self.fields = [fld for (typ,fld) in self.pf.field_list if typ=='boxlib']
        if verbose:
            print('Found fields:',self.fields)

        # 1-D cordinate arrays
        self.coords1d = [None,None,None] # x1, y1, z1

        # initialize slicing data
        self.slc_coords = [None,None,None] # yz, xz, xy planes
        self.slc_order = [None,None,None]
        self.slc_shape = [None,None,None]

    def to_xarray(self,verbose=False):
        """Convert plotfile raw data to an xarray.Dataset.

        Only single level handled for now. Will need to traverse
        through grids with index.grids[gid].Children[1].Children[0], etc...

        See https://yt-project.org/doc/examining/low_level_inspection.html
        """
        dslist = []
        for fldname in self.fields:
            # e.g., fldname == "x_velocity_stag"
            if fldname.endswith('_stag'):
                stagdim = fldname[0]
            else:
                stagdim = None
            if verbose:
                print(f'Reading {fldname} (staggered dim: {stagdim})')

            # loop over grids
            dalist = []
            for ig,g in enumerate(self.pf.index.grids):
                lo_pt = g.LeftEdge.value
                hi_pt = g.RightEdge.value
                if verbose:
                    print('  grid',ig,lo_pt,hi_pt)

                fld = g[('boxlib',fldname)]
                ncell = fld.shape

                # setup dimension coordinates
                coords = {'t': [self.pf.current_time.item()]}
                for idim,coord in enumerate(['x','y','z']):
                    if coord==stagdim:
                        coords[coord+'_stag'] = \
                                lo_pt[idim] \
                                + (hi_pt[idim] - lo_pt[idim]) \
                                * np.arange(ncell[idim]) / (ncell[idim]-1)
                    else:
                        coords[coord] = \
                                lo_pt[idim] \
                                + (hi_pt[idim] - lo_pt[idim]) \
                                * np.arange(0.5,ncell[idim]) / ncell[idim]

                # create dataarray
                da = xr.DataArray(fld.value[np.newaxis,:,:,:],
                                  coords=coords,
                                  name=fldname)
                dalist.append(da)

            ds = xr.merge(dalist)
            dslist.append(ds)

        return xr.merge(dslist)

    def slice(self, axis, loc, fields=None):
        """Create cutplane through the volume at index closest to the
        requested location

        Parameters
        ----------
        axis : int
            Slice orientation (x=0, y=1, z=2)
        loc : float
            Slice location
        """
        slc = self.pf.slice(axis, loc)
        actual_loc = slc.fcoords[0,axis]
        assert np.all(slc.fcoords[:,axis] == actual_loc)
        if self.verbose:
            print('Slice at',xyz[axis],'=',slc.fcoords[0,axis].value)
        slc_dims = [idim for idim in range(3) if idim != axis]

        if self.slc_coords[axis] is None:
            # process sliced coordinates
            coords = np.stack([slc.fcoords[:,idim].value for idim in slc_dims], axis=-1)
            order = np.lexsort((coords[:,1],coords[:,0]))
            shape = tuple(self.pf.domain_dimensions[idim] for idim in slc_dims)
            coords2d_0 = coords[order,0].reshape(shape)
            coords2d_1 = coords[order,1].reshape(shape)
            array_0 = coords2d_0[:,0]
            array_1 = coords2d_1[0,:]
            # set or check 1-D coordinate arrays
            if self.coords1d[slc_dims[0]] is None:
                if self.verbose:
                    print('Setting',xyz[slc_dims[0]],'coord')
                self.coords1d[slc_dims[0]] = array_0
            else:
                assert np.all(array_0 == self.coords1d[slc_dims[0]])
            if self.coords1d[slc_dims[1]] is None:
                if self.verbose:
                    print('Setting',xyz[slc_dims[1]],'coord')
                self.coords1d[slc_dims[1]] = array_1
            else:
                assert np.all(array_1 == self.coords1d[slc_dims[1]])
            self.slc_coords[axis] = coords
            self.slc_order[axis] = order
            self.slc_shape[axis] = shape

        # create an xarray dataset for the slice
        dimnames = tuple(xyz[idim] for idim in slc_dims)
        dimcoords = [self.coords1d[idim] for idim in slc_dims]
        if fields is None:
            fieldlist = self.fields
        elif isinstance(fields, str):
            fieldlist = [fields]
        else:
            fieldlist = fields
        sliceflds = {}
        for fld in fieldlist:
            fld1 = slc[fld].value[self.slc_order[axis]]
            fld2 = fld1.reshape(self.slc_shape[axis])
            sliceflds[fld] = (dimnames, fld2)
        ds = xr.Dataset(sliceflds, coords=dict(zip(dimnames, dimcoords)))
        ds = ds.expand_dims({xyz[axis]: [actual_loc]})

        return ds.transpose('x','y','z')


def calc_cloud_cover(pltfile):
    ds = yt.load(pltfile)
    lvl0 = ds.covering_grid(level=0,
                            left_edge=ds.domain_left_edge,
                            dims=ds.domain_dimensions)
    qc = lvl0['qc'].value
    has_cloud = (np.max(qc,axis=2) > 0)
    nx,ny,_ = ds.domain_dimensions
    cloud_cover = np.count_nonzero(has_cloud) / (nx*ny)
    return float(ds.current_time), cloud_cover
