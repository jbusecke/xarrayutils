"""
Collection of several useful routines for xarray
"""
import numpy as np
import xarray as xr
from numpy_utils import numpy_block_aggregate

def aggregate(dar,blocks,func=np.nanmean):
    """
    Aggregation method for xarray.

    Somewhat of a crutch, waiting for xarray.apply. Relatively fast implementation
    of a block average

    TODO:
    Better diagnostics
    Examples
    Give the array a better name and some other attributes

    """

    try:
        numpy_dims = [dar.dims.index(a[0]) for a in blocks]
    except ValueError:
        raise RuntimeError('Block specifier not found. Likely a typo or missing dims in da')

    numpy_blocks = [tuple([a,b[1]]) for a,b in zip(numpy_dims,blocks)]

    # not 100% happy with this since the first element of chunks is a 3 element tuple,
    # but in most cases this should work regardless since we mostly chunk in time
    new_shape = [a[0] for a in dar.chunks]
    # Construct new shape. Needed for dask
    for aa in numpy_blocks:
        new_shape[aa[0]] = new_shape[aa[0]]/aa[1]

    coarse = dar.data.map_blocks(numpy_block_aggregate,chunks=new_shape,blocks=numpy_blocks)
    old_coords = dar.coords
    new_coords = dict([])

    for cc in old_coords.keys():
        new_coords[cc] = old_coords[cc].load()
    for dd in blocks:
        new_coords[dd[0]] = new_coords[dd[0]][0::dd[1]]

    da_coarse = xr.DataArray(coarse,dims=dar.dims,coords=new_coords)
    return da_coarse


# def coarsen(da,dim='time'):
# #     return da.mean(dim=dim)
#     return da.sum(dim=dim)
#
# def multi_coarsen(da,bins):
#     for bb in bins:
#         print bb
#         da = da.groupby_bins(bb[0],bb[1]).apply(coarsen,dim=bb[0])
#         da[bb[0]+'_bins'] = bb[1][0:-1]
#         da = da.rename({bb[0]+'_bins':bb[0]})
#     return da
