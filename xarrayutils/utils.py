from __future__ import print_function
# from future.utils import iteritems
import numpy as np
import xarray as xr
from scipy.signal import filtfilt, gaussian
from dask.array import coarsen
from dask.array.core import Array
import warnings

"""
Collection of several useful routines for xarray
"""
"""
Lower Level implementation in numpy and dask
"""


def aggregate_w_nanmean(da, weights, blocks, **kwargs):
    """
    weighted nanmean for xarrays
    """
    # make sure that the missing values are exactly equal to each otherwise
    weights = weights.where(~np.isnan(da))
    if not np.all(np.isnan(da) == np.isnan(weights)):
        raise RuntimeError('weights cannot have more missing values \
        then the data array')

    weights_sum = aggregate(weights, blocks, func=np.nansum, **kwargs)
    da_sum = aggregate(da * weights, blocks, func=np.nansum, **kwargs)
    return da_sum / weights_sum


def aggregate(da, blocks, func=np.nanmean, debug=False):
    """
    Performs efficient block averaging in one or multiple dimensions.
    Only works on regular grid dimensions.

    Parameters
    ----------
    da : xarray DataArray (must be a dask array!)
    blocks : list
        List of tuples containing the dimension and interval to aggregate over
    func : function
        Aggregation function.Defaults to numpy.nanmean

    Returns
    -------
    da_agg : xarray Data
        Aggregated array

    Examples
    --------
    >>> from xarrayutils import aggregate
    >>> import numpy as np
    >>> import xarray as xr
    >>> import matplotlib.pyplot as plt
    >>> %matplotlib inline
    >>> import dask.array as da

    >>> x = np.arange(-10,10)
    >>> y = np.arange(-10,10)
    >>> xx,yy = np.meshgrid(x,y)
    >>> z = xx**2-yy**2
    >>> a = xr.DataArray(da.from_array(z, chunks=(20, 20)),
                         coords={'x':x,'y':y}, dims=['y','x'])
    >>> print a

    <xarray.DataArray 'array-7e422c91624f207a5f7ebac426c01769' (y: 20, x: 20)>
    dask.array<array-7..., shape=(20, 20), dtype=int64, chunksize=(20, 20)>
    Coordinates:
      * y        (y) int64 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9
      * x        (x) int64 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9

    >>> blocks = [('x',2),('y',5)]
    >>> a_coarse = aggregate(a,blocks,func=np.mean)
    >>> print a_coarse

    <xarray.DataArray 'array-7e422c91624f207a5f7ebac426c01769' (y: 2, x: 10)>
    dask.array<coarsen..., shape=(2, 10), dtype=float64, chunksize=(2, 10)>
    Coordinates:
      * y        (y) int64 -10 0
      * x        (x) int64 -10 -8 -6 -4 -2 0 2 4 6 8
    Attributes:
        Coarsened with: <function mean at 0x111754230>
        Coarsenblocks: [('x', 2), ('y', 10)]
    """
    # Check if the input is a dask array (I might want to convert this
    # automaticlaly in the future)
    if not isinstance(da.data, Array):
        raise RuntimeError('data array data must be a dask array')
    # Check data type of blocks
    # TODO write test
    if (not all(isinstance(n[0], str) for n in blocks) or
            not all(isinstance(n[1], int) for n in blocks)):

        print('blocks input', str(blocks))
        raise RuntimeError("block dimension must be dtype(str), \
        e.g. ('lon',4)")

    # Check if the given array has the dimension specified in blocks
    try:
        block_dict = dict((da.get_axis_num(x), y) for x, y in blocks)
    except ValueError:
        raise RuntimeError("'blocks' contains non matching dimension")

    # Check the size of the excess in each aggregated axis
    blocks = [(a[0], a[1], da.shape[da.get_axis_num(a[0])] % a[1])
              for a in blocks]

    # for now default to trimming the excess
    da_coarse = coarsen(func, da.data, block_dict, trim_excess=True)

    # for now default to only the dims
    new_coords = dict([])
    # for cc in da.coords.keys():
    warnings.warn("WARNING: only dimensions are carried over as coordinates")
    for cc in list(da.dims):
        new_coords[cc] = da.coords[cc]
        for dd in blocks:
            if dd[0] in list(da.coords[cc].dims):
                new_coords[cc] = \
                    new_coords[cc].isel(
                        **{dd[0]: slice(0, -(1 + dd[2]), dd[1])})

    attrs = {'Coarsened with': str(func), 'Coarsenblocks': str(blocks)}
    da_coarse = xr.DataArray(da_coarse, dims=da.dims, coords=new_coords,
                             name=da.name, attrs=attrs)
    return da_coarse


def fancymean(raw, dim=None, axis=None, method='arithmetic',
              weights=None, debug=False):
    """ extenden mean function for xarray

    Applies various methods to estimate mean values
    {arithmetic,geometric,harmonic} along specified
    dimension with optional weigthing values, which
    can be a coordinate in the passed xarray structure
    """
    if not isinstance(raw, xr.Dataset) and not isinstance(raw, xr.DataArray):
        raise RuntimeError('input needs to be xarray structure')

    # map dim to axis so this works on ndarrays and DataArray/Dataset
    # Below is the preferred way when passing a LOT of optional values
    # and when this is implemented as a class function

    # dim = kwargs.pop('dim', None)s
    # if dim is not None:
    #     if 'axis' in kwargs:
    #         raise ValueError('cannot set both `dim` and `axis`')
    #     kwargs['axis'] = self.get_axis_num(dim)

    # For now I will add this in a simple way
    if dim is not None:
        if axis is not None:
            raise ValueError('cannot set both `dim` and `axis`')
        if isinstance(raw, xr.Dataset):
            axis = raw[raw.data_vars.keys()[0]].get_axis_num(dim)
            if debug:
                print('dim ', dim, ' changed to axis ', axis)
        elif isinstance(raw, xr.DataArray):
            axis = raw.get_axis_num(dim)
            if debug:
                print('dim ', dim, ' changed to axis ', axis)

    if debug:
        print('axis', axis)

    if weights is None:
        w = 1
    elif isinstance(weights, str):
        w = raw[weights]
    elif isinstance(weights, np.ndarray):
        w = xr.DataArray(np.ones_like(raw.data),
                         coords=raw.coords, dims=raw.dims)

    # make sure the w array is the same size as the raw array
    # This way also nans will be propagated correctly in a bidirectional way
    ones = raw.copy()
    ones = (ones * 0) + 1
    w = w * ones
    # now transpose the w array to the same axisorder as raw
    order = raw.dims
    w = w.transpose(*order)

    if method == 'arithmetic':
        up = raw * w
        down = w
        out = up.sum(axis=axis) / down.sum(axis=axis)
    elif method == 'geometric':
        w = w.where(raw > 0)
        raw = raw.where(raw > 0)
        up = np.log10(raw) * w
        down = w
        out = 10**(up.sum(axis=axis) / down.sum(axis=axis))
    elif method == 'harmonic':
        w = w.where(raw != 0)
        raw = raw.where(raw != 0)
        up = w / raw
        down = w
        out = down.sum(axis=axis) / up.sum(axis=axis)
    if debug:
        print('w', w.shape)
        print('raw', raw.shape)
        print('up', up.shape)
        print('down', down.shape)
        print('out', out.shape)

    return out


def timefilter(xr_in, steps,
               step_spec, timename='time',
               filtertype='gaussian', stdev=0.1):
    timedim = xr_in.dims.index(timename)
    dt = np.diff(xr_in.time.data[0:2])[0]
    cut_dt = np.timedelta64(steps, step_spec)

    if filtertype == 'gaussian':
        win_length = (cut_dt / dt).astype(int)
        a = [1.0]
        win = gaussian(win_length, std=(float(win_length) * stdev))
        b = win / win.sum()
        if np.nansum(win) == 0:
            raise RuntimeError('window to short for time interval')
            print('win_length', str(win_length))
            print('stddev', str(stdev))
            print('win', str(win))

    filtered = filtfilt(b, a, xr_in.data, axis=timedim, padtype=None, padlen=0)
    out = xr.DataArray(filtered, dims=xr_in.dims, coords=xr_in.coords,
                       attrs=xr_in.attrs)
    out.attrs.update({'filterlength': (steps, step_spec),
                      'filtertype': filtertype})
    if xr_in.name:
        out.name = xr_in.name + '_lowpassed'
    return out


def extractBoxes(da, bo, xname=None, yname=None, xdim='lon', ydim='lat'):
    """ Extracts boxes from DataArray


    Keyword arguments:
    da -- xarray dataarray
    bo -- dict with box name as keys and box corner
    values as numpy array ([x0,x1,y0,y1])
    xdim -- dimension name for x (default: 'lon')
    ydim -- dimension name for y (default: 'lat')


    xname -- coordinate name for x (default: 'None')
    yname -- coordinate name for y (default: 'None')
    xname and yname have to be specified if coordinates are of differnt shape
    """

    if not type(xname) == type(yname):
        raise RuntimeError('xname and yname need to be the same type')

    timeseries = []
    for ii, bb in enumerate(bo.keys()):
        box = bo[bb]
        if xname is None:
            box_dict = {xdim: slice(box[0], box[1]),
                        ydim: slice(box[2], box[3])}
            temp = da.loc[box_dict]
        else:
            mask = np.logical_and(np.logical_and(da[xname] > box[0],
                                                 da[xname] < box[1]),
                                  np.logical_and(da[yname] > box[2],
                                                 da[yname] < box[3]))
            temp = da.where(mask)

        timeseries.append(temp)
    out = xr.concat(timeseries, 'boxname')
    out['boxname'] = bo.keys()
    return out


def composite(data, index, bounds):
    """
    Composites Dataarray according to index

    Parameters
    ----------
    data : xarray.Dataarray
    index : xarray.Dataarray
        Timeseries matching one dimension of 'data'. Values lower(higher) then
        'bounds' are composited in additional coordinate
    bounds : int or array_like
        Values determining the values of 'index' composited into
        ['low','neutral','high']. If given as int, bounds will be computed as
        [-std(index) std(index)]*bounds.

    Returns
    -------
    composited_array : array_like
        xarray like data with additional composite-coordinate
        ['low','neutral','high'] based on 'bounds'

    Examples
    --------
    TODO
    """
    if isinstance(bounds, int):
        bounds = float(bounds)

    if isinstance(bounds, float):
        bounds = [-bounds * np.std(index), bounds * np.std(index)]

    if len(bounds) != 2:
        raise RuntimeError('bounds can only have 1 or two elements')

    comp_name = 'composite'
    zones = [index >= bounds[1],
             np.logical_and(index < bounds[1],
                            index >= bounds[0]),
             index < bounds[0]]
    zones_coords = ['high', 'neutral', 'low']
    out = xr.concat([data.where(z) for z in zones], comp_name)
    out[comp_name] = zones_coords

    out.attrs['IndexName'] = index.name
    out.attrs['CompositeBounds'] = bounds

    return out


def corrmap(a, b, shifts=0,
            a_x_dim='i', a_y_dim='j',
            a_x_coord=None, a_y_coord=None,
            b_x_dim='i', b_y_dim='j',
            b_x_coord=None, b_y_coord=None,
            t_dim='time', debug=True):
    """
    a -- input
    b -- target ()

    TODO
    This thing is slow. I can most likely rewrite this with \
    numpy.apply_along_axis


    """

    from scipy.stats import linregress

    if not type(a_x_coord) == type(a_y_coord):
        raise RuntimeError('a_x_coord and a_y_coord need to be the same type')

    if not type(b_x_coord) == type(b_y_coord):
        raise RuntimeError('a_x_coord and a_y_coord need to be the same type')

    if isinstance(shifts, int):
        shifts = [shifts]

    # determine if the timseries is a timeseries or a 3d array
    if len(b.shape) == 3:
        arrayswitch = True
    elif len(b.shape) == 1:
        arrayswitch = False
    else:
        raise RuntimeWarning('this only works with a timseries \
            or map of timeseries')

    # shift timeseries
    slope = []
    corr = []
    p_value = []

    for sh, shift in enumerate(shifts):
        shifted_b = b.shift(time=shift)

        s = a.mean(dim=t_dim).copy()
        s[:] = np.nan
        s.name = a.name + ' regressed onto ' + b.name

        c = a.mean(dim=t_dim).copy()
        c[:] = np.nan
        c.name = 'Corr coeff ' + a.name + '/' + b.name

        p = a.mean(dim=t_dim).copy()
        p[:] = np.nan
        p.name = 'p value ' + a.name + '/' + b.name

        for ii in range(len(a[a_x_dim])):
            for jj in range(len(a[a_y_dim])):

                # Define the 'input' (position in a) correctly, accounting for
                # the possibility that the
                # lat/lon position can be defined in the coordinates
                # or dimensions
                # interp timeseries onto the data.time
                in_a = a[{a_x_dim: ii, a_y_dim: jj}]

                if arrayswitch:
                    if not a_x_coord:
                        in_x = in_a[a_x_dim].data
                        in_y = in_a[a_y_dim].data
                    else:
                        in_x = in_a[a_x_coord].data
                        in_y = in_a[a_y_coord].data

                    # rename the dimensions so it can be reindexed
                    if not b_x_coord:
                        in_b = xr.DataArray(shifted_b.data,
                                            coords={'xdim':
                                                    shifted_b[b_x_dim].data,
                                                    'ydim':
                                                    shifted_b[b_y_dim].data,
                                                    'time':
                                                    shifted_b.time.data},
                                            dims=['time', 'ydim', 'xdim'])
                    else:
                        raise RuntimeError('Not implemented yet')
                        # This would have to be acomplished by a mask of some
                        # sort
                        # (with some tolerance around the input position)

                    # extract the matching timeseries
                    in_b = in_b.sel(xdim=in_x, ydim=in_y, method='nearest')
                    reindexed_b = in_b.reindex_like(in_a.time,
                                                    method='nearest')
                else:
                    reindexed_b = shifted_b.reindex_like(in_a.time,
                                                         method='nearest')

                x = reindexed_b.data
                y = in_a.data

                idx = np.logical_and(~np.isnan(y), ~np.isnan(x))
                if y[idx].size:
                    s[{a_x_dim: ii, a_y_dim: jj}], _, c[{a_x_dim: ii,
                                                         a_y_dim: jj}],\
                        p[{a_x_dim: ii, a_y_dim: jj}], _ = linregress(x[idx],
                                                                      y[idx])
        slope.append(s)
        corr.append(c)
        p_value.append(p)

    out_s = xr.concat(slope, 'timeshifts')
    out_s['timeshifts'] = shifts
    # !!! I think this is a bug...this should be
    # possible with
    out_c = xr.concat(corr, 'timeshifts')
    out_c['timeshifts'] = shifts
    out_p = xr.concat(p_value, 'timeshifts',)
    out_p['timeshifts'] = shifts

    return out_c, out_p, out_s



def find_surf_ind(da, surf_val, dim):
    """finds the index along dim that corresponds to the value of
    da closest to surf_val for each remaining dim"""
    # TODO can I specify how to handle multiple occurences?
    # Right now it takes the first occurence"

    # Broadcast the surface value to the full shape of da
    surf_val_exp = (da*0)+surf_val
    return abs(da-surf_val_exp).argmin(dim)


# def extract_surf_ind(da, ind, dim):
#     """squash the array to the values defined in ind"""
#     # TODO: This seems error prone...is there another way then using the mean?
#     return da.where(da[dim] == ind).mean(dim)


def extract_surf(da_ind, da_target, surf_val, dim,
                 constant_dims=['time'], fill_value=1000, masking=True):
    """Extract values of 'da_target' on a surface in 'da_ind', specified as nearest
    value to 'surf_val along 'dim'"""
    # Mask out areas where the surface runs into the boundary
    if masking:
        condition = \
            xr.ufuncs.logical_or((da_ind.max(dim) <= surf_val),
                (da_ind.min(dim) >= surf_val))
        if constant_dims:
            condition = condition.any(constant_dims)

    da_ind_filled = da_ind.fillna(fill_value)
    if (isinstance(surf_val, float) or isinstance(surf_val, int)):
        surf_val_filled = surf_val
    else:
        surf_val_filled = surf_val.fillna(fill_value)

    ind = find_surf_ind(da_ind_filled, surf_val_filled, dim)
    # Expand ind into full dimensions
    ind_exp = (da_ind_filled*0)+ind

    target = (da_target[dim]*0)+range(len(da_target[dim]))
    target_exp = target+(da_ind_filled*0)
    target_pos = da_target[dim]+(da_ind_filled*0)

    found_ind = target_exp == ind_exp

    # Mask out the regions where the surface outcrops at any point over
    # 'constant_dims'
    if masking:
        da_target = da_target.where(~condition)
        target_pos = target_pos.where(~condition)

    surf = da_target.where(found_ind)
    surf_pos = target_pos.where(found_ind)

    surf_out = surf.mean(dim)
    pos_out = surf_pos.mean(dim)
    if masking:
        pos_out = pos_out.where(~xr.ufuncs.isnan(da_target[{dim: 0}]))
    return surf_out, pos_out
