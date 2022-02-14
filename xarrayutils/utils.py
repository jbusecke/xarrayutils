import numpy as np
import xarray as xr
from scipy.signal import filtfilt, gaussian
from scipy import stats

from dask.array import coarsen, ones_like
from dask.array.core import Array
import warnings
from xarrayutils.utilities import detect_dtype
from xarrayutils.filtering import filter_1D as filter_1D_refactored


"""
Collection of several useful routines for xarray
"""


# Needs testing
def shift_lon(ds, londim, shift=360, crit=0, smaller=True, sort=True):
    ds = ds.copy()
    lon = ds[londim].data

    if smaller:
        lon[lon < crit] = lon[lon < crit] + shift
    else:
        lon[lon > crit] = lon[lon > crit] + shift

    ds[londim].data = lon
    if sort:
        ds = ds.sortby(londim)
    return ds


def xr_linregress(x, y, dim="time"):
    """Calculates linear regression along dimension `dim`.
    Results are equivalent to `scipy.stats.linregress`.

    Parameters
    ----------
    x : {xr.DataArray}
        Independent variable for linear regression. E.g. time.
    y : {xr.DataArray, xr.Dataset}
        Dependent variable.
    dim : str
        Dimension over which to perform linear regression.
        Must be present in both `a` and `b` (the default is 'time').

    Returns
    -------
    type(b)
        Returns a dataarray containing the parameter values
        for each data_variable in `b`. The naming convention
        follows `scipy.stats.linregress`

    """
    # align the nan Values before...
    x = x.where(~np.isnan(y))
    y = y.where(~np.isnan(x))
    # TODO: think about making this optional? Right now I err on the side of caution

    # Inspired by this post https://stackoverflow.com/a/60352716 but adjusted, so that
    # results are exactly as with scipy.stats.linregress for 1d vectors.

    n = y.notnull().sum(dim)

    nanmask = np.isnan(y).all(dim)

    xmean = x.mean(dim)
    ymean = y.mean(dim)
    xstd = x.std(dim)
    ystd = y.std(dim)

    cov = ((x - xmean) * (y - ymean)).sum(dim) / (n)
    cor = cov / (xstd * ystd)

    slope = cov / (xstd**2)
    intercept = ymean - xmean * slope

    df = n - 2
    TINY = 1.0e-20
    tstats = cor * np.sqrt(df / ((1.0 - cor + TINY) * (1.0 + cor + TINY)))
    stderr = slope / tstats

    pval = (
        xr.apply_ufunc(
            stats.distributions.t.sf,
            abs(tstats),
            df,
            dask="parallelized",
            output_dtypes=[y.dtype],
        )
        * 2
    )

    return xr.Dataset(
        {
            "slope": slope,
            "intercept": intercept,
            "r_value": cor.fillna(0).where(~nanmask),
            "p_value": pval,
            "std_err": stderr.where(~np.isinf(stderr), 0),
        }
    )


def linear_trend(obj, dim):
    """Convenience wrapper for 'xr_linregress'. Calculates the trend per
    given timestep. E.g. if the data is passed as yearly values, the
    trend is in units/yr.
    """
    x = xr.DataArray(
        np.arange(len(obj[dim])).astype(np.float), dims=dim, coords={dim: obj[dim]}
    )
    trend = xr_linregress(x, obj, dim=dim)
    return trend


def _lin_trend_legacy(y):
    """ufunc to be used by linear_trend"""
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)


def aggregate_w_nanmean(da, weights, blocks, **kwargs):
    """
    weighted nanmean for xarrays
    """
    # make sure that the missing values are exactly equal to each otherwise
    weights = weights.where(~np.isnan(da))
    if not np.all(np.isnan(da) == np.isnan(weights)):
        raise RuntimeError(
            "weights cannot have more missing values \
        then the data array"
        )

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
        raise RuntimeError("data array data must be a dask array")
    # Check data type of blocks
    # TODO write test
    if not all(isinstance(n[0], str) for n in blocks) or not all(
        isinstance(n[1], int) for n in blocks
    ):

        print("blocks input", str(blocks))
        raise RuntimeError(
            "block dimension must be dtype(str), \
        e.g. ('lon',4)"
        )

    # Check if the given array has the dimension specified in blocks
    try:
        block_dict = dict((da.get_axis_num(x), y) for x, y in blocks)
    except ValueError:
        raise RuntimeError("'blocks' contains non matching dimension")

    # Check the size of the excess in each aggregated axis
    blocks = [(a[0], a[1], da.shape[da.get_axis_num(a[0])] % a[1]) for a in blocks]

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
                new_coords[cc] = new_coords[cc].isel(
                    **{dd[0]: slice(0, -(1 + dd[2]), dd[1])}
                )

    attrs = {"Coarsened with": str(func), "Coarsenblocks": str(blocks)}
    da_coarse = xr.DataArray(
        da_coarse, dims=da.dims, coords=new_coords, name=da.name, attrs=attrs
    )
    return da_coarse


def fancymean(raw, dim=None, axis=None, method="arithmetic", weights=None, debug=False):
    """extenden mean function for xarray

    Applies various methods to estimate mean values
    {arithmetic,geometric,harmonic} along specified
    dimension with optional weigthing values, which
    can be a coordinate in the passed xarray structure
    """
    if not isinstance(raw, xr.Dataset) and not isinstance(raw, xr.DataArray):
        raise RuntimeError("input needs to be xarray structure")

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
            raise ValueError("cannot set both `dim` and `axis`")
        if isinstance(raw, xr.Dataset):
            axis = raw[raw.data_vars.keys()[0]].get_axis_num(dim)
            if debug:
                print("dim ", dim, " changed to axis ", axis)
        elif isinstance(raw, xr.DataArray):
            axis = raw.get_axis_num(dim)
            if debug:
                print("dim ", dim, " changed to axis ", axis)

    if debug:
        print("axis", axis)

    if weights is None:
        w = 1
    elif isinstance(weights, str):
        w = raw[weights]
    elif isinstance(weights, np.ndarray):
        w = xr.DataArray(np.ones_like(raw.data), coords=raw.coords, dims=raw.dims)

    # make sure the w array is the same size as the raw array
    # This way also nans will be propagated correctly in a bidirectional way
    ones = raw.copy()
    ones = (ones * 0) + 1
    w = w * ones
    # now transpose the w array to the same axisorder as raw
    order = raw.dims
    w = w.transpose(*order)

    if method == "arithmetic":
        up = raw * w
        down = w
        out = up.sum(axis=axis) / down.sum(axis=axis)
    elif method == "geometric":
        w = w.where(raw > 0)
        raw = raw.where(raw > 0)
        up = np.log10(raw) * w
        down = w
        out = 10 ** (up.sum(axis=axis) / down.sum(axis=axis))
    elif method == "harmonic":
        w = w.where(raw != 0)
        raw = raw.where(raw != 0)
        up = w / raw
        down = w
        out = down.sum(axis=axis) / up.sum(axis=axis)
    if debug:
        print("w", w.shape)
        print("raw", raw.shape)
        print("up", up.shape)
        print("down", down.shape)
        print("out", out.shape)

    return out


def timefilter(
    xr_in, steps, step_spec, timename="time", filtertype="gaussian", stdev=0.1
):
    timedim = xr_in.dims.index(timename)
    dt = np.diff(xr_in.time.data[0:2])[0]
    cut_dt = np.timedelta64(steps, step_spec)

    if filtertype == "gaussian":
        win_length = (cut_dt / dt).astype(int)
        a = [1.0]
        win = gaussian(win_length, std=(float(win_length) * stdev))
        b = win / win.sum()
        if np.nansum(win) == 0:
            raise RuntimeError("window to short for time interval")
            print("win_length", str(win_length))
            print("stddev", str(stdev))
            print("win", str(win))

    filtered = filtfilt(b, a, xr_in.data, axis=timedim, padtype=None, padlen=0)
    out = xr.DataArray(
        filtered, dims=xr_in.dims, coords=xr_in.coords, attrs=xr_in.attrs
    )
    out.attrs.update({"filterlength": (steps, step_spec), "filtertype": filtertype})
    if xr_in.name:
        out.name = xr_in.name + "_lowpassed"
    return out


def extractBox(da, box, xdim="lon", ydim="lat"):
    print("This is deprecated. Use extractBox_dict")
    box_dict = {xdim: box[0, 1], ydim: box[2, 3]}

    return extractBox_dict(da, box_dict, concat_wrap=True)
    # box_dict = {xdim: slice(box[0], box[1]),
    #             ydim: slice(box[2], box[3])}
    # return da.loc[box_dict]


def extractBox_dict(ds, box, concat_wrap=True):
    """Advanced box extraction from xarray Dataset"""
    if not isinstance(concat_wrap, dict):
        concat_wrap_dict = dict()
        for kk in box.keys():
            concat_wrap_dict[kk] = concat_wrap
        concat_wrap = concat_wrap_dict

    ds = ds.copy()
    for dim, ind in box.items():
        wrap = concat_wrap[dim]
        if np.diff(ind) < 0:  # This would trigger a python 2 error
            # if (ind[1] - ind[0]) < 0: # box is defined over a discontinuity
            dim_data = ds[dim].data
            split_a = dim_data[dim_data > ind[0]].max()
            split_b = dim_data[dim_data < ind[1]].min()
            a = ds.loc[{dim: slice(ind[0], split_a)}]
            b = ds.loc[{dim: slice(split_b, ind[1])}]
            if wrap:
                c = (a, b)
            else:
                c = (b, a)
            ds = xr.concat(c, dim)
        else:
            ds = ds.loc[{dim: slice(ind[0], ind[1])}]
    return ds


# This will be deprecated
def extractBoxes(da, bo, xname=None, yname=None, xdim="lon", ydim="lat"):
    raise RuntimeWarning("Hard deprecated. Please use extractBox_dict instead")


# def extractBoxes(da, bo, xname=None, yname=None, xdim='lon', ydim='lat'):
#     """ Extracts boxes from DataArray
#
#
#     Keyword arguments:
#     da -- xarray dataarray
#     bo -- dict with box name as keys and box corner
#     values as numpy array ([x0,x1,y0,y1])
#     xdim -- dimension name for x (default: 'lon')
#     ydim -- dimension name for y (default: 'lat')
#
#
#     xname -- coordinate name for x (default: 'None')
#     yname -- coordinate name for y (default: 'None')
#     xname and yname have to be specified if coordinates are of differnt shape
#     """
#     raise RuntimeError("this function is hellla slow! DO NOT use on \
#                        large datasets")
#
#     if not type(xname) == type(yname):
#         raise RuntimeError('xname and yname need to be the same type')
#
#     timeseries = []
#     for ii, bb in enumerate(bo.keys()):
#         box = bo[bb]
#         if xname is None:
#             box_dict = {xdim: slice(box[0], box[1]),
#                         ydim: slice(box[2], box[3])}
#             temp = da.loc[box_dict]
#         else:
#             mask = np.logical_and(np.logical_and(da[xname] > box[0],
#                                                  da[xname] < box[1]),
#                                   np.logical_and(da[yname] > box[2],
#                                                  da[yname] < box[3]))
#             temp = da.where(mask)
#
#         timeseries.append(temp)
#     boxname_dim = concat_dim_da(list(bo.keys()), 'boxname')
#     out = xr.concat(timeseries, boxname_dim)
#     return out


# Mapping related stuff
def dll_dist(dlon, dlat, lon, lat):
    """Converts lat/lon differentials into distances

    PARAMETERS
    ----------
    dlon : xarray.DataArray longitude differentials
    dlat : xarray.DataArray latitude differentials
    lon  : xarray.DataArray longitude values
    lat  : xarray.DataArray latitude values

    RETURNS
    -------
    dx  : xarray.DataArray distance inferred from dlon
    dy  : xarray.DataArray distance inferred from dlat
    """

    dll_factor = 111000.0
    dx = dlon * xr.ufuncs.cos(xr.ufuncs.deg2rad(lat)) * dll_factor
    dy = ((lon * 0) + 1) * dlat * dll_factor
    return dx, dy


# TODO: This needs a test and perhaps I can refactor it into a 'budget tools
# Module'
def convert_flux_array(da, da_full, dim, top=True, fillval=0):
    dummy = xr.DataArray(
        ones_like(da_full.data) * fillval, coords=da_full.coords, dims=da_full.dims
    )
    if top:
        da.coords[dim] = da_full[dim][0]
        dummy_cut = dummy[{dim: slice(1, None)}]
        out = xr.concat([da, dummy_cut], dim=dim)
    else:
        da.coords[dim] = da_full[dim][-1]
        dummy_cut = dummy[{dim: slice(0, -1)}]
        out = xr.concat([dummy_cut, da], dim=dim)
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
        raise RuntimeError("bounds can only have 1 or two elements")

    comp_name = "composite"
    zones = [
        index >= bounds[1],
        np.logical_and(index < bounds[1], index >= bounds[0]),
        index < bounds[0],
    ]
    zones_coords = ["high", "neutral", "low"]
    out = xr.concat([data.where(z) for z in zones], comp_name)
    out[comp_name] = zones_coords
    counts = np.array([a.sum().data for a in zones])
    out.coords["counts"] = xr.DataArray(counts, coords=[out[comp_name]])
    out.attrs["IndexName"] = index.name
    out.attrs["CompositeBounds"] = bounds

    return out


def corrmap(
    a,
    b,
    shifts=0,
    a_x_dim="i",
    a_y_dim="j",
    a_x_coord=None,
    a_y_coord=None,
    b_x_dim="i",
    b_y_dim="j",
    b_x_coord=None,
    b_y_coord=None,
    t_dim="time",
    debug=True,
):
    """
    a -- input
    b -- target ()

    TODO
    This thing is slow. I can most likely rewrite this with \
    numpy.apply_along_axis


    """

    from scipy.stats import linregress

    if not type(a_x_coord) == type(a_y_coord):
        raise RuntimeError("a_x_coord and a_y_coord need to be the same type")

    if not type(b_x_coord) == type(b_y_coord):
        raise RuntimeError("a_x_coord and a_y_coord need to be the same type")

    if isinstance(shifts, int):
        shifts = [shifts]

    # determine if the timseries is a timeseries or a 3d array
    if len(b.shape) == 3:
        arrayswitch = True
    elif len(b.shape) == 1:
        arrayswitch = False
    else:
        raise RuntimeWarning(
            "this only works with a timseries \
            or map of timeseries"
        )

    # shift timeseries
    slope = []
    corr = []
    p_value = []

    for sh, shift in enumerate(shifts):
        shifted_b = b.shift(time=shift)

        s = a.mean(dim=t_dim).copy()
        s[:] = np.nan
        s.name = a.name + " regressed onto " + b.name

        c = a.mean(dim=t_dim).copy()
        c[:] = np.nan
        c.name = "Corr coeff " + a.name + "/" + b.name

        p = a.mean(dim=t_dim).copy()
        p[:] = np.nan
        p.name = "p value " + a.name + "/" + b.name

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
                        in_b = xr.DataArray(
                            shifted_b.data,
                            coords={
                                "xdim": shifted_b[b_x_dim].data,
                                "ydim": shifted_b[b_y_dim].data,
                                "time": shifted_b.time.data,
                            },
                            dims=["time", "ydim", "xdim"],
                        )
                    else:
                        raise RuntimeError("Not implemented yet")
                        # This would have to be acomplished by a mask of some
                        # sort
                        # (with some tolerance around the input position)

                    # extract the matching timeseries
                    in_b = in_b.sel(xdim=in_x, ydim=in_y, method="nearest")
                    reindexed_b = in_b.reindex_like(in_a.time, method="nearest")
                else:
                    reindexed_b = shifted_b.reindex_like(in_a.time, method="nearest")

                x = reindexed_b.data
                y = in_a.data

                idx = np.logical_and(~np.isnan(y), ~np.isnan(x))
                if y[idx].size:
                    (
                        s[{a_x_dim: ii, a_y_dim: jj}],
                        _,
                        c[{a_x_dim: ii, a_y_dim: jj}],
                        p[{a_x_dim: ii, a_y_dim: jj}],
                        _,
                    ) = linregress(x[idx], y[idx])
        slope.append(s)
        corr.append(c)
        p_value.append(p)

    out_s = xr.concat(slope, "timeshifts")
    out_s["timeshifts"] = shifts
    # !!! I think this is a bug...this should be
    # possible with
    out_c = xr.concat(corr, "timeshifts")
    out_c["timeshifts"] = shifts
    out_p = xr.concat(p_value, "timeshifts")
    out_p["timeshifts"] = shifts

    return out_c, out_p, out_s


def concat_dim_da(data, name):
    """creates an xarray.Dataarray to label the concat dim in xarray.concat.
    data is the dimension array and name is the name (DuHHHHH)"""
    return xr.DataArray(data, dims=[name], coords={name: (name, data)}, name=name)


def xr_detrend(b, dim="time", trend_params=None, convert_datetime=True):
    """Removes linear trend along dimension `dim` from dataarray `b`.
    If no `trend_params` are passed (default),
    the linear trend is calculated using `xr_linregress`.
    Parameters
    ----------
    b : {xr.DataArray, xr.Dataset}
        Data source to be detrended.
    dim : str
        Dimension along which to remove linear trend
    trend_params: {xr.DataArray, xr.Dataset, None}
        Precomputed output of xr_linregress.
        This can be usefull for large datasets where intermediate results are
        saved already. Defaults to None, meaning the linear trend is computed
        within the function.
    convert_datetime: bool
        If true (default), the dimension `dim` is converted from a datetime to
        float.
    """
    if convert_datetime:
        t_data = b[dim].astype(np.float64)
    else:
        t_data = b[dim]

    if trend_params is None:
        out = xr_linregress(t_data, b)
    else:
        out = trend_params

    # Create new time dataarray
    trend_full = t_data * out.slope + out.intercept
    trend_full = trend_full.assign_coords({dim: b[dim].data})
    return b - trend_full


def lag_and_combine(ds, lags, dim="time"):
    """Creates lagged versions of the input object,
    combined along new `lag` dimension.
    NOTE: Lagging produces missing values at boundary. Use `.fillna(...)`
    to avoid problems with e.g. xr_linregress.

    Parameters
    ----------
    ds : {xr.DataArray, xr.Dataset}
        Input object
    lags : np.Array
        Lags to be computed and combined. Values denote number of timesteps.
        Negative lag indicates a shift backwards (to the left of the axis).
    dim : str
        dimension of `ds` to be lagged

    Returns
    -------
    {xr.DataArray, xr.Dataset}
        Lagged version of `ds` with additional dimension `lag`

    """

    datasets = []
    for ll in lags:
        datasets.append(ds.shift(**{dim: ll}))
    return xr.concat(datasets, dim=concat_dim_da(lags, "lag"))


def sign_agreement(da, ds_ref, dim, threshold=0.75, mask=True, count_nans=True):
    """[summary]

    Parameters
    ----------
    da : xr.DataArray
        Input data
    ds_ref : xr.DataArray
        Reference data to compare the sign to . E.g. a mean over `dim`
    dim : str
        Dimension of `da` over which the sign agreement is evaluated
    threshold : float, optional
        The minimum fraction of elements that have to agree along `dim`, by default 0.75 (75%)
    mask : bool, optional
        If True, datapoints with all nan values along `dim` get masked out in the output, by default True
    count_nans : bool, optional
        If True, nans along `dim` are counted towards the threshold. If False sign agreement is
        calculated according to non-nan values only, by default True

    """
    if mask:
        mask_data = np.isnan(da).all(dim)
    if count_nans:
        ndim = len(da[dim].data)
    else:
        ndim = (~np.isnan(da)).sum(dim)
    sign_agreement = (np.sign(da) == np.sign(ds_ref)).sum(dim) >= (threshold * ndim)
    if mask:
        sign_agreement = sign_agreement.where(~mask_data)
    return sign_agreement


def mask_mixedlayer(
    ds,
    mld,
    mask="outside",
    z_dim="lev",
    z_bounds="lev_bounds",
    ref_var=None,
    bound_dim="bnds",
):
    """
    Remove all values from input data `ds` that are above the depth defined by `mld`.
    If cell bounds are given in the input data, the selection is more accurate, otherwise
    masking will be perfomed based on cell center values.

    Parameters
    ----------
    ds : xr.Dataset
        Input data
    mld : xr.Dataarray
        Mixed Layer Depth input
    mask : str, optional
        Switch that determines if values outside (`outside`) or (`inside`) are preserved by the masking
    z_dim : str, optional
        Depth dimension of `ds`, by default "lev"
    z_bounds : str, optional
        Cell bounds coordinates along `z_dim`, by default "lev_bounds"
    ref_var : str, optional
        Reference variable to broadcast against, by default None

    Returns
    -------
    xr.Dataset
        `ds` with mixed layer values replaced by missing values
    """
    if ref_var is None:
        ref_var = list(ds.data_vars)[0]

    # broadcast the mld against a full 3d variable to adjust the chunks...
    mld = xr.ones_like(ds[ref_var]) * mld

    crit_name = z_dim
    if z_bounds in ds.coords:
        crit = ds[z_bounds].isel({bound_dim: 1})
        crit_name = z_bounds
        # The proper way to select.
        # This excludes all cells that have an upper bound bound shallower than the mld
    else:
        warnings.warn(
            "Cell bounds [{z_bounds}] not found in input. Masking is performed with cell centers, which might be less accurate"
        )
        crit = ds[z_dim]
        # Fallback. Use the center depth of the cell.
        # Could still be influenced by ML, but probably not that bad.

    if mask == "outside":
        out = ds.where(mld < crit)
    elif mask == "inside":
        out = ds.where(mld >= crit)
    else:
        raise ValueError("`mask` has to be either `inside` or `outside`")
    out.attrs.update({"mixed_layer_values_removed_based_on": crit_name})
    if z_bounds in ds.coords:
        out = out.assign_coords({z_bounds: ds[z_bounds]})
    return out


def remove_bottom_values(ds, dim="lev", fill_val=-1e10):
    """Remove the deepest values that are not nan along the dimension `dim`"""
    # for now assume that values of `dim` increase along the dimension
    if ds[dim][0] > ds[dim][-1]:
        raise ValueError(
            f"It seems like `{dim}` has decreasing values. This is not supported yet. Please sort before."
        )
    else:
        dim_broadcasted = ((ds * 0) + 1) * ds[
            dim
        ]  # broadcast and mask missing values according to each variable in the dataset
        # fill with very large negative value
        bottom_layer_val = dim_broadcasted.fillna(fill_val).max(dim=dim)

        # now mask, by only retaining the values that are smaller than the bottom value.
        # this takes also care of columns that have all the same values or are all nan
        ds_masked = ds.where(dim_broadcasted < bottom_layer_val)
        return ds_masked


##################
# Refactored stuff


def filter_1D(data, std, dim="time", dtype=None):
    warnings.warn(
        "This version of 1D filter is outdated. \
        Please import from xarrayutils.filtering",
        DeprecationWarning,
    )
    return filter_1D_refactored(data, std, dim="time", dtype=None)
