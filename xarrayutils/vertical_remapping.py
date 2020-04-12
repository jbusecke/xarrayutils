import numpy as np
import xarray as xr

#########################
# Density layer version #
#########################
# for later: handle the intervals just like groupby. For now workaround
# Does not handle decreasing values, this is probably related to the above


def _groupby_vert(data, group_data, bins):
    # Replicates the behaviour of xarrays `groupby_bins` along one dimension
    # with numpy
    axis = -1  # xr.apply_ufunc transposes core dims to the end
    layers = []

    for b in range(len(bins) - 1):
        bb = [bins[b], bins[b + 1]]
        # This should be customizable like in groupby
        mask = np.logical_and(bb[0] < group_data, bb[1] >= group_data)
        data_masked = data.copy()
        data_masked[~mask] = np.nan
        nanmask = np.all(~mask, axis=axis)
        # There were some problems with passing the function as func=...
        # kwarg. So for now I will hardcode the solution
        # layer = func(data_masked, axis=axis)
        layer = np.nansum(data_masked, axis=axis)
        # there might be an exeption when this is run on a 1d vector...
        # but for now this works...
        # I formerly did this with ma.masked_array,
        # but somehow that did not work with apply_ufunc

        # special treatment for 1d input arrays
        if not isinstance(layer, np.ndarray):
            layer = np.array(layer)

        layer[nanmask] = np.nan
        layers.append(layer)
    return np.stack(layers, axis=-1)


def xr_1d_groupby(data, group_data, bins, dim):
    """Short summary.

    Parameters
    ----------
    data : type
        Description of parameter `data`.
    group_data : type
        Description of parameter `group_data`.
    bins : type
        Description of parameter `bins`.
    dim : type
        Description of parameter `dim`.
    func : type
        Description of parameter `func` (the default is np.nansum).

    Returns
    -------
    xr.DataArray
        Remapped data
    """
    bin_name = group_data.name
    bin_dim = "%s_layer" % bin_name
    bin_center = (bins[:-1] + bins[1:]) / 2

    name = group_data.name
    if name is None:
        raise ValueError("`group_data` array must have name")

    remapped = xr.apply_ufunc(
        _groupby_vert,
        data,
        group_data,
        bins,
        input_core_dims=[[dim], [dim], ["bins"]],
        output_core_dims=[[bin_dim]],
        dask="parallelized",
        output_dtypes=[data.dtype],
        output_sizes={bin_dim: len(bins) - 1},
    )

    remapped.coords[bin_dim] = bin_center
    remapped.coords[bin_dim + "_lower"] = (bin_dim, bins[:-1])
    remapped.coords[bin_dim + "_upper"] = (bin_dim, bins[1:])
    #     remapped.coords[bin_dim+'_intervals'] = (['']list(zip(,bins[1:]))
    return remapped


###############
# Needs tests #
###############


def xr_remapping(
    da_data, da_group, bins, dim, distance_coord, content_var=False, return_average=True
):
    """Performs conservative remapping into another tracer coordinate system.

    Parameters
    ----------
    da_data : xr.DataArray
        Data array to be remapped.
    da_group : xr.DataArray
        Data array of values to remap onto (e.g. potential density).
    bins : array-like
        Spacing for new coordinates, in units of `da_group`. `da_data` values
        are binned between values of `bins`.
    dim : str
        Dimension along remapping is performed (e.g. depth).
    distance_coord : str
        Name of coordinate in `da_data` which contains distances along `dim`.
        Required to acurately weight data points.
    content_var : bool
        Option for preweighted values. If True, `distance_coord` will not be
        multiplied with `da_data` before binning  (the default is False).
    return_average : bool
        Option to return layer averages (True) or layer integrals (False).

    Returns
    -------
    xr.DataArray
        Remapped data with additional coordinates;
         `{da_group.name}_layer_up/down`(upper/lower bound of remapped layer)
         `{da_group.name}_layer_{distance_coord.name}` (thickness of remapped
                                                        layer)
         `{da_group.name}_layer_{dim}` (mean position of layer along `dim`,
                                        e.g. mean depth of isopycnal layer)

    """
    da_data = da_data.copy()
    da_group = da_group.copy()
    if not (set(da_data.dims) == set(da_group.dims)):
        raise ValueError(
            "`da_data` and `da_group` do not have identical dims. \
            Please interpolate broadcast appropriately before remapping"
        )

    da_thick = da_data.coords[distance_coord].copy()
    da_dim = da_data.coords[dim].copy()

    # make sure that the thickness data is not counted
    # anywhere else but where there is data
    thick_name = da_thick.name  # seems to be overwritten by the line below
    da_thick = da_thick * ((da_data * 0) + 1)
    # Same for the layer position
    da_dim = da_dim * ((da_data * 0) + 1)

    # Weight da_dim and da_data (only for content_var=False) with da_thick
    da_dim = da_dim * da_thick
    if not content_var:
        da_data = da_data * da_thick

    data_remapped = xr_1d_groupby(da_data, da_group, bins, dim)
    thickness = xr_1d_groupby(da_thick, da_group, bins, dim)
    layer_pos = xr_1d_groupby(da_dim, da_group, bins, dim)

    if return_average:
        data_remapped = data_remapped / thickness

    data_remapped.coords["%s_layer_%s" % (da_group.name, thick_name)] = thickness
    # calculate the mean depth of the layer
    data_remapped.coords["%s_layer_%s" % (da_group.name, dim)] = layer_pos / thickness
    data_remapped.name = da_data.name

    return data_remapped
