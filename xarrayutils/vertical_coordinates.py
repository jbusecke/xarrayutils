import xarray as xr
import numpy as np
from scipy import interpolate


def _strip_dim(da, dim):
    """if `dim` has coordinate values drop them"""
    if dim in da.coords:
        # da = da.drop_vars(
        #     dim
        # )  # this doesnt work with xarray <0.14, reacativate once dependencies are updataed
        da = da.drop(dim)  # th
    return da


def conservative_remap(
    data,
    z_bnds_source,
    z_bnds_target,
    z_dim="z",
    z_dim_target="remapped",
    z_target=None,
    z_bnd_dim="z_bounds",
    z_bnd_dim_target="z_bounds",
    mask=False,
    debug=False,
):
    """Conservatively remap `data` array from depth cells bound by `z_bnds_source` to depths bound by `z_bnds_target`

    Parameters
    ----------
    data : xr.Dataarray
        Input data on source coordinate system. The values need to be in units/depth, integrated quantities wont work (  I think... )
    z_bnds_source : xr.Dataarray
        Vertical cell bounds of the source coordinate system.
    z_bnds_target : xr.Dataarray
        Vertical cell bounds for the target coordinate system. !Needs to cover the full range of `z_bnds_source` to conserve depth integral.
    z_dim : str
        Dimension of `data` that corresponds to depth. Defaults to `'z'`
    z_dim_target: str
        Dimension of returned data array corresponding to depth. Defaults to `'remapped'`.
    z_target: xr.DataArray
        Coordinate values of remapped depth dimensions. Defaults to `None`, which infers cell center values from cell bounds.
    z_bnd_dim: str
        As `z_dim` but for `z_bnds_source` instead of `data`. ! The dimension length needs to be +1 for the bounds
    z_bnd_dim_target: str
        As `z_dim` but for `z_bnds_target` instead of `data`. ! The dimension length needs to be +1 for the bounds.
    mask: bool
        Optional masking of completely empty cells. Will produce nans in the target cells, which do not overlap with any source cells.
        Defaults to `False`, which produces zeros in these cells

    Returns
    -------
    xr.Dataarray
        Remapped data. Has the dimensions of data, but `z_dim` is replaced with `z_dim_target`.

    """

    # TODO: auto detect the dim names, when 1d arrays are provide
    # TODO: how to deal with different depth conventions most elegantly...
    # TODO: put out values for new z_dimension!
    # if (z_bnds_source < 0).any() or (z_bnds_target < 0).any():
    #     raise ValueError("At the moment only positive depth values are supported.")

    # STEP1: Homogenize the depth dimension naming to unambigous internal naming scheme
    data = data.rename({z_dim: "z"})
    z_bnds_source = z_bnds_source.rename({z_bnd_dim: "z_bnd_src"})
    z_bnds_target = z_bnds_target.rename({z_bnd_dim_target: "z_bnd_tar"})

    # STEP2: Strip out the coordinate values of the depth dimensions.
    # With this we avoid matching on different values later. We operate strictly on
    # logical indicies from here on.
    data = _strip_dim(data, "z")
    z_bnds_source = _strip_dim(z_bnds_source, "z_bnd_src")
    z_bnds_target = _strip_dim(z_bnds_target, "z_bnd_tar")

    # STEP3: Now select the upper and lower cell boundaries and rename to a common
    # depth dimension.
    z_up = z_bnds_source.isel(z_bnd_src=slice(0, -1)).rename({"z_bnd_src": "z"})
    z_down = z_bnds_source.isel(z_bnd_src=slice(1, None)).rename({"z_bnd_src": "z"})

    z_up_tar = z_bnds_target.isel(z_bnd_tar=slice(0, -1)).rename({"z_bnd_tar": "z_tar"})
    z_down_tar = z_bnds_target.isel(z_bnd_tar=slice(1, None)).rename(
        {"z_bnd_tar": "z_tar"}
    )

    # STEP4: Compute bounding depth of cell intersection for each possible
    # combination of depth cells.
    bound_up = xr.ufuncs.maximum(z_up, z_up_tar)
    bound_down = xr.ufuncs.minimum(z_down, z_down_tar)

    if debug:
        print("bound up", bound_up)

    # STEP5: Calculate intersection cell depth
    # all negative values indicate that the cells do not overlap and are
    # replaced with zero
    delta_z = bound_down - bound_up
    if debug:
        print("delta_z", delta_z)

    delta_z = delta_z.where(delta_z > 0, 0)

    # calculate the target grid dz
    delta_z_tar = z_down_tar - z_up_tar

    # the weights for each cell are the partial dz of the source cell (delta_z) divided by the
    # target grid dz for each possible combination
    w = delta_z / delta_z_tar
    # when using the regridding function, the input cell thickness can be 0
    # at the boundary values are repeated. Set resulting nans to 0
    w = w.where(np.isfinite(w), 0)

    # The dot product of the data with the weight matrix gives the remapped values on the new grid cells.
    new_data = xr.dot(data, w, dims="z")  # .drop("z")

    # infer new depth interval data from input if not given
    # if z_target is None:
    #     z_target = 0.5 * (z_up_tar + z_down_tar)
    # else:
    #     pass
    # new_data.coords["z_tar"] = z_target

    # mask for values that have no input ()
    if mask:
        nanmask = (w == 0).all("z")
        new_data = new_data.where(~nanmask)

    return new_data.rename({"z_tar": z_dim_target})


def _regular_interp(x, y, target_values):
    # remove all nans from input
    idx = np.logical_or(np.isnan(x), np.isnan(y))
    x = x[~idx]
    y = y[~idx]

    # replace nans in target_values with out of bound Values
    target_values = np.where(~np.isnan(target_values), target_values, np.nanmax(x) + 1)

    interpolated = interpolate.interp1d(x, y, bounds_error=False)(target_values)
    return interpolated


def linear_interpolation_remap(
    z, data, z_regridded, z_dim=None, z_regridded_dim="regridded", output_dim="remapped"
):
    # infer dim from input
    if z_dim is None:
        if len(z.dims) != 1:
            raise RuntimeError(
                "if z_dim is not specified, \
                               x must be a 1D array."
            )
        dim = z.dims[0]
    else:
        dim = z_dim

    # if dataset is passed drop all data_vars that dont contain dim
    if isinstance(data, xr.Dataset):
        raise ValueError("Dataset input is not supported yet")
        # TODO: for a datset input just apply the function for each appropriate array

    kwargs = dict(
        input_core_dims=[[dim], [dim], [z_regridded_dim]],
        output_core_dims=[[output_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[data.dtype],
        output_sizes={output_dim: len(z_regridded[z_regridded_dim])},
    )
    remapped = xr.apply_ufunc(_regular_interp, z, data, z_regridded, **kwargs)

    remapped.coords[output_dim] = z_regridded.rename(
        {z_regridded_dim: output_dim}
    ).coords[output_dim]
    return remapped


def _coord_interp(z, data, target_values, pad_left=None, pad_right=None):
    """Remap dataarray onto new dimension. E.g. express z(data) as z(target_value)
    using interpolation. Out of range values are padded with `pad_left` and
    `pad_right` assuming that z is sorted in ascending order!"""
    # check amount of nans in input
    idx = np.logical_or(np.isnan(data), np.isnan(z))

    if sum(~idx) < 2:
        z_regridded = target_values * np.nan
        # Maybe I should replace these with padding values as well?
    else:
        # remove all nans from input
        z = z[~idx]
        data = data[~idx]

        z_regridded = interpolate.interp1d(data, z, bounds_error=False)(target_values)

        if (pad_left is not None) and (pad_right is not None):
            # create index for target_values that are out of range of the data
            idx_range_max = (
                target_values >= data.max()
            )  # =, because if the value is exactly the same this will still not preserve the cell depth
            idx_range_min = target_values <= data.min()
            # then figure out which side of padding to assign
            if data[-1] - data[0] > 0:
                # values increase with depth, and depth is assumed to increase to the pad_right
                idx_range_right = idx_range_max
                idx_range_left = idx_range_min
            else:
                idx_range_right = idx_range_min
                idx_range_left = idx_range_max

            z_regridded[idx_range_right] = pad_right
            z_regridded[idx_range_left] = pad_left
    return z_regridded


def linear_interpolation_regrid(
    z,
    data,
    target_values,
    z_bounds=None,
    z_dim=None,
    target_value_dim="target",
    output_dim="regridded",
    z_bounds_dim=None,
):
    # if dataset is passed drop all data_vars that dont contain dim
    if isinstance(data, xr.Dataset):
        raise ValueError("Dataset input is not supported yet")
        # TODO: for a datset input just apply the function for each appropriate array

    # infer dim from input
    if z_dim is None:
        if len(z.dims) != 1:
            raise RuntimeError(
                "if z_dim is not specified, \
                               x must be a 1D array."
            )
        dim = z.dims[0]
    else:
        dim = z_dim

    # pick the padding values from the z_bounds (not z, otherwise remapping will
    # not conserve tracer mass)
    pad_left = None
    pad_right = None
    if z_bounds is not None:
        if z_bounds_dim is None:
            raise ValueError(
                "When `z_bounds` is given, `z_bounds_dim` has to be specified"
            )
        else:
            pad_left = z_bounds[{z_bounds_dim: 0}]
            pad_right = z_bounds[{z_bounds_dim: -1}]

    kwargs = dict(
        input_core_dims=[[dim], [dim], [target_value_dim], [], []],
        output_core_dims=[[output_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[data.dtype],
        output_sizes={output_dim: len(target_values)},
    )
    regridded = xr.apply_ufunc(
        _coord_interp, z, data, target_values, pad_left, pad_right, **kwargs
    )
    regridded.coords[output_dim] = target_values.rename(
        {target_value_dim: output_dim}
    ).coords[output_dim]

    return regridded


# TODO : High level wrapper function `rebinning` that takes data, depth , target
# and has options to use linear interpolation or conservative as remapping function.
# maybe also provide option to choose from nearest value (index) or linear Interpolation
# for the regridding (not sure if that makes sense?). This should be able to internally
# figure out target bin centers from bounds and vice versa (with options to privide input)
#
#
#
# def coord_re(x, y, y_target, remap, x_dim=None, remap_dim=None):
#     """
#     remaps datasets/dataarray `y` with coordinate `x` to new coordinate
#     `y_target` with values specified in `remap`
#     E.g. a dataset with coordinate depth(x) will be remapped to coordinate
#     temp(y_target) as a vertical coordinate, with spacing given by `remap`)
#
#     Parameters
#     ----------
#
#     x: xr.DataArray
#         The original dim/coordinate used for remapping
#
#     y: {xr.DataArray, xr.Dataset}
#         the data to be remapped
#
#     y_target: xr.DataArray
#         The new coordinate used for remapping
#
#     remap: {range, np.array, xr.DataArray}
#         Values of `y_target` used as new coordinate.
#
#     Returns
#     -------
#     remapped_y: xr.Dataset
#         dataset with remapped variables of y and the remapped position of
#         x (e.g. depth of the temperature values given in remap)
#     """
#
#     # infer dim from input
#     if x_dim is None:
#         if len(x.dims) != 1:
#             raise RuntimeError(
#                 "if x_dim is not specified, \
#                                x must be a 1D array."
#             )
#         dim = x.dims[0]
#     else:
#         dim = x_dim
#
#     if remap_dim is not None:
#         raise RuntimeError("multidim remap is not implemented yet.")
#
#     # if dataset is passed drop all data_vars that dont contain dim
#     if isinstance(y, xr.Dataset):
#         drop_vars = [a for a in y.data_vars if dim not in y[a].dims]
#         if drop_vars:
#             print(
#                 "Found incompatible data variables (%s) in dataset. They do \
#             not contain the dimension `%s` and will be dropped."
#                 % (drop_vars, dim)
#             )
#             y = y.drop(drop_vars)
#
#     # convert remap to dataarray?
#     if not isinstance(remap, xr.DataArray):
#         remap = xr.DataArray(remap, coords=[("remapped_dim", remap)])
#
#     args = (y_target, remap)
#     kwargs = dict(
#         input_core_dims=[[dim], [dim], [dim], ["remapped_dim"]],
#         output_core_dims=[["remapped_dim"]],
#         vectorize=True,
#         dask="parallelized",
#         output_dtypes=[y_target.dtype],
#         output_sizes={"remapped_dim": len(remap)},
#     )
#
#     remapped_y = xr.apply_ufunc(_coord_remapping_interp, x, y, *args, **kwargs)
#
#     remapped_pos = xr.apply_ufunc(_coord_remapping_interp, x, x, *args, **kwargs)
#     remapped_y.coords["remapped_%s" % x.name] = remapped_pos
#     return remapped_y
#
#
# def extract_surf(
#     da_target,
#     da_ind,
#     surf_val,
#     dim,
#     masking=False,
#     method="index",
#     fill_value=-1e15,
#     **kwargs
# ):
#     """Extract a surface and surface position out of `da_target`.
#     The surface is defined by lookup of `surf_val` along dimension `dim` in
#     `da_target`.
#
#     Parameters
#     ----------
#     da_target : {xr.DataArray, xr.Dataset}
#         Description of parameter `da_target`.
#     da_ind : xr.DataArray
#         Description of parameter `da_ind`.
#     surf_val : {xr.DataArray, float, {'min', 'max'}}
#         Value of surface to be extracted.
#     dim : str
#         Dimension if `da_ind` along which to extract surface.
#     masking : bool
#         If True, masks values that are at the first or last value of `dim`.
#     method : {'index'}
#         Method to find surface. Either through indexing or interpolation.
#     fill_value : float
#         Value used to pad missing values.
#         Should be well outside of the data range (the default is -1e15).
#     **kwargs :
#          Unused atm.
#
#     Returns
#     -------
#     target_on_surf : {xr.DataArray, xr.Dataset}
#         `da_target` on the extracted surface
#     dim_on_surf : xr.DataArray
#         position of the surface on `dim`
#     """
#     # !!!TODO: The naming is ambiguous...change
#     # da_ind cannot be a dataset
#     if not isinstance(da_ind, xr.DataArray):
#         raise RuntimeError("`da_ind` must be a DataArray.")
#
#     # check if dim is in all dataarrays
#     for ds_check in [da_ind, da_target]:
#         if dim not in list(ds_check.dims):
#             raise RuntimeError("no dimension %s found in input arrays" % dim)
#
#     if isinstance(da_target, xr.DataArray):
#         if not set(da_ind.dims).issubset(set(da_target.dims)):
#             raise RuntimeError("da_target has non matching dimensions.")
#     elif isinstance(da_target, xr.Dataset):
#         # all datavariable have to have all the dimensions of da_ind
#         non_matching_vars = []
#         for vv in list(da_target.data_vars):
#             if not set(da_ind.dims).issubset(set(da_target[vv].dims)):
#                 non_matching_vars.append(vv)
#         if non_matching_vars:
#             da_target = da_target.drop(non_matching_vars)
#             print(
#                 "`da_target` contains variables with non matching dimension. \
#                   %s have been dropped "
#                 % non_matching_vars
#             )
#     else:
#         raise RuntimeError("da_target needs to be xarray DataArray or Dataset")
#
#     if surf_val == "min":
#         surf_val = da_ind.min(dim)
#     elif surf_val == "max":
#         surf_val = da_ind.max(dim)
#     else:
#         if type(surf_val) not in [xr.DataArray, int, float]:
#             raise ValueError(
#                 "`surf_val needs to be a scalar, xr.DataArray (with matching dimensions) \
#             or one of 'min'/'max'"
#             )
#
#     # Mask out areas where the surface runs into the boundary
#     da_ind = da_ind.copy()
#     da_target = da_target.copy()
#
#     if masking:
#         condition = xr.ufuncs.logical_or(
#             (da_ind.max(dim) < surf_val), (da_ind.min(dim) > surf_val)
#         )
#
#     if method == "index":
#         # fill index array, since otherwise armin does raise ValueError
#         da_ind = da_ind.fillna(fill_value)
#         idx = abs(da_ind - surf_val.fillna(0)).argmin(dim)
#
#         # if the idx data is a dask array it needs to be loaded
#         if isinstance(idx.data, Array):
#             idx = idx.load()
#         target_on_surf = da_target[{dim: idx}]
#         dim_on_surf = da_target[dim][{dim: idx}]
#         # remove all values where the dim_on_surf is equal to the fill_value
#         target_on_surf = target_on_surf.where(dim_on_surf != fill_value)
#         dim_on_surf = dim_on_surf.where(dim_on_surf != fill_value)
#     else:
#         print("No other methods implemented yet. Interpolation is coming soon")
#
#     # Mask out the regions where the surface outcrops at the top or bottom
#     if masking:
#         target_on_surf = target_on_surf.where(~condition)
#         dim_on_surf = dim_on_surf.where(~condition)
#
#     return target_on_surf, dim_on_surf
#
#
#
# # TODO: Deprecate this function
# def extract_surf_legacy(
#     da_ind,
#     da_target,
#     surf_val,
#     dim,
#     constant_dims=["time"],
#     fill_value=1000,
#     masking=True,
# ):
#     """Extract values of 'da_target' on a surface in 'da_ind', specified as nearest
#     value to 'surf_val along 'dim'"""
#     # Mask out areas where the surface runs into the boundary
#     da_ind = da_ind.copy()
#     da_target = da_target.copy()
#     if masking:
#         condition = xr.ufuncs.logical_or(
#             (da_ind.max(dim) < surf_val), (da_ind.min(dim) > surf_val)
#         )
#         if constant_dims:
#             condition = condition.any(constant_dims)
#
#     da_ind_filled = da_ind.fillna(fill_value)
#     if isinstance(surf_val, float) or isinstance(surf_val, int):
#         surf_val_filled = surf_val
#     else:
#         surf_val_filled = surf_val.fillna(fill_value).copy()
#
#     ind = find_surf_ind(da_ind_filled, surf_val_filled, dim)
#     # Expand ind into full dimensions
#     ind_exp = (da_ind_filled * 0) + ind
#
#     target = (da_target[dim] * 0) + range(len(da_target[dim]))
#     target_exp = target + (da_ind_filled * 0)
#     target_pos = da_target[dim] + (da_ind_filled * 0)
#
#     found_ind = target_exp == ind_exp
#
#     # Mask out the regions where the surface outcrops at any point over
#     # 'constant_dims'
#     if masking:
#         da_target = da_target.where(~condition)
#         target_pos = target_pos.where(~condition)
#
#     surf = da_target.where(found_ind)
#     surf_pos = target_pos.where(found_ind)
#
#     surf_out = surf.mean(dim)
#     pos_out = surf_pos.mean(dim)
#     if masking:
#         pos_out = pos_out.where(~xr.ufuncs.isnan(da_ind[{dim: 0}]))
#     return surf_out, pos_out
