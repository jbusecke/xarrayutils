import xarray as xr


def conservative_remap(data, z_bnds_source, z_bnds_target, z_dim='z', z_dim_target='remapped',
          z_bnd_dim='z_bounds', z_bnd_dim_target='z_bounds', mask=False):
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
    # TODO: auto detect the dim names, when 1d arrays are provided?

    # rename dimensions (this is particularly important when both have the same dim name)
    data = data.rename({z_dim:'z'})
    z_up = z_bnds_source[{z_bnd_dim:slice(0,-1)}].rename({z_bnd_dim:'z'})
    z_down = z_bnds_source[{z_bnd_dim:slice(1,None)}].rename({z_bnd_dim:'z'})

    z_up_tar = z_bnds_target[{z_bnd_dim_target:slice(0,-1)}].rename({z_bnd_dim_target:'z_tar'})
    z_down_tar = z_bnds_target[{z_bnd_dim_target:slice(1,None)}].rename({z_bnd_dim_target:'z_tar'})

    # Compute boundind depth of cell intersection for each combination of depth cells.
    bound_up = xr.ufuncs.maximum(z_up,z_up_tar)
    bound_down = xr.ufuncs.minimum(z_down,z_down_tar)

    # Calculate intersection cell depth
    # all negative values indicate that the cells do not overlap and are replaced with zero
    delta_z = bound_down - bound_up
    delta_z = delta_z.where(delta_z > 0, 0)

    # calculate the target grid dz
    delta_z_tar = z_down_tar - z_up_tar

    #the weights for each cell are the partial dz of the source cell (delta_z) divided by the
    # target grid dz for each possible combination
    w = delta_z / delta_z_tar

    # The dot product of the data with the weight matrix gives the remapped values on the new grid cells.
    new_data = xr.dot(data, w, dims='z')

    # mask for values that have no input ()
    if mask:
        nanmask = (w == 0).all('z')
        new_data = new_data.where(~nanmask)

    return new_data.rename({'z_tar':z_dim_target})
