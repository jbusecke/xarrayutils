def derivative(grid, data, axis, debug=False):
    """Calculate gradient along single axis.
    PARAMETERS
    ----------
    grid : xgcm.Grid
    data : xarray.DataArray
        The data to interpolate
    axis: {'X', 'Y'}
        The name of the axis along which to calculate
    RETURNS
    -------
    da_i : xarray.DataArray
        gradient along axis
    """
    delta = grid.diff(data, axis)
    dx = get_dx(grid, delta, axis)

    if dx is None:
        raise RuntimeError(
            "grid distance could not be \
                            extracted check grid input"
        )
    return delta / dx


def gradient(grid, data, interpolate=False, debug=False):
    """compute the gradient in x,y direction.

    PARAMETERS
    ----------
    grid : xgcm.Grid
    data : xarray.DataArray
        The data to interpolate
    axis: {'X', 'Y'}
        The name of the axis along which to calculate

    interpolate : bool, optional
        Should values be interpreted from grid face to center or v
        ice versa.

    RETURNS
    -------
    da_i : xarray.DataArray
        gradient along axis

    TODO
    ----
        Currently only performs a first order forward gradient.
        It could be good to implement different order gradients later
    """

    grad_x = derivative(grid, data, "X", debug=debug)
    grad_y = derivative(grid, data, "Y", debug=debug)

    if interpolate:
        grad_x = grid.interp(grad_x, "X")
        grad_y = grid.interp(grad_y, "Y")

    return grad_x, grad_y


def laplacian(grid, data):
    gradx, grady = gradient(grid, data)
    grad2x, dummy = gradient(grid, gradx)
    dummy, grad2y = gradient(grid, grady)
    return grad2y + grad2x


def gradient_sq_amplitude(grid, data):
    gradx, grady = gradient(grid, data, interpolate=True)
    return gradx**2 + grady**2


# Silly functions
def get_hfac(grid, data):
    # TODO: This is not general enough...need to
    """Figure out the correct hfac given array dimensions."""
    hfac = None
    if "i" in data.dims and "j" in data.dims and "hFacC" in grid._ds:
        hfac = grid._ds.hFacC
    if "i" in data.dims and "j_g" in data.dims and "hFacS" in grid._ds:
        hfac = grid._ds.hFacS
    if "i_g" in data.dims and "j" in data.dims and "hFacW" in grid._ds:
        hfac = grid._ds.hFacW
    return hfac


def get_dx(grid, data, axis):
    """Figure out the correct hfac given array dimensions."""
    dx = None
    if axis == "X":
        if "i" in data.dims and "j" in data.dims and "dxG" in grid._ds:
            dx = grid.interp(grid._ds.dxG, "Y")
        # Is this right or is there a different dxC for the vorticity cell?
        if "i" in data.dims and "j_g" in data.dims and "dxG" in grid._ds:
            dx = grid._ds.dxG

        if "i_g" in data.dims and "j" in data.dims and "dxC" in grid._ds:
            dx = grid._ds.dxC
        # Is this right or is there a different dxC for the vorticity cell?
        if "i_g" in data.dims and "j_g" in data.dims and "dxC" in grid._ds:
            dx = grid.interp(grid._ds.dxC, "Y")

    elif axis == "Y":
        if "i" in data.dims and "j" in data.dims and "dyG" in grid._ds:
            dx = grid.interp(grid._ds.dyG, "X")
        # Is this right or is there a different dxC for the vorticity cell?
        if "i_g" in data.dims and "j" in data.dims and "dyG" in grid._ds:
            dx = grid._ds.dyG

        if "i" in data.dims and "j_g" in data.dims and "dyC" in grid._ds:
            dx = grid._ds.dyC
        # Is this right or is there a different dxC for the vorticity cell?
        if "i_g" in data.dims and "j_g" in data.dims and "dyC" in grid._ds:
            dx = grid.interp(grid._ds.dyC, "X")
    return dx


def matching_coords(grid, dims):
    # Fill in all coordinates from grid that match the new dims
    c = []
    for kk in grid.coords.keys():
        check = list(grid[kk].dims)
        if all([a in dims for a in check]):
            c.append(kk)

    c_dict = dict([])
    for ii in c:
        c_dict[ii] = grid[ii]
    return c_dict
