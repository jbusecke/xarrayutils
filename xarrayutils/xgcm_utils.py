# TODO:
# - infer_gridtype fails with outer (need to figure out if there are grids,
# that put the velocity on the outer position and ajust logic...)
import xarray as xr


def _get_name(coord):
    """Gets name from coord if xr.DataArray is passed."""
    # It seems the content of grid.axes[..].coords has changed from a name
    # to the actual DataArray. This is just for backwards compatibility.
    if isinstance(coord, xr.DataArray):
        return coord.name
    elif isinstance(coord, str):
        return coord
    else:
        raise ValueError(
            "coord input not recognized.\
         Needs to be xr.DataArray or str. Is %s"
            % (type(coord))
        )


def _get_axis_dim(grid, axis, da):
    co = grid.axes[axis].coords
    return [k for k, v in co.items() if _get_name(v) in da.dims][0]


def _infer_gridtype(grid, u, v, verbose=False):
    """Infer Grid type (https://en.wikipedia.org/wiki/Arakawa_grids).
    Currently supports B and C grids"""
    u = u.copy()
    v = v.copy()

    u_x_pos = _get_axis_dim(grid, "X", u)
    u_y_pos = _get_axis_dim(grid, "Y", u)

    v_x_pos = _get_axis_dim(grid, "X", v)
    v_y_pos = _get_axis_dim(grid, "Y", v)

    # should I check if each of these has more than one element?
    if any(
        [a in ["outer", "inner"] for a in [u_x_pos, u_y_pos, v_x_pos, v_y_pos]]
    ):
        raise RuntimeError(
            "`inner` or `outer` grid positions are not supported yet."
        )

    if verbose:
        print(
            "Found: (u @X=%s,Y=%s and v @X=%s,Y=%s)"
            % (u_x_pos, u_y_pos, v_x_pos, v_y_pos)
        )

    if ((u_x_pos == "right") and (u_y_pos == "right")) and (
        (v_x_pos == "right") and (v_y_pos == "right")
    ):
        gridtype = "B"
    elif (u_x_pos == "right" and u_y_pos == "center") and (
        v_x_pos == "center" and v_y_pos == "right"
    ):
        gridtype = "C"
    else:
        raise RuntimeError(
            "Gridtype not recognized. \
        Currently only supports \
        B-grids(u @X=right,Y=right and v @X=right,Y=right) and \
        C-grids (u @X=right,Y=center and v @X=center,Y=right). \
        Found: (u @X=%s,Y=%s and v @X=%s,Y=%s)"
            % (u_x_pos, u_y_pos, v_x_pos, v_y_pos)
        )
    return gridtype


def _check_dims(a, b, a_name):
    if not all([dd in a.dims for dd in b.dims]):
        raise RuntimeError(
            "%s does not have the appropriate dimensions. \
            Expected %s, but found %s"
            % (a_name, list(b.dims), list(a.dims))
        )
    else:
        return True


def calculate_rel_vorticity(grid, u, v, dx, dy, area, gridtype=None):
    """Calculate the relative vorticity `zeta` (dv/dx - du/dy) on b and c grids.
    Parameters
    ----------
    grid : xgcm.Grid
        xgcm grid object
    u : xr.DataArray
        Zonal velocity
    v : xr.DataArray
        Meridional velocity
    dx : xr.DataArray
        zonal distance centered at u grid location
    dy : xr.DataArray
        meridional distance centered at v grid location
    area : xr.DataArray
        Cell area of the resulting vorticity position (B-grid: tracer position;
        C-grid: north-east tracer cell corner)
    gridtype : str, None
        Arakawa grid layout. Supports `B` and `C` grids
        (the default is None, detecting the layout from `u` and `v` dims)
    Returns
    -------
    xr.DataArray
        DataArray of zeta values.
    """
    u = u.copy()
    v = v.copy()
    dx = dx.copy()
    dy = dy.copy()
    # infer Gridtype
    if gridtype is None:
        gridtype = _infer_gridtype(grid, u, v)

    # convert u and v to total mass flux using the cell face distances
    _check_dims(u, dx, "dx")
    u_int = u * dx
    _check_dims(v, dy, "dy")
    v_int = v * dy

    # always first remap and then difference in the 'target' cell.
    # For c grid this is good, for b grid remap first
    if gridtype == "B":
        u_int = grid.interp(u_int, "X")
        v_int = grid.interp(v_int, "Y")

    dx_v = grid.diff(v_int, "X")
    dy_u = grid.diff(u_int, "Y")

    _check_dims(dx_v, area, "area and dv/dx")
    _check_dims(dy_u, area, "area and du/dy")
    zeta = (dx_v - dy_u) / area

    # add attributes
    zeta.name = "relative vorticity"

    return zeta


# Write xgcm reinterpolation routine


def interp_all(grid, ds, target="center"):
    """Interpolates all variables and coordinates in `ds` onto common dimensions,
    specified by target."""

    ds = ds.copy()
    ds_new = xr.Dataset()

    def _core_interp(da, grid):
        for ax in grid.axes.keys():
            # Check if any dimension matches this axis
            ax_coords = [_get_name(a) for a in grid.axes[ax].coords.values()]
            match = [a for a in da.dims if a in ax_coords]
            if len(match) > 0:
                pos = [
                    p
                    for p, a in grid.axes[ax].coords.items()
                    if _get_name(a) in match
                ]
                if target not in pos:
                    da = grid.interp(da, ax, to=target)
        return da

    for vv in ds.data_vars:
        ds_new[vv] = _core_interp(ds[vv], grid)
    return ds_new
