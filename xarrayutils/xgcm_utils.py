# TODO:
# - infer_gridtype fails with outer (need to figure out if there are grids,
# that put the velocity on the outer position and ajust logic...)
import xarray as xr
from xarrayutils.weighted_operations import weighted_mean


def _get_name(coord):
    """Gets name from coord if xr.DataArray is passed.
    This was an ad-hoc solution, there might be something better
    in xgcm itself"""
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


def _get_axis_pos(grid, axis, da):
    # return the cgrid location along `axis` for `da`
    if axis not in grid.axes.keys():
        return None
    else:
        co = grid.axes[axis].coords
        return [k for k, v in co.items() if _get_name(v) in da.dims][0]


def _find_dim(grid, obj, axis):
    if axis not in grid.axes.keys():
        return None
    else:
        dimlist = list(grid.axes[axis].coords.values())
        dimlist = [_get_name(d) for d in dimlist]
        # this seems clunky, is there a better way to do this with the xgcm
        # internals?
        matches = [d for d in dimlist if d in obj.dims]
        if len(matches) == 0:
            return None
        else:
            return matches


def _infer_gridtype(grid, u, v, verbose=False):
    """Infer Grid type (https://en.wikipedia.org/wiki/Arakawa_grids).
    Currently supports B and C grids"""
    u = u.copy()
    v = v.copy()

    u_x_pos = _get_axis_pos(grid, "X", u)
    u_y_pos = _get_axis_pos(grid, "Y", u)

    v_x_pos = _get_axis_pos(grid, "X", v)
    v_y_pos = _get_axis_pos(grid, "Y", v)

    # should I check if each of these has more than one element?
    if any([a in ["outer", "inner"] for a in [u_x_pos, u_y_pos, v_x_pos, v_y_pos]]):
        raise RuntimeError("`inner` or `outer` grid positions are not supported yet.")

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
    """Checks if all dims of a are found in b"""
    if not all([dd in a.dims for dd in b.dims]):
        raise RuntimeError(
            "%s does not have the appropriate dimensions. \
            Expected %s, but found %s"
            % (a_name, list(b.dims), list(a.dims))
        )
    else:
        return True


def _find_metric(da, dim_metric_list):
    """given a list of dims/coords `dim_metric_list`,
    find the one whos dims match `da`"""
    matches = [m for m in dim_metric_list if m in da.coords]
    if len(matches) > 1:
        raise ValueError(
            "found more than one matching metric(%s), \
                something is wrong with the `metric_list`"
            % matches
        )
    elif len(matches) == 0:
        return None
    else:
        return matches[0]


# Simple Operations using metrics #


def w_mean(grid, dat, axis, dim_metric_list, verbose=False):
    dat = dat.copy()
    dim = _find_dim(grid, dat, axis)
    # if there is no dim found, return the original data
    if dim is None:
        return dat
    else:
        # Proceed with weighted average
        metric = _find_metric(dat, dim_metric_list)
        if verbose:
            print(metric)
        if metric is None:
            # If no corresponding metric is found return the original dataarray
            # (this could throw an error in the future...)
            return dat
        else:
            # I previously used my own implementation, which does some
            # additional broadcasting, but that is slow here.
            # This is faster but I will try to wait for the upstream
            # integration in xarray to finetune this.
            # (dat * dat.coords[metric]).sum(dim) / (
            #    dat.coords[metric].sum(dim)
            # )
            return weighted_mean(dat, dat.coords[metric], dim=dim)


def xgcm_weighted_mean(grid, dat, axis, dim_metric_list, verbose=False):
    # TODO: Should keep attrs and also add details in them about the
    # processing...
    # the dim metric list should be callable from the grid object
    dat = dat.copy()
    if isinstance(dat, xr.Dataset):
        ds_mean = xr.Dataset()
        for vv in dat.data_vars:
            ds_mean[vv] = w_mean(grid, dat[vv], axis, dim_metric_list, verbose=verbose)
    elif isinstance(dat, xr.DataArray):
        ds_mean = w_mean(grid, dat, axis, dim_metric_list, verbose=verbose)
    return ds_mean


# High level vector calculus #


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


# convenience functions #


def interp_all(grid, ds, target="center", keep_coords=True):
    """Interpolates all variables and coordinates in `ds` onto common dimensions,
    specified by target.

    Parameters
    ----------
    grid : xgcm.Grid
        Grid object matching `ds`.
    ds : xr.DataArray or xr.Dataset
        Input data.
    target : str
        Cell position target. See xgcm definitons (the default is "center").
    keep_coords : bool
        Switch to keep all coordinates of the input object
        (the default is True).

    Returns
    -------
    type
        Description of returned object.

    """
    """"""

    ds = ds.copy()
    ds_new = xr.Dataset()

    def _core_interp(da, grid):
        for ax in grid.axes.keys():
            # Check if any dimension matches this axis
            ax_coords = [_get_name(a) for a in grid.axes[ax].coords.values()]
            match = [a for a in da.dims if a in ax_coords]
            if len(match) > 0:
                pos = [
                    p for p, a in grid.axes[ax].coords.items() if _get_name(a) in match
                ]
                if target not in pos:
                    da = grid.interp(da, ax, to=target)
        return da

    for vv in ds.data_vars:
        ds_new[vv] = _core_interp(ds[vv], grid)
    if keep_coords:
        for co in ds.coords:
            if co not in list(ds_new.coords):
                ds_new.coords[co] = ds[co]
    return ds_new


# mappiing/autogenerate stuff #


def dll_dist(dlon, dlat, lon, lat):
    """Converts lat/lon differentials into distances in meters

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

    distance_1deg_equator = 111000.0
    dx = dlon * xr.ufuncs.cos(xr.ufuncs.deg2rad(lat)) * distance_1deg_equator
    dy = ((lon * 0) + 1) * dlat * distance_1deg_equator
    return dx, dy
