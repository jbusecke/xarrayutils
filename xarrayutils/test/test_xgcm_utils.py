import pytest

Grid = pytest.importorskip("xgcm.Grid")
import xarray as xr
import numpy as np

from xarray.testing import assert_allclose
from xarrayutils.weighted_operations import weighted_mean
from xarrayutils.xgcm_utils import (
    _infer_gridtype,
    _get_name,
    _get_axis_pos,
    _check_dims,
    _find_metric,
    _find_dim,
    w_mean,
    xgcm_weighted_mean,
    interp_all,
    calculate_rel_vorticity,
    dll_dist,
)


def datasets():
    xt = np.arange(4)
    xu = xt + 0.5
    yt = np.arange(4)
    yu = yt + 0.5
    # add a non x,y variable to test how its handled throughout.
    t = np.arange(10)

    # Need to add a tracer here to get the tracer dimsuffix
    tr = xr.DataArray(
        np.random.rand(len(xt), len(yt), len(t)),
        coords=[("xt", xt), ("yt", yt), ("time", t)],
    )

    u_b = xr.DataArray(
        np.random.rand(len(xt), len(yt), len(t)),
        coords=[("xu", xu), ("yu", yu), ("time", t)],
    )
    v_b = xr.DataArray(
        np.random.rand(len(xt), len(yt), len(t)),
        coords=[("xu", xu), ("yu", yu), ("time", t)],
    )

    u_c = xr.DataArray(
        np.random.rand(len(xt), len(yt), len(t)),
        coords=[("xu", xu), ("yt", yt), ("time", t)],
    )
    v_c = xr.DataArray(
        np.random.rand(len(xt), len(yt), len(t)),
        coords=[("xt", xt), ("yu", yu), ("time", t)],
    )
    # maybe also add some other combo of x,t y,t arrays....
    timeseries = xr.DataArray(np.random.rand(len(t)), coords=[("time", t)])

    # northeast distance
    dx = 0.3
    dy = 2

    dx_ne = xr.DataArray(np.ones([4, 4]) * dx - 0.1, coords=[("xu", xu), ("yu", yu)])
    dx_n = xr.DataArray(np.ones([4, 4]) * dx - 0.2, coords=[("xt", xt), ("yu", yu)])
    dx_e = xr.DataArray(np.ones([4, 4]) * dx - 0.3, coords=[("xu", xu), ("yt", yt)])
    dx_t = xr.DataArray(np.ones([4, 4]) * dx - 0.4, coords=[("xt", xt), ("yt", yt)])

    dy_ne = xr.DataArray(np.ones([4, 4]) * dy + 0.1, coords=[("xu", xu), ("yu", yu)])
    dy_n = xr.DataArray(np.ones([4, 4]) * dy + 0.2, coords=[("xt", xt), ("yu", yu)])
    dy_e = xr.DataArray(np.ones([4, 4]) * dy + 0.3, coords=[("xu", xu), ("yt", yt)])
    dy_t = xr.DataArray(np.ones([4, 4]) * dy + 0.4, coords=[("xt", xt), ("yt", yt)])

    area_ne = dx_ne * dy_ne
    area_n = dx_n * dy_n
    area_e = dx_e * dy_e
    area_t = dx_t * dy_t

    def _add_metrics(obj):
        obj = obj.copy()
        for name, data in zip(
            [
                "dx_ne",
                "dx_n",
                "dx_e",
                "dx_t",
                "dy_ne",
                "dy_n",
                "dy_e",
                "dy_t",
                "area_ne",
                "area_n",
                "area_e",
                "area_t",
            ],
            [
                dx_ne,
                dx_n,
                dx_e,
                dx_t,
                dy_ne,
                dy_n,
                dy_e,
                dy_t,
                area_ne,
                area_n,
                area_e,
                area_t,
            ],
        ):
            obj.coords[name] = data
        # add xgcm attrs
        for ii in ["xu", "xt"]:
            obj[ii].attrs["axis"] = "X"
        for ii in ["yu", "yt"]:
            obj[ii].attrs["axis"] = "Y"
        for ii in ["xu", "yu"]:
            obj[ii].attrs["c_grid_axis_shift"] = 0.5
        return obj

    coords = {
        "X": {"center": "xt", "right": "xu"},
        "Y": {"center": "yt", "right": "yu"},
    }
    coords_outer = {
        "X": {"center": "xt", "outer": "xu"},
        "Y": {"center": "yt", "outer": "yu"},
    }

    ds_b = _add_metrics(
        xr.Dataset({"u": u_b, "v": v_b, "tracer": tr, "timeseries": timeseries})
    )
    ds_c = _add_metrics(
        xr.Dataset({"u": u_c, "v": v_c, "tracer": tr, "timeseries": timeseries})
    )

    ds_fail = _add_metrics(
        xr.Dataset({"u": u_b, "v": v_c, "tracer": tr, "timeseries": timeseries})
    )
    ds_fail2 = _add_metrics(
        xr.Dataset({"u": u_b, "v": v_c, "tracer": tr, "timeseries": timeseries})
    )

    return {
        "B": ds_b,
        "C": ds_c,
        "fail_gridtype": ds_fail,
        "fail_dimtype": ds_fail2,
        "coords": coords,
        "fail_coords": coords_outer,
    }


def test_find_metric():
    datadict = datasets()
    ds = datadict["C"]
    metric_list = ["dx_n", "dx_e", "dx_t", "dx_ne"]
    fail_metric_list = ["dx_n", "dy_n"]
    assert _find_metric(ds["tracer"], metric_list) == "dx_t"
    assert _find_metric(ds["u"], metric_list) == "dx_e"
    assert _find_metric(ds["v"], metric_list) == "dx_n"
    assert _find_metric(ds["u"].drop("dx_e"), metric_list) is None
    with pytest.raises(ValueError):
        _find_metric(ds["v"], fail_metric_list)


def test_find_dim():
    datadict = datasets()
    ds = datadict["C"]
    grid = Grid(ds)
    assert _find_dim(grid, ds, "X") == ["xt", "xu"]
    assert _find_dim(grid, ds, "Z") is None
    assert _find_dim(grid, ds["timeseries"], "X") is None
    assert _find_dim(grid, ds["timeseries"], "X") is None
    assert _find_dim(grid, ds["tracer"], "X") == ["xt"]
    assert _find_dim(grid, ds["u"], "X") == ["xu"]


def test_get_name():
    datadict = datasets()
    ds = datadict["C"]
    assert _get_name(ds.xt) == "xt"


def test_get_axis_pos():
    datadict = datasets()
    ds = datadict["C"]
    coords = datadict["coords"]
    grid = Grid(ds, coords=coords)
    assert _get_axis_pos(grid, "X", ds.u) == "right"
    assert _get_axis_pos(grid, "X", ds.tracer) == "center"
    assert _get_axis_pos(grid, "Z", ds.u) is None


def test_infer_gridtype():
    datadict = datasets()
    coords = datadict["coords"]
    ds_b = datadict["B"]
    grid_b = Grid(ds_b, coords=coords)

    ds_c = datadict["C"]
    grid_c = Grid(ds_c, coords=coords)

    # This should fail(unkown gridtype)
    ds_fail = datadict["fail_gridtype"]
    grid_fail = Grid(ds_fail, coords=coords)

    # This is not supported yet ('inner' and 'outer' dims)
    coords2 = datadict["fail_coords"]
    ds_fail2 = datadict["fail_dimtype"]
    grid_fail2 = Grid(ds_fail2, coords=coords2)

    assert _infer_gridtype(grid_b, ds_b.u, ds_b.v) == "B"
    assert _infer_gridtype(grid_c, ds_c.u, ds_c.v) == "C"
    with pytest.raises(RuntimeError, match=r"Gridtype not recognized *"):
        _infer_gridtype(grid_fail, ds_fail.u, ds_fail.v)
    with pytest.raises(RuntimeError):  # , match=r'`inner` or `outer` *'
        _infer_gridtype(grid_fail2, ds_fail2.u, ds_fail2.v)


def test_check_dims():
    datadict = datasets()
    ds = datadict["C"]
    assert _check_dims(ds.u, ds.u, "dummy")
    with pytest.raises(RuntimeError):
        _check_dims(ds.u, ds.v, "dummy")


@pytest.mark.parametrize(
    "axis, metric_list",
    [
        ("X", ["dx_t", "dx_e", "dx_n", "dx_ne"]),
        ("X", ["dy_t", "dy_e", "dy_n", "dy_ne"]),
    ],
)
@pytest.mark.parametrize("gridtype", ["B", "C"])
def test_w_mean(axis, metric_list, gridtype):
    fail_metric_list = ["fail"]
    ds = datasets()[gridtype]
    grid = Grid(ds)
    for var in ds.data_vars:
        metric = _find_metric(ds[var], metric_list)
        dim = _find_dim(grid, ds[var], axis)
        a = w_mean(grid, ds[var], axis, metric_list, verbose=True)
        if dim is None:  # no dimension found, return the input arrays
            b = ds[var]
        else:
            b = weighted_mean(ds[var], ds[metric], dim=dim)
        assert_allclose(a, b)

        # original array should be returned if a non matching metric list
        # is supplied
        a_fail = w_mean(grid, ds[var], axis, fail_metric_list)
        assert_allclose(a_fail, ds[var])


@pytest.mark.parametrize(
    "axis, metric_list",
    [
        ("X", ["dx_t", "dx_e", "dx_n", "dx_ne"]),
        ("X", ["dy_t", "dy_e", "dy_n", "dy_ne"]),
    ],
)
@pytest.mark.parametrize("gridtype", ["B", "C"])
def test_xgcm_weighted_mean(axis, metric_list, gridtype):
    ds = datasets()[gridtype]
    grid = Grid(ds)
    a = xgcm_weighted_mean(grid, ds, axis, metric_list)
    for var in ["tracer", "u", "v"]:
        b = w_mean(grid, ds[var], axis, metric_list)
        c = xgcm_weighted_mean(grid, ds[var], axis, metric_list)

        assert_allclose(a[var], b)
        assert_allclose(b, c)

    for var in ["timeseries"]:
        b = ds[var]
        c = xgcm_weighted_mean(grid, ds[var], axis, metric_list)

        assert_allclose(a[var], b)
        assert_allclose(b, c)


def test_calculate_rel_vorticity():
    datadict = datasets()
    coords = datadict["coords"]
    ds_b = datadict["B"]
    grid_b = Grid(ds_b, coords=coords)

    ds_c = datadict["C"]
    grid_c = Grid(ds_c, coords=coords)

    test_b = (
        grid_b.diff(grid_b.interp(ds_b.v * ds_b.dy_ne, "Y"), "X")
        - grid_b.diff(grid_b.interp(ds_b.u * ds_b.dx_ne, "X"), "Y")
    ) / ds_b.area_t

    zeta_b = calculate_rel_vorticity(
        grid_b, ds_b.u, ds_b.v, ds_b.dx_ne, ds_b.dy_ne, ds_b.area_t, gridtype=None
    )

    test_c = (
        grid_c.diff(ds_c.v * ds_c.dy_n, "X") - grid_c.diff(ds_c.u * ds_c.dx_e, "Y")
    ) / ds_c.area_ne

    zeta_c = calculate_rel_vorticity(
        grid_c, ds_c.u, ds_c.v, ds_c.dx_e, ds_c.dy_n, ds_c.area_ne, gridtype=None
    )

    assert_allclose(test_b, zeta_b)
    assert_allclose(test_c, zeta_c)
    with pytest.raises(RuntimeError):
        zeta_c = calculate_rel_vorticity(
            grid_b,
            ds_c.u,
            ds_c.v,
            ds_c.dx_n,  # wrong coordinate
            ds_c.dy_n,
            ds_c.area_ne,
            gridtype=None,
        )


def test_interp_all():
    datadict = datasets()
    coords = datadict["coords"]
    ds_b = datadict["B"]
    grid_b = Grid(ds_b, coords=coords)

    ds_c = datadict["C"]
    grid_c = Grid(ds_c, coords=coords)

    for var in ["u", "v", "tracer"]:
        for ds, grid in zip([ds_b, ds_c], [grid_b, grid_c]):
            for target, control_dims in zip(
                ["center", "right"], [["xt", "yt", "time"], ["xu", "yu", "time"]]
            ):
                print(ds)
                print(grid)
                ds_interp = interp_all(grid, ds, target=target)
                assert set(ds_interp[var].dims) == set(control_dims)
                assert set(ds_interp.coords) == set(ds.coords)
                ds_interp_nocoords = interp_all(
                    grid, ds, target=target, keep_coords=False
                )
                assert set(ds_interp_nocoords.coords) != set(ds.coords)


def test_dll_dist():
    lon = np.arange(-180, 180, 10)
    lat = np.arange(-90, 90, 10)
    llon, llat = np.meshgrid(lon, lat)
    dlon = np.diff(llon, axis=1)
    dlat = np.diff(llat, axis=0)

    # lon = lon[1:]
    # lat = lat[1:]
    # llon = llon[1:, 1:]
    # llat = llat[1:, 1:]
    # dlon = dlon[1:, :]
    # dlat = dlat[:, 1:]

    lon = lon[:-1]
    lat = lat[:-1]
    llon = llon[:-1, :-1]
    llat = llat[:-1, :-1]
    dlon = dlon[:-1, :]
    dlat = dlat[:, :-1]

    # convert to xarrays
    da_lon = xr.DataArray(lon, coords=[("lon", lon)])
    da_lat = xr.DataArray(lat, coords=[("lat", lat)])
    print(dlon.shape)
    print(lon.shape)
    print(lat.shape)
    da_dlon = xr.DataArray(dlon, coords=[lat, lon], dims=["lat", "lon"])
    da_dlat = xr.DataArray(dlat, coords=[lat, lon], dims=["lat", "lon"])

    d_raw = 111000.0  # represents the diatance of 1 deg on the Eq in m
    dx_test = dlon * np.cos(np.deg2rad(llat)) * d_raw
    dy_test = dlat * d_raw
    dy_test = dy_test.T

    dx, dy = dll_dist(da_dlon, da_dlat, da_lon, da_lat)
    np.testing.assert_allclose(dx.data, dx_test)
    np.testing.assert_allclose(dx_test[:, 0], dx.data[:, 0])
    np.testing.assert_allclose(dy.data, dy_test)
