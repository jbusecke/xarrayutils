import pytest
import xarray as xr
import numpy as np
import dask.array as dsa
from scipy import stats

# import os

from xarrayutils.utils import (
    aggregate,
    aggregate_w_nanmean,
    extractBox_dict,
    linear_trend,
    _lin_trend_legacy,
    _linregress_ufunc,
    xr_linregress,
    xr_detrend,
    lag_and_combine,
    filter_1D,
)

from numpy.testing import assert_allclose

from .datasets import dataarray_2d_example, dataarray_2d_ones, dataarray_2d_ones_nan


def test_filter_1D():
    ds = xr.DataArray(np.random.rand(4, 4), dims=["time", "something"])
    with pytest.warns(DeprecationWarning):
        filter_1D(ds, 10)


def test_lag_and_combine():
    x = np.arange(5)
    y = np.arange(5) * 3
    y_lagged = np.hstack((y[1:], [np.nan]))
    da = xr.DataArray(y, coords=[x], dims=["x"])
    da_lagged = lag_and_combine(da, lags=[-1], dim="x")
    da_lagged_test = xr.DataArray(y_lagged, coords=[x], dims=["x"])
    assert_allclose(da_lagged.squeeze().data, da_lagged_test.data)


@pytest.mark.parametrize(
    "box, concat_wrap, result",
    [
        (
            {"x": np.array([0, 1]), "y": np.array([0, 1])},
            True,
            np.array([[0, 0], [10, 20]]),
        ),
        (
            {"x": np.array([0, 1]), "y": np.array([3.5, 1])},
            True,
            np.array([[0, 0, 0], [50, 10, 20]]),
        ),
        (
            {"x": np.array([0, 1]), "y": np.array([3.5, 1])},
            False,
            np.array([[0, 0, 0], [10, 20, 50]]),
        ),
        (
            {"x": np.array([2.5, 0.5]), "y": np.array([3.5, 1])},
            True,
            np.array([[150, 30, 60], [0, 0, 0]]),
        ),
        (
            {"x": np.array([2.5, 0.5]), "y": np.array([3.5, 1])},
            {"x": True, "y": False},
            np.array([[30, 60, 150], [0, 0, 0]]),
        ),
    ],
)
def test_extractBox_dict(box, concat_wrap, result):
    x = xr.DataArray(
        np.array([0, 1, 2, 3]),
        dims=["x"],
        coords={"x": (["x"], np.array([0, 1, 2, 3]))},
    )
    y = xr.DataArray(
        np.array([10, 20, 30, 40, 50]),
        dims=["y"],
        coords={"y": (["y"], np.array(range(5)))},
    )
    c = x * y
    box_cut = extractBox_dict(c, box, concat_wrap=concat_wrap)
    assert_allclose(box_cut.data, result)

    c = c.chunk({"x": 1})
    box_cut_dask = extractBox_dict(c, box, concat_wrap=concat_wrap)
    assert isinstance(box_cut_dask.data, dsa.Array)
    assert_allclose(box_cut_dask.data, result)


# @pytest.mark.parametrize(
#     "box, concat_wrap, result",
#     [
#         (
#             {"x": np.array([0, 1]), "y": np.array([0, 1])},
#             True,
#             np.array([[0, 0], [10, 20]]),
#         ),
#         (
#             {"x": np.array([0, 1]), "y": np.array([3.5, 1])},
#             True,
#             np.array([[0, 0, 0], [50, 10, 20]]),
#         ),
#     ],
# )
# def test_extractBox(box, concat_wrap, result):
#     x = xr.DataArray(
#         np.array([0, 1, 2, 3]),
#         dims=["x"],
#         coords={"x": (["x"], np.array([0, 1, 2, 3]))},
#     )
#     y = xr.DataArray(
#         np.array([10, 20, 30, 40, 50]),
#         dims=["y"],
#         coords={"y": (["y"], np.array(range(5)))},
#     )
#     c = x * y
#     box_cut = extractBox_dict(c, box, concat_wrap=concat_wrap)
#     assert_allclose(box_cut.data, result)


def test_lin_trend_full_legacy():
    # This is meant to be a test if the old ufunc for the trend produces
    # identical output
    y = np.random.random(20)
    x = np.arange(len(y))
    fit = _linregress_ufunc(x, y)[0:2]
    fit_legacy = _lin_trend_legacy(y)
    assert np.allclose(fit, fit_legacy)


def test_linregress_ufunc():
    y = np.random.random(20)
    x = np.arange(len(y))
    fit = np.array(stats.linregress(x, y))
    assert np.allclose(fit, _linregress_ufunc(x, y))
    # test with nans
    y[0] = np.nan
    fit = np.array(stats.linregress(x[1:], y[1:]))
    assert np.isnan(_linregress_ufunc(x, y)).all()
    assert np.allclose(fit, _linregress_ufunc(x, y, nanmask=True))
    # test with all nansum
    y[:] = np.nan
    assert np.isnan(_linregress_ufunc(x, y, nanmask=True)).all()


@pytest.mark.parametrize("chunks", [None, {"x": -1, "y": 1}, {"x": 1, "y": 1}])
@pytest.mark.parametrize("variant", range(3))
@pytest.mark.parametrize("dtype", [None, np.float])
@pytest.mark.parametrize("nans", [False, True])
def test_xr_linregress(chunks, variant, dtype, nans):
    a = xr.DataArray(np.random.rand(3, 13, 5), dims=["x", "time", "y"])
    b = xr.DataArray(np.random.rand(3, 5, 13), dims=["x", "y", "time"])
    if nans:
        # add nans at random positions
        a.data[np.unravel_index(np.random.randint(0, 2 * 4 * 12, 10), a.shape)] = np.nan
        b.data[np.unravel_index(np.random.randint(0, 2 * 4 * 12, 10), b.shape)] = np.nan

    if chunks is not None:
        if variant == 0:
            a = a.chunk(chunks)
        elif variant == 1:
            b = b.chunk(chunks)
        elif variant == 2:
            a = a.chunk(chunks)
            b = b.chunk(chunks)

    reg = xr_linregress(a, b, dtype=dtype)
    for xx in range(len(a.x)):
        for yy in range(len(a.y)):
            pos = dict(x=xx, y=yy)
            expected = _linregress_ufunc(a.isel(**pos), b.isel(**pos))
            reg_sub = reg.isel(**pos)
            for ni, nn in enumerate(
                ["slope", "intercept", "r_value", "p_value", "std_err"]
            ):
                np.testing.assert_allclose(reg_sub[nn].data, expected[ni])


def test_linear_trend():
    # TODO implement a test for nans
    data = dsa.from_array(np.random.random([10, 2, 2]), chunks=(10, 1, 1))
    t = range(10)
    x = range(2)
    y = range(2)
    data_da = xr.DataArray(
        data,
        dims=["time", "x", "y"],
        coords={"time": ("time", t), "x": ("x", x), "y": ("y", y)},
    )

    fit_da = linear_trend(data_da, "time")

    for xi in x:
        for yi in y:
            x_fit = t
            y_fit = data[:, xi, yi]
            fit = np.array(stats.linregress(x_fit, y_fit))
            test = fit_da.sel(x=xi, y=yi).data
            assert np.allclose(fit, test)

    # Test with other timedim (previously was not caught)
    data = dsa.from_array(np.random.random([2, 10, 2]), chunks=(1, 10, 1))
    data_da = xr.DataArray(
        data,
        dims=["x", "time", "y"],
        coords={"x": ("x", x), "time": ("time", t), "y": ("y", y)},
    )

    fit_da = linear_trend(data_da, "time")

    for xi in x:
        for yi in y:
            x_fit = t
            y_fit = data[xi, :, yi]
            fit = np.array(stats.linregress(x_fit, y_fit))
            assert np.allclose(fit, fit_da.sel(x=xi, y=yi).data)


# @pytest.mark.parametrize(
#     "box, concat_wrap, result",
#     [
#         (
#             {"x": np.array([0, 1]), "y": np.array([0, 1])},
#             True,
#             np.array([[0, 0], [10, 20]]),
#         ),
#         (
#             {"x": np.array([0, 1]), "y": np.array([3.5, 1])},
#             True,
#             np.array([[0, 0, 0], [50, 10, 20]]),
#         ),
#         (
#             {"x": np.array([0, 1]), "y": np.array([3.5, 1])},
#             False,
#             np.array([[0, 0, 0], [10, 20, 50]]),
#         ),
#         (
#             {"x": np.array([2.5, 0.5]), "y": np.array([3.5, 1])},
#             True,
#             np.array([[150, 30, 60], [0, 0, 0]]),
#         ),
#         (
#             {"x": np.array([2.5, 0.5]), "y": np.array([3.5, 1])},
#             {"x": True, "y": False},
#             np.array([[30, 60, 150], [0, 0, 0]]),
#         ),
#     ],
# )
# def test_extractBox_dict(box, concat_wrap, result):
#     x = xr.DataArray(
#         np.array([0, 1, 2, 3]),
#         dims=["x"],
#         coords={"x": (["x"], np.array([0, 1, 2, 3]))},
#     )
#     y = xr.DataArray(
#         np.array([10, 20, 30, 40, 50]),
#         dims=["y"],
#         coords={"y": (["y"], np.array(range(5)))},
#     )
#     c = x * y
#     box_cut = extractBox_dict(c, box, concat_wrap=concat_wrap)
#     assert_allclose(box_cut.data, result)
#
#     c = c.chunk({"x": 1})
#     box_cut_dask = extractBox_dict(c, box, concat_wrap=concat_wrap)
#     assert isinstance(box_cut_dask.data, dsa.Array)
#     assert_allclose(box_cut_dask.data, result)


# @pytest.mark.parametrize(
#     "box, concat_wrap, result",
#     [
#         (
#             {"x": np.array([0, 1]), "y": np.array([0, 1])},
#             True,
#             np.array([[0, 0], [10, 20]]),
#         ),
#         (
#             {"x": np.array([0, 1]), "y": np.array([3.5, 1])},
#             True,
#             np.array([[0, 0, 0], [50, 10, 20]]),
#         ),
#     ],
# )
# def test_extractBox(box, concat_wrap, result):
#     x = xr.DataArray(
#         np.array([0, 1, 2, 3]),
#         dims=["x"],
#         coords={"x": (["x"], np.array([0, 1, 2, 3]))},
#     )
#     y = xr.DataArray(
#         np.array([10, 20, 30, 40, 50]),
#         dims=["y"],
#         coords={"y": (["y"], np.array(range(5)))},
#     )
#     c = x * y
#     box_cut = extractBox_dict(c, box, concat_wrap=concat_wrap)
#     assert_allclose(box_cut.data, result)


@pytest.mark.parametrize(
    "func,expected_result",
    [
        (np.nanmean, np.array([[1, 3], [2, 4]])),
        (np.mean, np.array([[np.nan, 3], [2, 4]])),
    ],
)
def test_aggregate_regular_func(dataarray_2d_example, func, expected_result):
    blocks = [("i", 3), ("j", 3)]
    a = aggregate(dataarray_2d_example, blocks, func=func)
    assert_allclose(a.data.compute(), expected_result)


@pytest.mark.parametrize(
    "blocks,expected_result",
    [
        (
            [("i", 2), ("j", 2)],
            np.array([[1, 2, 3, 5], [1.5, 2.5, 3.5, 5.5], [2, 3, 4, 6]]),
        )
    ],
)
def test_aggregate_regular_blocks(dataarray_2d_example, blocks, expected_result):
    func = np.nanmean
    a = aggregate(dataarray_2d_example, blocks, func=func)
    assert_allclose(a.data, expected_result)


@pytest.mark.parametrize(
    "blocks_fail",
    [
        [("i", 3.4), ("j", 2)],
        # non int interval
        [("blah", 2), ("blubb", 3)],
        # no matching labels
        [(2, 2), ("j", 2)]
        # non str dim label
    ],
)
def test_aggregate_input_blocks(dataarray_2d_example, blocks_fail):
    with pytest.raises(RuntimeError):
        aggregate(dataarray_2d_example, blocks_fail, func=np.nanmean)


def test_aggregate_input_da(dataarray_2d_example):
    blocks = [("i", 3), ("j", 3)]
    with pytest.raises(RuntimeError):
        aggregate(dataarray_2d_example.compute(), blocks, func=np.nanmean)


def test_aggregate_w_nanmean(dataarray_2d_ones, dataarray_2d_ones_nan):
    expected_result = np.array([[1, 1], [1, 1]], dtype=np.float)
    blocks = [("i", 3), ("j", 3)]

    data = dataarray_2d_ones_nan
    weights = dataarray_2d_ones
    a = aggregate_w_nanmean(data, weights, blocks)
    assert_allclose(a.data.compute(), expected_result)

    data = dataarray_2d_ones_nan
    weights = dataarray_2d_ones_nan
    a = aggregate_w_nanmean(data, weights, blocks)
    assert_allclose(a.data.compute(), expected_result)

    with pytest.raises(RuntimeError):
        data = dataarray_2d_ones
        weights = dataarray_2d_ones_nan
        a = aggregate_w_nanmean(data, weights, blocks)


def test_detrend():
    # based on test_linear_trend
    # TODO implement a test for nans
    data = dsa.from_array(np.random.random([10, 2, 2]), chunks=(10, 1, 1))
    t = range(10)
    x = range(2)
    y = range(2)
    data_da = xr.DataArray(
        data,
        dims=["time", "x", "y"],
        coords={"time": ("time", t), "x": ("x", x), "y": ("y", y)},
    )

    detrended_da = xr_detrend(data_da)

    for xi in x:
        for yi in y:
            x_fit = np.array(t)
            y_fit = data[:, xi, yi]
            fit = np.array(stats.linregress(x_fit, y_fit))
            detrended = y_fit - (fit[1] + fit[0] * x_fit)
            test = detrended_da.sel(x=xi, y=yi).data
            assert np.allclose(detrended, test)

    # Test with other timedim (previously was not caught)
    # could test with timedim named other than 'time'
    data = dsa.from_array(np.random.random([2, 10, 2]), chunks=(1, 10, 1))
    data_da = xr.DataArray(
        data,
        dims=["x", "time", "y"],
        coords={"x": ("x", x), "time": ("time", t), "y": ("y", y)},
    )

    detrended_da = xr_detrend(data_da)

    for xi in x:
        for yi in y:
            x_fit = np.array(t)
            y_fit = data[xi, :, yi]
            fit = np.array(stats.linregress(x_fit, y_fit))
            detrended = y_fit - (fit[1] + fit[0] * x_fit)
            test = detrended_da.sel(x=xi, y=yi).data
            assert np.allclose(detrended, test)
