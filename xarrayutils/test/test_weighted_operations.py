import xarray as xr
import pytest
import numpy as np
import dask.array as dsa
from xarrayutils.weighted_operations import _broadcast_weights, \
    weighted_mean, weighted_sum


def test_weighted_mean():
    chunks = 1
    data_test = np.array([[np.nan, 2, np.nan], [2, 2, 4]])
    weight_test = np.array([1, 1, 2])

    # a = xr.DataArray(data_test, dims=['x', 'y'])
    # w = xr.DataArray(weight_test, dims=['y'])

    a_dsa = xr.DataArray(dsa.from_array(data_test, chunks=chunks),
                         dims=['x', 'y'])
    w_dsa = xr.DataArray(dsa.from_array(weight_test, chunks=chunks),
                         dims=['y'])

    expected_mean = np.array(2.8)
    expected_ymean = np.array([2.0, 3.0])
    expected_xmean = np.array([2, 2, 4])

    mean = weighted_mean(a_dsa, w_dsa, dim=['x', 'y'],
                         dimcheck=False)
    ymean = weighted_mean(a_dsa, w_dsa, dim=['y'],
                          dimcheck=False)
    xmean = weighted_mean(a_dsa, w_dsa, dim=['x'],
                          dimcheck=False)

    with pytest.raises(RuntimeError):
        weighted_mean(a_dsa, w_dsa, dim=['x', 'y'], dimcheck=True)

    assert np.isclose(mean, expected_mean)
    assert np.all(np.isclose(ymean, expected_ymean))
    assert np.all(np.isclose(xmean, expected_xmean))


def test_weighted_sum():
    chunks = 1
    data_test = np.array([[np.nan, 2, 1], [2, 2, 4]])
    weight_test = np.array([1, 1, 2])

    # a = xr.DataArray(data_test, dims=['x', 'y'])
    # w = xr.DataArray(weight_test, dims=['y'])

    a_dsa = xr.DataArray(dsa.from_array(data_test, chunks=chunks),
                         dims=['x', 'y'])
    w_dsa = xr.DataArray(dsa.from_array(weight_test, chunks=chunks),
                         dims=['y'])

    expected_mean = np.array(16.0)
    expected_ymean = np.array([4.0, 12.0])
    expected_xmean = np.array([2.0, 4.0, 10.0])

    mean = weighted_sum(a_dsa, w_dsa, dim=['x', 'y'],
                        dimcheck=False)
    ymean = weighted_sum(a_dsa, w_dsa, dim=['y'],
                         dimcheck=False)
    xmean = weighted_sum(a_dsa, w_dsa, dim=['x'],
                         dimcheck=False)

    print(xmean)
    print(xmean.data.compute())

    with pytest.raises(RuntimeError):
        weighted_sum(a_dsa, w_dsa, dim=['x', 'y'], dimcheck=True)

    assert np.isclose(mean, expected_mean)
    assert np.all(np.isclose(ymean, expected_ymean))
    assert np.all(np.isclose(xmean, expected_xmean))


def test_broadcast_weights():
    chunks = 1
    data_test = np.array([[np.nan, 1, 2], [1, 2, 4]])
    weight_test = np.array([1, 3, 5])
    weight_expanded_test = np.array([[np.nan, 3, 5], [1, 3, 5]])
    weight_dsa_expanded_test = \
        xr.DataArray(dsa.from_array(weight_expanded_test,
                                    chunks=chunks),
                     dims=['x', 'y'])

    a = xr.DataArray(data_test, dims=['x', 'y'])
    w = xr.DataArray(weight_test, dims=['y'])

    a_dsa = xr.DataArray(dsa.from_array(data_test, chunks=chunks),
                         dims=['x', 'y'])
    w_dsa = xr.DataArray(dsa.from_array(weight_test, chunks=chunks),
                         dims=['y'])

    w_dsa_expanded = _broadcast_weights(a_dsa, w_dsa)
    w_expanded = _broadcast_weights(a, w)
    xr.testing.assert_allclose(w_dsa_expanded, weight_dsa_expanded_test)
    np.testing.assert_allclose(w_expanded, weight_expanded_test)
