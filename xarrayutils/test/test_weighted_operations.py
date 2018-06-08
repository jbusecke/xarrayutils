import xarray as xr
import pytest
import numpy as np
import dask.array as dsa
from xarray.testing import assert_allclose
from xarrayutils.weighted_operations import _broadcast_weights, \
    weighted_mean, weighted_sum, weighted_sum_raw


def test_broadcast_weights():
    chunks = 1
    data_test = np.array([[np.nan, 1, 2], [1, 2, 4]])
    weight_test = np.array([1, 3, 5])
    attrs = {'test': 'content'}

    weight_expanded_test = \
        xr.DataArray(np.array([[np.nan, 3, 5], [1, 3, 5]]),
                     dims=['x', 'y'], name='data', attrs=attrs)
    weight_dsa_expanded_test = \
        xr.DataArray(dsa.from_array(weight_expanded_test,
                                    chunks=chunks),
                     dims=['x', 'y'], name='data', attrs=attrs)

    a = xr.DataArray(data_test, dims=['x', 'y'], name='data', attrs=attrs)
    w = xr.DataArray(weight_test, dims=['y'], name='data', attrs=attrs)

    a_dsa = xr.DataArray(dsa.from_array(data_test, chunks=chunks),
                         dims=['x', 'y'], name='data', attrs=attrs)
    w_dsa = xr.DataArray(dsa.from_array(weight_test, chunks=chunks),
                         dims=['y'], name='data', attrs=attrs)

    w_dsa_expanded = _broadcast_weights(a_dsa, w_dsa)
    w_dsa_expanded_attrs = _broadcast_weights(a_dsa, w_dsa, keep_attrs=True)
    w_expanded = _broadcast_weights(a, w)
    xr.testing.assert_allclose(w_dsa_expanded, weight_dsa_expanded_test)
    xr.testing.assert_identical(w_dsa_expanded_attrs,
                                weight_dsa_expanded_test)
    xr.testing.assert_allclose(w_expanded, weight_expanded_test)


def test_weighted_sum_raw():
    chunks = 1
    data_test = np.array([[np.nan, 2, 1], [2, 2, 4]])
    weight_test = np.array([1, 1, 2])

    attrs = {'test': 'content'}
    a_dsa = xr.DataArray(dsa.from_array(data_test, chunks=chunks),
                         dims=['x', 'y'], attrs=attrs)
    w_dsa = xr.DataArray(dsa.from_array(weight_test, chunks=chunks),
                         dims=['y'], attrs=attrs)

    data_sum, weight_sum = weighted_sum_raw(a_dsa, w_dsa, dim=['x', 'y'],
                                            dimcheck=False)
    data_sum_attrs, \
        weight_sum_attrs = weighted_sum_raw(a_dsa, w_dsa, dim=['x', 'y'],
                                            dimcheck=False, keep_attrs=True)

    assert weight_sum_attrs.attrs['test'] == attrs['test']
    assert data_sum_attrs.attrs['test'] == attrs['test']

def test_weighted_mean():
    chunks = 1
    data_test = np.array([[np.nan, 2, np.nan], [2, 2, 4]])
    weight_test = np.array([1, 1, 2])

    attrs = {'test': 'content'}
    a_dsa = xr.DataArray(dsa.from_array(data_test, chunks=chunks),
                         dims=['x', 'y'], attrs=attrs)
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
    xmean_alt = weighted_mean(a_dsa, w_dsa, dim='x',
                              dimcheck=False, keep_attrs=True)

    with pytest.raises(RuntimeError):
        weighted_mean(a_dsa, w_dsa, dim=['x', 'y'], dimcheck=True)

    assert np.isclose(mean, expected_mean)
    assert np.all(np.isclose(ymean, expected_ymean))
    assert np.all(np.isclose(xmean, expected_xmean))
    assert np.all(np.isclose(xmean, xmean_alt))
    assert xmean_alt.attrs['test'] == attrs['test']


def test_weighted_sum():
    chunks = 1
    data_test = np.array([[np.nan, 2, 1], [2, 2, 4]])
    weight_test = np.array([1, 1, 2])

    attrs = {'test': 'content'}
    a_dsa = xr.DataArray(dsa.from_array(data_test, chunks=chunks),
                         dims=['x', 'y'], attrs=attrs)
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
    xmean_alt = weighted_sum(a_dsa, w_dsa, dim='x',
                             dimcheck=False, keep_attrs=True)

    with pytest.raises(RuntimeError):
        weighted_sum(a_dsa, w_dsa, dim=['x', 'y'], dimcheck=True)

    assert np.isclose(mean, expected_mean)
    assert np.all(np.isclose(ymean, expected_ymean))
    assert np.all(np.isclose(xmean, expected_xmean))
    assert np.all(np.isclose(xmean, xmean_alt))
    assert xmean_alt.attrs['test'] == attrs['test']
