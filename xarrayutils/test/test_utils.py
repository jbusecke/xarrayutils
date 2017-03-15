import pytest
import xarray as xr
import numpy as np
import os
from xarrayutils.utils import aggregate,aggregate_w_nanmean
from numpy.testing import assert_allclose

from . datasets import dataarray_2d_example,dataarray_2d_ones,dataarray_2d_ones_nan

@pytest.mark.parametrize(
    "func,expected_result",[
        (np.nanmean,
        np.array(
        [[1,3],
        [2,4]])),
        (np.mean,
        np.array(
        [[np.nan,3],
        [2,4]])
        )])

def test_aggregate_regular_func(dataarray_2d_example,func,expected_result):
    blocks = [('i',3),('j',3)]
    a = aggregate(dataarray_2d_example,blocks,func=func)
    assert_allclose(a.data.compute(),expected_result)

@pytest.mark.parametrize(
    "blocks,expected_result",[
        ([('i',2),('j',2)],
        np.array(
        [[1,2,3,5],
        [1.5,2.5,3.5,5.5],
        [2,3,4,6]]))
        #,
        # ([('i',8),('j',6)],
        # [[3.29787234043]])
        #
        ]
        )

def test_aggregate_regular_blocks(dataarray_2d_example,blocks,expected_result):
    func = np.nanmean
    a = aggregate(dataarray_2d_example,blocks,func=func)
    assert_allclose(a.data,expected_result)

@pytest.mark.parametrize(
    "blocks_fail",[
    [('i',3.4),('j',2)], #non int interval
    [('blah',2),('blubb',3)], # no matching labels
    [(2,2),('j',2)] #non str dim label
    ])

def test_aggregate_input_blocks(dataarray_2d_example,blocks_fail):
    with pytest.raises(RuntimeError):
        aggregate(dataarray_2d_example,blocks_fail,func=np.nanmean)

def test_aggregate_input_da(dataarray_2d_example):
    blocks = [('i',3),('j',3)]
    with pytest.raises(RuntimeError):
        aggregate(dataarray_2d_example.compute(),blocks,func=np.nanmean)

def test_aggregate_w_nanmean(dataarray_2d_ones,dataarray_2d_ones_nan):
    expected_result = np.array([
        [1,1],
        [1,1]
        ],dtype=np.float)
    blocks = [('i',3),('j',3)]

    data = dataarray_2d_ones_nan
    weights = dataarray_2d_ones
    a = aggregate_w_nanmean(data,weights,blocks)
    assert_allclose(a.data.compute(),expected_result)

    data = dataarray_2d_ones_nan
    weights = dataarray_2d_ones_nan
    a = aggregate_w_nanmean(data,weights,blocks)
    assert_allclose(a.data.compute(),expected_result)

    with pytest.raises(RuntimeError):
        data = dataarray_2d_ones
        weights = dataarray_2d_ones_nan
        a = aggregate_w_nanmean(data,weights,blocks)
