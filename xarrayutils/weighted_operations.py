from __future__ import print_function
from future.utils import iteritems
import dask.array as dsa
import xarray as xr
import numpy as np


def weighted_mean(da_data, da_weight, dim=None,
                  preweighted=False, dimcheck=True):
    """calculate average of da_data weighted by da_weight

    Parameters
    ----------

    da_data : xarray.DataArray
        Data to be averaged
    da_weight : xarray.DataArray
        weights to be used during averaging. Dimensions have to be
        matching with 'da_data'
    dim : {None, str, list}, optional
        Dimensions to average over
    preweighted: Bool, optional
        Specifies whether weights will be applied (False, default) or
        have already been
        applied to da_data (True).
    dim_check: Bool, optional
        Activates check for dimension consistency. If dimensions of 'da_weight'
        do not include all elements of 'dim' error is raised
    """

    # Check dimension consistency
    if dim:
        if dimcheck:
            if not set(dim).issubset(da_weight.dims):
                raise RuntimeError("Dimensions of 'da_weight' do not include \
            all averaging dimensions. Broadcast da_weight \
            or deactivate 'dim_check'.")

    weight_expanded = _broadcast_weights(da_data, da_weight)

    if preweighted:
        data = da_data
    else:
        data = da_data*weight_expanded

    return data.sum(dim)/weight_expanded.sum(dim)

def weighted_sum(da_data, da_weight, dim=None,
                 preweighted=False, dimcheck=True):
    """calculate sum of da_data weighted by da_weight

    Parameters
    ----------

    da_data : xarray.DataArray
        Data to be averaged
    da_weight : xarray.DataArray
        weights to be used during averaging. Dimensions have to be matching
        with 'da_data'
    dim : {None, str, list}, optional
        Dimensions to average over
    preweighted: Bool, optional
        Specifies whether weights will be applied (False, default) or have
        already been
        applied to da_data (True).
    dim_check: Bool, optional
        Activates check for dimension consistency. If dimensions of
        'da_weight' do not include all elements of 'dim' error is raised
    """

    # Check dimension consistency
    if dim:
        if dimcheck:
            if not set(dim).issubset(da_weight.dims):
                raise RuntimeError("Dimensions of 'da_weight' do not include all averaging dimensions.\
                Broadcast da_weight or deactivate 'dim_check'.")

    weight_expanded = _broadcast_weights(da_data, da_weight)

    if preweighted:
        data = da_data
    else:
        data = da_data*weight_expanded

    return data.sum(dim)


def _broadcast_weights(da_data, da_weight):
    """broadcasts da_weights to the same shape as da_data and \
    masks missing values"""
    ones = (da_data.copy()*0)+1
    weights_expanded = ones*da_weight
    return weights_expanded
