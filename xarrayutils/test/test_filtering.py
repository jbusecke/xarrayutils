import xarray as xr
import numpy as np
import pytest

astropy = pytest.importorskip("astropy")
from astropy.convolution import convolve_fft, Gaussian1DKernel, Gaussian2DKernel
from xarrayutils.filtering import filter_1D, filter_2D


@pytest.mark.parametrize("std", [1, 2, 4])
def test_filter_1D(std):
    data = np.random.rand(4, 4)
    ds = xr.DataArray(data, dims=["time", "something"])
    ds_filt = filter_1D(ds, std)

    for xi, xx in enumerate(ds.something.data):
        sample = ds_filt.sel({"something": xx})
        kernel = Gaussian1DKernel(std)
        expected = convolve_fft(data[:, xi], kernel, boundary="wrap")
        # result[np.isnan(raw_data)] = np.nan
        np.testing.assert_allclose(sample, expected)

    # TODO test case with nans


@pytest.mark.parametrize("radius", [1, 2, 4])
def test_filter_2D(radius):
    data = np.random.rand(4, 3, 6)
    ds = xr.DataArray(data, dims=["x", "something", "y"])
    ds_filt = filter_2D(ds, radius, dim=["x", "y"])

    for xi, xx in enumerate(ds.something.data):
        sample = ds_filt.sel({"something": xx})
        kernel = Gaussian2DKernel(radius)
        expected = convolve_fft(data[:, xi, :], kernel, boundary="wrap")
        # result[np.isnan(raw_data)] = np.nan
        np.testing.assert_allclose(sample, expected)

    # TODO test case with nans (astropy interpolates nicely.
    # I want to make the nan substitution optiona)
