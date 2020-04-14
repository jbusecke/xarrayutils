import xarray as xr
import numpy as np

try:
    from astropy.convolution import convolve_fft, Gaussian1DKernel, Gaussian2DKernel

    astropy = True
except ImportError:
    astropy = None

from xarrayutils.utilities import detect_dtype


def filter_1D(data, std, dim="time", dtype=None):
    if astropy is None:
        raise RuntimeError(
            "Module `astropy` not found. Please install optional dependency with `conda install -c conda-forge astropy"
        )

    if dtype is None:
        dtype = detect_dtype(data)

    kernel = Gaussian1DKernel(std)

    def smooth_raw(data):
        raw_data = getattr(data, "values", data)
        result = convolve_fft(raw_data, kernel, boundary="wrap")
        result[np.isnan(raw_data)] = np.nan
        return result

    def temporal_smoother(data):
        dims = [dim]
        return xr.apply_ufunc(
            smooth_raw,
            data,
            vectorize=True,
            dask="parallelized",
            input_core_dims=[dims],
            output_core_dims=[dims],
            output_dtypes=[dtype],
        )

    return temporal_smoother(data)


def filter_2D(data, std, dim, dtype=None):
    if astropy is None:
        raise RuntimeError(
            "Module `astropy` not found. Please install optional dependency with `conda install -c conda-forge astropy"
        )

    if dtype is None:
        dtype = detect_dtype(data)

    kernel = Gaussian2DKernel(std)

    def smooth_raw(data):
        raw_data = getattr(data, "values", data)
        result = convolve_fft(raw_data, kernel, boundary="wrap")
        result[np.isnan(raw_data)] = np.nan
        return result

    def smoother(data):
        dims = dim
        # this is different from the 1d case

        return xr.apply_ufunc(
            smooth_raw,
            data,
            vectorize=True,
            dask="parallelized",
            input_core_dims=[dims],
            output_core_dims=[dims],
            output_dtypes=[dtype],
        )

    return smoother(data)


# TODO spatial filter
