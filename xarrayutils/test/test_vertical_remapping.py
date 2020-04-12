import numpy as np
import xarray as xr
from xarrayutils.vertical_remapping import _groupby_vert, xr_1d_groupby


import pytest

bins = [
    np.arange(26, 28, 0.1),  # regular spaced bins
    np.array([26.5, 26.6, 26.7, 26.8, 26.9, 27, 27.5, 28.0]),
]

depth = np.arange(10)
dens = np.linspace(26, 29, len(depth))


@pytest.mark.parametrize("bins", bins)
def test_groupby_vert(bins):
    # Create dummy DataArray
    data = np.random.rand(len(depth))
    da = xr.DataArray(data, coords=[("depth", depth)])
    da_group = xr.DataArray(dens, coords=[("depth", depth)])
    da_group.name = "density"  # needed for groupby_bins

    truth = da.groupby_bins(da_group, bins).sum()
    new = _groupby_vert(da.data, da_group.data, bins)

    np.testing.assert_allclose(new, truth.data)

    # test with different functions...

    #
    # # for the bigger tests
    # Needs to fail for unnamed da.


@pytest.mark.parametrize("bins", bins)
def test_xr_1d_groupby(bins):
    data = np.random.rand(2, 2, len(depth), 3)
    data_dens = abs(data) + dens[np.newaxis, np.newaxis, :, np.newaxis]
    da = xr.DataArray(
        data,
        coords=[("x", range(2)), ("y", range(2)), ("depth", depth), ("time", range(3))],
    )
    da_group = xr.DataArray(
        data_dens,
        coords=[("x", range(2)), ("y", range(2)), ("depth", depth), ("time", range(3))],
    )
    da_group_unnamed = da_group.copy()
    da_group.name = "density"
    # test also without
    sample = dict(x=1, y=0, time=1)  # randomize

    check = da.isel(**sample).groupby_bins(da_group.isel(**sample), bins).sum()
    test = xr_1d_groupby(da, da_group, bins, "depth").isel(**sample)

    np.testing.assert_allclose(check.data, test.data)
    with pytest.raises(ValueError):
        xr_1d_groupby(da, da_group_unnamed, bins, "depth")
