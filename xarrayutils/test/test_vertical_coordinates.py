import xarray as xr
import numpy as np
import pytest
from xarrayutils.vertical_coordinates import (
    conservative_remap,
    _strip_dim,
    _coord_interp,
)


def random_broadcast(da):
    # add random noise with additionial dimensions to input array
    raw_noise = np.random.rand(2, 6, 12)
    noise = xr.DataArray(raw_noise, dims=["test_a", "test_b", "test_c"])
    return ((noise) * 4) + da


def test_strip_dim():
    data = np.array([0, 3, 4])
    a = xr.DataArray(data, coords=[("x", data)])
    b = xr.DataArray(data, dims="x")
    xr.testing.assert_identical(_strip_dim(a, "x"), b)


# TODO: make an explicit test for 1D case that compares values
@pytest.mark.parametrize("mask", [True, False])
@pytest.mark.parametrize("multi_dim", [True, False])
@pytest.mark.parametrize("dask", [True, False])
@pytest.mark.parametrize("coords", [True, False])
@pytest.mark.parametrize(
    "dat, src, tar",
    [
        (
            # first example, going from low res to high res
            # NOTE: the target bounds always need to cover all values of the
            # source, otherwise properties are not conserved
            np.array([30, 12.3, 5]),
            np.array([4.5, 9, 23, 45.6]),
            np.array([0, 2, 4, 10, 11, 13.4, 23, 55.6, 80, 100]),
        ),
        (
            # second example, going from high res to low res
            # NOTE: the target bounds always need to cover all values of the
            # source, otherwise properties are not conserved
            np.array([30, 12.3, 5, 2, -1, 4]),
            np.array([4.5, 9, 23, 45.6, 46, 70, 90]),
            np.array([0, 10, 100]),
        ),
        (
            # third example, using negative depth values sorted
            np.array([30, 12.3, 5, 2, -1, 4]),
            np.array([-90, -70, -46, -45.6, -23, -9, -4.5]),
            np.array([-100, -10, -0]),
        ),
        # (
        #     # forth example, using negative depth values unsorted
        #     # this is not supported atm, because I need to find a way to sort a
        #     # multidim depth array consistently (would be easy with 1d.)
        #     np.array([30, 12.3, 5, 2, -1, 4]),
        #     np.array([-4.5, -9, -23, -45.6, -46, -70, -90]),
        #     np.array([-0, -10, -100]),
        # ),
        (
            # fifth example, using negative AND positive depth values sorted
            np.array([30, 12.3, 5, 2, -1, 4]),
            np.array([-90, -70, -45.6, -23, 0, 1, 3]),
            np.array([-100, -10, 0, 10]),
        ),
    ],
)
def test_conservative_remap(mask, multi_dim, dask, dat, src, tar, coords):
    # create test datasets
    if coords:
        z_bounds_source = xr.DataArray(src, dims=["z_bounds"])
        z_bounds_target = xr.DataArray(tar, dims=["z_bounds"])
        # test for input values which have coordinate value in their dimension
        # (not just empty dimensions like above)
        z_raw = 0.5 * (src[1:] + src[0:-1])
        data = xr.DataArray(np.array(dat), dims=["z"], coords=[("z", z_raw)])

        z_bounds_source = xr.DataArray(
            src, dims=["z_bounds"], coords=[("z_bounds", src)]
        )
        z_bounds_target = xr.DataArray(
            tar, dims=["z_bounds"], coords=[("z_bounds", tar)]
        )
    else:
        data = xr.DataArray(np.array(dat), dims=["z"])
        z_bounds_source = xr.DataArray(src, dims=["z_bounds"])
        z_bounds_target = xr.DataArray(tar, dims=["z_bounds"])

    if multi_dim:
        # Add more dimension and shift values for each depth profile by a random amount
        data = random_broadcast(data)
        z_bounds_source = random_broadcast(z_bounds_source)
        z_bounds_target = random_broadcast(z_bounds_target)

    if dask:
        if multi_dim:
            chunks = {"test_c": 1}
        else:
            chunks = {}
        data = data.chunk(chunks)
        z_bounds_source = z_bounds_source.chunk(chunks)
        z_bounds_target = z_bounds_target.chunk(chunks)

    data_new = conservative_remap(
        data, z_bounds_source, z_bounds_target, mask=mask, debug=False
    )

    # Calculate cell thickness and rename
    # in case the bounds had coordinate values, these need to be stripped
    # to align properly
    dz_source = (
        _strip_dim(z_bounds_source, "z_bounds")
        .diff("z_bounds")
        .rename({"z_bounds": "z"})
    )
    dz_target = (
        _strip_dim(z_bounds_target, "z_bounds")
        .diff("z_bounds")
        .rename({"z_bounds": "remapped"})
    )

    raw = (data * dz_source).sum("z")
    remapped = (data_new * dz_target).sum("remapped")
    print(raw)
    print(remapped)
    xr.testing.assert_allclose(raw, remapped)


def test_coord_interp():
    # super simple test
    z = np.array([0, 2, 6, 20, 400])
    data = np.array([30, 20, 10, 5, 4])
    z_new = _coord_interp(z, data, 15)
    assert z_new == 4.0

    # test multiple targets
    z = np.array([0, 2, 6, 20, 400])
    data = np.array([30, 20, 10, 5, 4])
    z_new = _coord_interp(z, data, [15, 4.5])
    np.testing.assert_allclose(z_new, np.array([4.0, 210.0]))

    # test with nans
    z = np.array([0, 2, 6, np.nan, 400])
    data = np.array([30, 20, 10, 5, 4])
    z_new = _coord_interp(z, data, 15)
    assert z_new == 4.0

    z = np.array([0, 2, 6, 20, 400])
    data = np.array([30, 20, 10, 5, 4])
    for zz in [34, 2]:
        z_new = _coord_interp(z, data, zz)
        assert np.isnan(z_new)

    # test out of range with padding
    z = np.array([0, 2, 6, 20, 400])
    data = np.array([30, 20, 10, 5, 4])
    for pad_left in [-10, -5]:
        for pad_right in [400, 800]:
            z_new = _coord_interp(z, data, [40, 15, 4.5, 2], pad_left, pad_right)
            np.testing.assert_allclose(
                z_new, np.array([pad_left, 4.0, 210.0, pad_right])
            )
            z_new = _coord_interp(z, data, [2, 4.5, 15, 40], pad_left, pad_right)
            np.testing.assert_allclose(
                z_new, np.array([pad_right, 210.0, 4.0, pad_left])
            )

    # test out of range (with ascending data)
    z = np.array([0, 2, 6, 20, 400])
    data = np.array([30, 20, 10, 5, 4])
    for zz in [34, 2]:
        z_new = _coord_interp(z, data, zz)
        assert np.isnan(z_new)

    # test out of range with padding
    z = np.array([0, 2, 6, 20, 400])
    data = np.array([4, 5, 10, 20, 30])
    for pad_left in [-10, -5]:
        for pad_right in [400, 800]:
            z_new = _coord_interp(z, data, [40, 15, 4.5, 2], pad_left, pad_right)
            np.testing.assert_allclose(
                z_new, np.array([pad_right, 13.0, 1.0, pad_left])
            )
            z_new = _coord_interp(z, data, [2, 4.5, 15, 40], pad_left, pad_right)
            np.testing.assert_allclose(
                z_new, np.array([pad_left, 1.0, 13.0, pad_right])
            )

    # assert z_new == 780

    # # test out of range (above) flipped
    # z = np.array([0, 2, 6, 20, 400])
    # data = np.array([4, 5, 10, 20, 30])
    # z_new = _coord_interp(z, data, 34)
    # assert z_new == 400
    # # assert z_new == 780
    #
    # # test out of range (below) flipped
    # z = np.array([0, 2, 6, 20, 400])
    # data = np.array([4, 5, 10, 20, 30])
    # z_new = _coord_interp(z, data, 2)
    # assert z_new == 0
    # # assert z_new == -2
    #
    # # test out of range with multiple vals
    # z = np.array([0, 2, 6, 20, 400])
    # data = np.array([30, 20, 10, 5, 4])
    # z_new = _coord_interp(z, data, np.array([34, 30, 15, 4.5, 2]))
    # np.testing.assert_allclose(z_new, np.array([0, 0, 4.0, 210.0, 400]))

    # test super short array
    z = np.array([2, 4])
    data = np.array([20, 30])
    z_new = _coord_interp(z, data, 25)
    assert z_new == 3.0

    # test boundary value
    z = np.array([2, 4])
    data = np.array([20, 30])
    z_new = _coord_interp(z, data, 20)
    assert z_new == 2  # this is the default interpolation behaviour
    z_new = _coord_interp(z, data, 20, -10, 200)
    assert z_new == -10
    z_new = _coord_interp(z, data, 30, -10, 200)
    assert z_new == 200

    # test to short to interpolate
    z = np.array([2, np.nan, np.nan])
    data = np.array([30, np.nan, np.nan])
    z_new = _coord_interp(z, data, 30)
    assert np.isnan(z_new)
