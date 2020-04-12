import xarray as xr
import numpy as np
import pytest
from xarrayutils.vertical_coordinates import (
    conservative_remap,
    linear_interpolation_remap,
    linear_interpolation_regrid,
    _strip_dim,
    _coord_interp,
    _regular_interp,
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


def test_regular_interp():
    x = np.arange(5, dtype=np.float)
    y = np.linspace(3, 7, 5)
    target = np.array([0.5, 3.0, np.nan, 7.0])  #
    interpolated = _regular_interp(x, y, target)
    expected = np.array([3.5, 6.0, np.nan, np.nan])
    np.testing.assert_allclose(interpolated, expected)

    # so when I specify the literal boundary value it returns nan...
    # not sure if this is causing and problems. Ill leave it here commented
    # target = np.array([5.0])
    # interpolated = _regular_interp(x, y, target)
    # expected = np.array([7.0])


@pytest.mark.parametrize("multi_dim", [True, False])
@pytest.mark.parametrize("coords", [True, False])
@pytest.mark.parametrize("z_dim", ["z", "blob"])
@pytest.mark.parametrize("z_regridded_dim", ["z_new", "blab"])
@pytest.mark.parametrize("output_dim", ["remapped", "boink"])
@pytest.mark.parametrize(
    "dat, src, tar",
    [
        (
            # first example, going from low res to high res
            # NOTE: the target bounds always need to cover all values of the
            # source, otherwise properties are not conserved
            np.array([30, 12.3, 5]),
            np.array([4.5, 9, 23]),
            np.array([0, 2, 4, 10, 11, 13.4, 23, 55.6, 80, 100]),
        ),
        (
            # second example, going from high res to low res
            # NOTE: the target bounds always need to cover all values of the
            # source, otherwise properties are not conserved
            np.array([30, 12.3, 5, 2, -1, 4]),
            np.array([4.5, 9, 23, 45.6, 46, 70]),
            np.array([0, 10, 100]),
        ),
        (
            # third example, using negative depth values sorted
            np.array([30, 12.3, 5, 2, -1, 4]),
            np.array([-90, -70, -46, -45.6, -23, -9]),
            np.array([-100, -10, -0]),
        ),
        (
            # forth example, using negative depth values unsorted
            # this is not supported atm, because I need to find a way to sort a
            # multidim depth array consistently (would be easy with 1d.)
            np.array([30, 12.3, 5, 2, -1, 4]),
            np.array([-4.5, -9, -23, -45.6, -46, -70]),
            np.array([-0, -10, -100]),
        ),
        (
            # fifth example, using negative AND positive depth values sorted
            np.array([30, 12.3, 5, 2, -1, 4]),
            np.array([-90, -70, -45.6, -23, 0, 1]),
            np.array([-100, -10, 0, 10]),
        ),
    ],
)
def test_linear_interpolation_remap(
    dat, src, tar, coords, multi_dim, z_dim, z_regridded_dim, output_dim
):
    # create test datasets
    if coords:
        data = xr.DataArray(np.array(dat), dims=[z_dim], coords=[(z_dim, src)])

        z_source = xr.DataArray(src, dims=[z_dim], coords=[(z_dim, src)])
        z_target = xr.DataArray(
            tar, dims=[z_regridded_dim], coords=[(z_regridded_dim, tar)]
        )
    else:
        data = xr.DataArray(np.array(dat), dims=[z_dim])
        z_source = xr.DataArray(src, dims=[z_dim])
        z_target = xr.DataArray(tar, dims=[z_regridded_dim])

    # the target and data should always have other dimensions
    data = random_broadcast(data)
    z_target = random_broadcast(z_target)
    if multi_dim:
        z_source = random_broadcast(z_source)

    remapped = linear_interpolation_remap(
        z_source,
        data,
        z_target,
        z_dim=z_dim,
        z_regridded_dim=z_regridded_dim,
        output_dim=output_dim,
    )
    # select random sample for 3 times
    for iteration in range(3):
        sample = {
            "test_a": np.random.randint(0, len(remapped.test_a)),
            "test_b": np.random.randint(0, len(remapped.test_b)),
            "test_c": np.random.randint(0, len(remapped.test_c)),
        }
        profile = remapped.isel(**sample).load().data
        if multi_dim:
            x = z_source.isel(**sample)
        else:
            x = z_source

        target = z_target.isel(**sample)
        y = data.isel(**sample)
        expected_profile = _regular_interp(x.data, y.data, target.data)

        np.testing.assert_allclose(profile, expected_profile)

    # check output coords
    np.testing.assert_allclose(
        remapped.coords[output_dim].data, z_target.coords[z_regridded_dim].data
    )


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


@pytest.mark.parametrize("multi_dim", [True, False])
@pytest.mark.parametrize("coords", [True, False])
@pytest.mark.parametrize("z_dim", ["z", "blob"])
@pytest.mark.parametrize("target_value_dim", ["temp", "blab"])
@pytest.mark.parametrize("output_dim", ["remapped", "boink"])
@pytest.mark.parametrize("z_bounds", [False, True])
@pytest.mark.parametrize("z_bounds_dim", ["z_bounds", "bawoosh"])
@pytest.mark.parametrize(
    "dat, src, tar",
    [
        (
            # first example, going from low res to high res
            # NOTE: the target bounds always need to cover all values of the
            # source, otherwise properties are not conserved
            np.array([30, 12.3, 5]),
            np.array([4.5, 9, 23, 40]),
            np.array([0, 2, 4, 10, 11, 13.4, 23, 55.6, 80, 100]),
        ),
        (
            # second example, going from high res to low res
            # NOTE: the target bounds always need to cover all values of the
            # source, otherwise properties are not conserved
            np.array([30, 12.3, 5, 2, -1, 4]),
            np.array([4.5, 9, 23, 45.6, 46, 70, 90]),
            np.array([0, 10, 30, 100]),
        ),
        (
            # third example, using negative depth values sorted
            np.array([30, 12.3, 5, 2, -1, 4]),
            np.array([-90, -70, -46, -45.6, -23, -9, -2]),
            np.array([-100, -10, -4, -0]),
        ),
        (
            # forth example, using negative depth values unsorted
            # this is not supported atm, because I need to find a way to sort a
            # multidim depth array consistently (would be easy with 1d.)
            np.array([30, 12.3, 5, 2, -1, 4]),
            np.array([-4.5, -9, -23, -45.6, -46, -70, -80]),
            np.array([-0, -10, -40, -100]),
        ),
        (
            # fifth example, using negative AND positive depth values sorted
            np.array([30, 12.3, 5, 2, -1, 4]),
            np.array([-90, -70, -45.6, -23, 0, 1, 2]),
            np.array([-100, -10, 0, 10]),
        ),
    ],
)
def test_linear_interpolation_regrid(
    dat,
    src,
    tar,
    coords,
    multi_dim,
    z_dim,
    target_value_dim,
    output_dim,
    z_bounds,
    z_bounds_dim,
):
    # this is a lot of setup but the test is simple. Create some random input arrays
    # of different shapes, and see if for random samples, the profile data is what we
    # would expect from the utility function. All other issues should be addressed within
    # that function.

    # create test datasets
    src = np.array(src)
    # reconstruct the source depth center from cell bounds
    z_raw = 0.5 * (src[1:] + src[0:-1])
    if coords:
        data = xr.DataArray(dat, coords=[(z_dim, z_raw)])
        z_source_bnds = xr.DataArray(src, coords=[(z_bounds_dim, src)])
        z_target = xr.DataArray(tar, coords=[(target_value_dim, tar)])
        z_source = xr.DataArray(z_raw, coords=[(z_dim, z_raw)])
    else:
        data = xr.DataArray(np.array(dat), dims=[z_dim])
        z_source_bnds = xr.DataArray(src, dims=[z_bounds_dim])
        z_target = xr.DataArray(tar, dims=[target_value_dim])
        z_source = xr.DataArray(z_raw, dims=[z_dim])

    # the target and data should always have other dimensions
    data = random_broadcast(data)
    z_target = random_broadcast(z_target)
    if multi_dim:
        z_source_bnds = random_broadcast(z_source_bnds)
        z_source = random_broadcast(z_source)

    if z_bounds:
        bnds = z_source_bnds
    else:
        bnds = None

    regridded = linear_interpolation_regrid(
        z_source,
        data,
        z_target,
        z_dim=z_dim,
        target_value_dim=target_value_dim,
        output_dim=output_dim,
        z_bounds=bnds,
        z_bounds_dim=z_bounds_dim,
    )
    # select random sample for 2 times
    for iteration in range(2):
        sample = {
            "test_a": np.random.randint(0, len(regridded.test_a)),
            "test_b": np.random.randint(0, len(regridded.test_b)),
            "test_c": np.random.randint(0, len(regridded.test_c)),
        }
        profile = regridded.isel(**sample).load().data
        if multi_dim:
            z = z_source.isel(**sample)
        else:
            z = z_source

        target = z_target.isel(**sample)
        d = data.isel(**sample)

        if z_bounds:
            pad_left = bnds[{z_bounds_dim: 0}]
            pad_right = bnds[{z_bounds_dim: -1}]
            if multi_dim:
                pad_left = pad_left.isel(**sample)
                pad_right = pad_right.isel(**sample)
            pad_left = pad_left.data
            pad_right = pad_right.data
        else:
            pad_left = pad_right = None

        expected_profile = _coord_interp(
            z.data, d.data, target.data, pad_left=pad_left, pad_right=pad_right
        )

        np.testing.assert_allclose(profile, expected_profile)

    # check output coords
    np.testing.assert_allclose(
        regridded.coords[output_dim].data, z_target.coords[target_value_dim].data
    )
