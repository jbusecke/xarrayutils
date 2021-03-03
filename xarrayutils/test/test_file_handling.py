import xarray as xr
import numpy as np
import pytest
import pathlib
from xarrayutils.file_handling import temp_write_split


@pytest.mark.parametrize("dask", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("already_exists", [True, False])
@pytest.mark.parametrize("method", ["dimension", "variables", "wrong"])
def test_temp_write_split(dask, method, verbose, already_exists, tmpdir):
    folder = tmpdir.mkdir("sub")
    folder = pathlib.Path(folder)

    # create test dataset
    data = np.random.rand()
    time = xr.cftime_range("1850", freq="1AS", periods=12)
    ds = xr.DataArray(data, dims=["x", "y", "time"], coords={"time": time}).to_dataset(
        name="data"
    )
    if dask:
        ds = ds.chunk({"time": 1})

    # write a manual copy (with wrong data) to test the erasing
    (ds.isel(time=0) + 100).to_zarr(
        folder.joinpath("temp_write_split_0.zarr"), consolidated=True
    )

    if method == "wrong":
        with pytest.raises(ValueError):
            temp_write_split(
                ds,
                folder,
                method=method,
                split_interval=1,
            )
    else:
        ds_reloaded, filelist = temp_write_split(
            ds,
            folder,
            method=method,
            verbose=verbose,
            split_interval=1,
        )
        xr.testing.assert_allclose(ds, ds_reloaded)
