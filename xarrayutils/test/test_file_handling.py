import xarray as xr
import numpy as np
import pytest
import pathlib
from xarrayutils.file_handling import (
    temp_write_split,
    maybe_create_folder,
    total_nested_size,
    write,
)


@pytest.fixture
def ds():
    data = np.random.rand()
    time = xr.cftime_range("1850", freq="1AS", periods=12)
    ds = xr.DataArray(data, dims=["x", "y", "time"], coords={"time": time}).to_dataset(
        name="data"
    )
    return ds


@pytest.mark.parametrize("dask", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("already_exists", [True, False])
@pytest.mark.parametrize("method", ["dimension", "variables", "wrong"])
def test_temp_write_split(ds, dask, method, verbose, already_exists, tmpdir):
    folder = tmpdir.mkdir("sub")
    folder = pathlib.Path(folder)

    # create test dataset
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


@pytest.mark.parametrize("sub", ["sub", "nested/sub/path"])
def test_maybe_create_folder(sub, tmp_path):
    folder = pathlib.Path(tmp_path)
    subfolder = folder.joinpath(sub)

    maybe_create_folder(subfolder)

    assert subfolder.exists()

    with pytest.warns(UserWarning):
        maybe_create_folder(subfolder)


def test_total_nested_size(ds):

    # create a bunch of broadcasted copies of a dataset
    a = ds.copy(deep=True).expand_dims(new=2)
    b = ds.copy(deep=True).expand_dims(new=5)
    c = ds.copy(deep=True).expand_dims(new=4, new_new=10)

    # nest them into a dict
    nested_dict = {"experiment_a": a, "experiment_b": {"label_x": b, "label_y": c}}
    size_nested = total_nested_size(nested_dict)

    assert size_nested == np.sum(np.array([i.nbytes for i in [a, b, c]]))


@pytest.mark.parametrize("strpath", [True, False])
@pytest.mark.parametrize("reload_saved", [True, False])
@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("filetype", [".nc", ".zarr"])
def test_write(ds, strpath, reload_saved, overwrite, filetype, tmpdir):
    def _load(path):
        if filetype == ".nc":
            return xr.open_dataset(path, use_cftime=True)
        else:
            return xr.open_zarr(path, use_cftime=True)

    folder = pathlib.Path(tmpdir)
    path = folder.joinpath("file" + filetype)
    if strpath:
        path_write = str(path)
    else:
        path_write = path

    write(ds, path)
    assert path.exists()
    xr.testing.assert_allclose(ds, _load(path))

    # create modified
    ds_modified = ds * 4
    dummy = write(
        ds_modified, path_write, overwrite=overwrite, reload_saved=reload_saved
    )

    if not overwrite:
        # this should not overwrite
        xr.testing.assert_allclose(_load(path_write), ds)

    else:
        # this should
        xr.testing.assert_allclose(_load(path_write), ds_modified)

    # check the reloaded file
    dummy = dummy.load()
    if reload_saved:
        xr.testing.assert_allclose(dummy, _load(path_write))
    else:
        xr.testing.assert_allclose(dummy, ds_modified)
