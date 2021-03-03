import xarray as xr

try:
    from fastprogress.fastprogress import progress_bar
    import fastprogress
except ImportError:
    fastprogress = None


import shutil


def temp_write_split(
    ds_in,
    folder,
    method="dimension",
    dim="time",
    split_interval=40,
    zarr_write_kwargs={},
    zarr_read_kwargs={},
    file_name_pattern="temp_write_split",
    verbose=False,
):
    """Splits the input dataset `ds_in` up either along a dimension `method='dimension'` or per variable `method='variable'`,
    and then writes these out to zarr. The data is then loaded back from the store.
    This is primarliy used to avoid too complex dask graphs by saving some intermediate step"""

    zarr_write_kwargs.setdefault("consolidated", False)
    zarr_read_kwargs.setdefault("use_cftime", True)
    zarr_read_kwargs.setdefault("consolidated", False)

    flist = []
    if method == "dimension":
        split_points = list(range(0, len(ds_in[dim]), split_interval)) + [None]
        if verbose:
            print(f" Split indicies: {split_points}")

        nsi = len(split_points) - 1
        if fastprogress:
            progress = progress_bar(range(nsi))
        else:
            progress = range(nsi)

        for si in progress:
            fname = folder.joinpath(f"{file_name_pattern}_{si}.zarr")
            if fname.exists():
                shutil.rmtree(fname)
            ds_in.isel({dim: slice(split_points[si], split_points[si + 1])}).to_zarr(
                fname, **zarr_write_kwargs
            )
            flist.append(fname)
        ds_out = xr.concat(
            [xr.open_zarr(f, **zarr_read_kwargs) for f in flist], dim=dim
        )
    elif method == "variables":
        # move all coords to data variables to avoid doubling up the writing for expensive (time resolved) coords
        reset_coords = [co for co in ds_in.coords if co not in ds_in.dims]
        ds_in = ds_in.reset_coords(reset_coords)

        variables = list(ds_in.data_vars)
        if verbose:
            print(variables)
        for var in variables:
            fname = folder.joinpath(f"{file_name_pattern}_{var}.zarr")
            if fname.exists():
                shutil.rmtree(
                    fname
                )  # can I just overwrite with zarr? This can take long!
            ds_in[var].to_dataset(name=var).to_zarr(fname, **zarr_write_kwargs)
            flist.append(fname)
        ds_out = xr.merge([xr.open_zarr(f, **zarr_read_kwargs) for f in flist])
        ds_out = ds_out.set_coords(reset_coords)
    else:
        raise ValueError(f"Method '{method}' not recognized.")
    return ds_out, flist
