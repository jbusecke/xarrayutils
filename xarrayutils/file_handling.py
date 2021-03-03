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
    """[summary]

    Parameters
    ----------
    ds_in : xr.Dataset
        input
    folder : pathlib.Path
        Target folder for temporary files
    method : str, optional
        Defines if the temporary files are split by an increment along a certain
        dimension("dimension") or by the variables of the dataset ("variables"),
        by default "dimension"
    dim : str, optional
        Dimension to split along (only relevant for `method="dimension"`), by default "time"
    split_interval : int, optional
        Steps along `dim` for each temporary file (only relevant for `method="dimension"`), by default 40
    zarr_write_kwargs : dict, optional
        Kwargs parsed to `xr.to_zarr()`, by default {}
    zarr_read_kwargs : dict, optional
        Kwargs parsed to `xr.open_zarr()`, by default {}
    file_name_pattern : str, optional
        Pattern used to name the temporary files, by default "temp_write_split"
    verbose : bool, optional
        Activates printing, by default False

    Returns
    -------
    ds_out : xr.Dataset
        reloaded dataset, with value identical to `ds_in`
    flist : list
        List of paths to temporary datasets written.
    """

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
