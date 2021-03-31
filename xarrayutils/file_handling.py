import pathlib
import warnings
import functools
import numpy as np
import xarray as xr


try:
    from fastprogress.fastprogress import progress_bar

    fastprogress = 1
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


def maybe_create_folder(path):
    p = pathlib.Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    else:
        warnings.warn(f"Folder {path} does already exist.", UserWarning)
    return p


def total_nested_size(nested):
    """Calculate the size of a nested dict full of xarray objects

    Parameters
    ----------
    nested : dict
        Input dictionary. Can have arbitrary nesting levels

    Returns
    -------
    float
        total size in bytes
    """
    size = []

    def _size(obj):
        if not (isinstance(obj, xr.Dataset) or isinstance(obj, xr.DataArray)):
            return {k: _size(v) for k, v in obj.items()}
        else:
            size.append(obj.nbytes)

    _size(nested)

    return np.sum(np.array(size))


def _maybe_pathlib(path):
    if not isinstance(path, pathlib.PosixPath):
        path = pathlib.Path(path)
    return path


def _file_iszarr(path):
    if ".nc" in str(path):
        zarr = False
    elif ".zarr" in str(path):
        zarr = True
    return zarr


def file_exist_check(filepath, check_zarr_consolidated_complete=True):
    """Check if a file exists, with some extra checks for zarr files

    Parameters
    ----------
    filepath : path
        path to the file to check
    check_zarr_consolidated_complete : bool, optional
        Check if .zmetadata file was written (consolidated metadata), by default True
    """
    filepath = _maybe_pathlib(filepath)

    zarr = _file_iszarr(filepath)

    basic_check = filepath.exists()
    if zarr and check_zarr_consolidated_complete:
        check = filepath.joinpath(".zmetadata").exists()
    else:
        check = True

    return check and basic_check


def checkpath(func):
    @functools.wraps(func)
    def wrapper_checkpath(*args, **kwargs):
        ds = args[0]
        path = _maybe_pathlib(args[1])

        # Do something before
        overwrite = kwargs.pop("overwrite", False)
        check_zarr_consolidated_complete = kwargs.pop(
            "check_zarr_consolidated_complete", False
        )
        reload_saved = kwargs.pop("reload_saved", True)
        write_kwargs = kwargs.pop("write_kwargs", {})
        load_kwargs = kwargs.pop("load_kwargs", {})

        load_kwargs.setdefault("use_cftime", True)
        load_kwargs.setdefault("consolidated", True)
        write_kwargs.setdefault("consolidated", load_kwargs["consolidated"])

        zarr = _file_iszarr(path)
        check = file_exist_check(
            path, check_zarr_consolidated_complete=check_zarr_consolidated_complete
        )

        # check for the consolidated stuff... or just rewrite it?
        if check and not overwrite:
            print(f"File [{str(path)}] already exists. Skipping.")
        else:
            # the file might still exist (inclomplete) and then needs to be removed.
            if path.exists():
                print(f"Removing file {str(path)}")
                if zarr:
                    shutil.rmtree(path)
                else:
                    path.unlink()

            func(ds, path, **write_kwargs)

        # Do something after
        ds_out = ds
        if reload_saved:
            print(f"$ Reloading file")
            consolidated = load_kwargs.pop("consolidated")
            if not zarr:
                ds_out = xr.open_dataset(str(path), **load_kwargs)
            else:
                ds_out = xr.open_zarr(
                    str(path), consolidated=consolidated, **load_kwargs
                )

        return ds_out

    return wrapper_checkpath


@checkpath
def write(
    ds,
    path,
    print_size=True,
    consolidated=True,
    **kwargs,
):
    """Convenience function to save large datasets.
    Performs the following additional steps (compared to e.g. xr.to_netcdf() or xr.to_zarr())

    1. Checks for existing files (with special checks for zarr files)
    2. Handles existing files via `overwrite` argument.
    3. Checks attributes for incompatible values
    4. Optional: Prints size of saved dataset
    4. Optional: Returns the saved dataset loaded from disk (e.g. for quality control)

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    path : pathlib.Path
        filepath to save to. Ending determines the output type (`.nc` for netcdf, `.zarr` for zarr)
    print_size : bool, optional
        If true prints the size of the dataset before saving, by default True
    reload_saved : bool, optional
        If true the returned datasets is opened from the written file,
        otherwise the input is returned, by default True
    open_kwargs : dict
        Arguments passed to the reloading function (either xr.open_dataset or xr.open_zarr based on filename)
    write_kwargs : dict
        Arguments passed to the writing function (either xr.to_netcdf or xr.to_zarr based on filename)
    overwrite : bool, optional
        If True, overwrite existing files, by default False
    check_zarr_consolidated_complete: bool, optional
        If True check if `.zmetadata` is present in zarr store, and overwrite if not present, by default False

    Returns
    -------
    xr.Dataset
        Returns either the unmodified input dataset or a reloaded version from the written file

    """

    for k, v in ds.attrs.items():
        if isinstance(v, xr.Dataset) or isinstance(v, xr.DataArray):
            raise RuntimeError(f"Found an attrs ({k}) in with xarray values:{v}.")

    zarr = _file_iszarr(path)

    if print_size:
        print(f"$ Saving {ds.nbytes/1e9}GB to {path}")

    if zarr:
        ds.to_zarr(path, consolidated=consolidated, **kwargs)
    else:
        ds.to_netcdf(path, **kwargs)
