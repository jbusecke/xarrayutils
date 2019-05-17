import xarray as xr


def detect_dtype(aa):
    if isinstance(aa, xr.Dataset):
        dtype = aa[list(aa.data_vars)[0]].dtype
        print(
            "No `dtype` chosen. Input is Dataset. \
        Defaults to %s"
            % dtype
        )
    elif isinstance(aa, xr.DataArray):
        dtype = aa.dtype
    return dtype
