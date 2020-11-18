import xarray as xr
from xarrayutils.weighted_operations import weighted_mean
from .utils import xr_detrend


def calculate_ninox_index(
    ds_surf,
    area,
    timedim="time",
    xdim="xt_ocean",
    ydim="yt_ocean",
    clim_period=None,
    detrend=False,
):
    """Calculates NINOx index following the methodology in
    `climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni`
    If detrend is true, a linear trend is removed for each gridpoint
    seperately, before applying the processing.
    Note that dask arrays cannot be chunked along `timedim`.
    Use (ds_surf.chunk({timedim:-1}), before processing.
    Warning: This can significantly increase memory usage,
    and chunking along another dimension might be necessary.
    """

    if detrend:
        ds_surf = xr_detrend(ds_surf, dim=timedim)

    # (a) Compute area averaged total SST from Niño X region
    sst_mean = weighted_mean(ds_surf, area, dim=[xdim, ydim]).load()

    # (b) Compute monthly climatology (e.g., 1950-1979) for area averaged
    # total SST from Niño X region,
    # and subtract climatology from area averaged total SST time series to
    # obtain anomalies;

    if clim_period:
        sst_clim = sst_mean.loc[{timedim: clim_period}]
    else:
        sst_clim = sst_mean
    # !!! generalize this for weeks, days if the data is of higher resolution?
    # for now it only works for monthly data

    clim_time_spec = "%s.month" % timedim
    sst_clim = sst_clim.groupby(clim_time_spec).mean("time")
    sst_anomaly = sst_mean.groupby(clim_time_spec) - sst_clim

    # (c) Smooth the anomalies with a 5-month running mean
    steps_in_month = 1
    sst_anomaly_smooth = (
        sst_anomaly.rolling(**{timedim: 5 * steps_in_month, "center": True})
        .construct("win")
        .mean("win")
    )

    # Normalize the smoothed values by its standard deviation over the
    # climatological period
    if clim_period:
        clim_std = sst_anomaly_smooth.loc[{timedim: clim_period}].std([timedim])
    else:
        clim_std = sst_anomaly_smooth.std(timedim)

    index = sst_anomaly_smooth / clim_std

    return index


def extract_climate_indicies(
    ds,
    timedim="time",
    depth_dim="st_ocean",
    xdim="xt_ocean",
    ydim="yt_ocean",
    temp_var="temp",
    area_coord="area_t",
    print_map=False,
    **kwargs
):
    """Calculates various climate indicies from an xarray dataset.

    kwargs are passed to the climate indicies processing functions.
    Most notable right now would be the `detrend` option (disabled by default),
    which removes a linear trend from each gridbox before processing.
    """
    # !!! TODO, make this work with other lon conventions like -180-180 etc.

    # NINO boxes
    NINO_boxes = {
        "NINO1+2": {xdim: slice(-90, -80), ydim: slice(-10, 0)},
        "NINO3": {xdim: slice(-150, -90), ydim: slice(-5, 5)},
        "NINO3.4": {xdim: slice(-170, -120), ydim: slice(-5, 5)},
        "NINO4": {xdim: slice(-200, -150), ydim: slice(-5, 5)},
    }

    # visualize the data to make sure that all indicies areas are covered
    if print_map:
        from xarrayutils.plotting import box_plot_dict
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        surface_fld = ds[temp_var]

        for slice_dim in [timedim, depth_dim]:
            if slice_dim in ds.dims:
                surface_fld = surface_fld[{slice_dim: 0}]

        ax = plt.axes(projection=ccrs.Robinson(180))
        surface_fld.plot(ax=ax, transform=ccrs.PlateCarree())
        for nb in NINO_boxes.keys():
            box_plot_dict(
                NINO_boxes[nb],
                xdim=xdim,
                ydim=ydim,
                ax=ax,
                transform=ccrs.PlateCarree(),
            )
        ax.set_global()
        ax.coastlines()

    # calulate indicies
    ds_indicies = xr.Dataset()
    for nb in NINO_boxes.keys():
        print("Calculating %s index" % nb)
        box = ds.sel(**NINO_boxes[nb])
        if depth_dim in ds.dims:
            box = box[{depth_dim: 0}]
        ds_indicies[nb] = calculate_ninox_index(
            box[temp_var], box[area_coord], **kwargs
        )

    # Calculate additional ENSO inicies
    print("Calculating TNI index")
    ds_indicies["TNI"] = ds_indicies["NINO4"] - ds_indicies["NINO1+2"]
    return ds_indicies
