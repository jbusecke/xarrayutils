from tkinter import Y
import numpy as np
import xarray as xr
import warnings

# import mpl and change the backend before other mpl imports
try:
    import matplotlib as mpl
    from matplotlib.transforms import blended_transform_factory

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl = True
except ImportError:
    raise RuntimeError(
        "The `plotting` module requires `matplotlib`. Install using conda install -c conda-forge matplotlib "
    )

try:
    import gsw
except:
    gsw = None

import string

try:
    import cartopy
except ImportError:
    cartopy = None


def xr_violinplot(ds, ax=None, x_dim="xt_ocean", width=1, color="0.5"):
    """Wrapper of matplotlib violinplot for xarray.DataArray.

    Parameters
    ----------
    ds : xr.DataArray
        Input data.
    ax : matplotlib.axis
        Plotting axis (the default is None).
    x_dim : str
        dimension that defines the x-axis of the
        plot (the default is 'xt_ocean').
    width : float
        Scaling width of each violin (the default is 1).
    color : type
        Color of the violin (the default is '0.5').

    Returns
    -------
    type
        Description of returned object.

    """
    x = ds[x_dim].data.copy()
    y = [ds.loc[{x_dim: xx}].data for xx in x]
    y = [data[~np.isnan(data)] for data in y]
    # check if all are nan
    idx = [len(dat) == 0 for dat in y]
    x = [xx for xx, ii in zip(x, idx) if not ii]
    y = [yy for yy, ii in zip(y, idx) if not ii]

    if ax is None:
        ax = plt.gca()
    vp = ax.violinplot(
        y, x, widths=width, showextrema=False, showmedians=False, showmeans=True
    )
    [item.set_facecolor(color) for item in vp["bodies"]]

    for item in ["cmaxes", "cmins", "cbars", "cmedians", "cmeans"]:
        if item in vp.keys():
            vp[item].set_edgecolor(color)

    return vp


def axis_arrow(ax, x_loc, text, arrowprops={}, **kwargs):
    """Puts an arrow pointing at `x_loc` onto (but outside of ) the xaxis of
    a plot.For now only works on xaxis and on the top. Modify when necessary

    Parameters
    ----------
    ax : matplotlib.axis
        axis to plot on.
    x_loc : type
        Position of the arrow (in units of `ax` x-axis).
    text : str
        Text next to arrow.
    arrowprops: dict
        Additional arguments to pass to arrowprops.
        See mpl.axes.annotate for details.
    kwargs:
        additional keyword arguments passed to ax.annotate

    """
    ar_props = dict(dict(fc="k", lw=1.5, ec=None))
    ar_props.update(arrowprops)

    tform = blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate(
        text,
        xy=[x_loc, 1],
        xytext=(x_loc, 1.25),
        xycoords=tform,
        textcoords=tform,
        ha="center",
        va="center",
        arrowprops=ar_props,
        **kwargs,
    )


def letter_subplots(axes, start_idx=0, box_color=None, labels=None, **kwargs):
    """Adds panel letters in boxes to each element of `axes` in the
    upper left corner.

    Parameters
    ----------
    axes : list, array_like
        List or array of matplotlib axes objects.
    start_idx : type
        Starting index in the alphabet (e.g. 0 is 'a').
    box_color : type
        Color of the box behind each letter (the default is None).
    labels: list
        List of strings used as labels (if None (default), uses lowercase alphabet followed by uppercase alphabet)
    **kwargs : type
        kwargs passed to matplotlib.axis.text

    """
    if labels is None:
        labels = list(string.ascii_letters)

    for ax, letter in zip(axes.flat, labels[start_idx:]):
        t = ax.text(
            0.1,
            0.85,
            letter + ")",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            **kwargs,
        )
        if box_color:
            t.set_bbox(dict(facecolor=box_color, alpha=0.5, edgecolor=None))


def map_util_plot(
    ax, land_color="0.7", coast_color="0.3", lake_alpha=0.5, labels=False
):
    """Helper tool to add good default map to cartopy axes.

    Parameters
    ----------
    ax : cartopy.geoaxes (not sure this is right)
        The axis to plot on (must be a cartopy axis).
    land_color : type
        Color of land fill (the default is '0.7').
    coast_color : type
        Color of costline (the default is '0.3').
    lake_alpha : type
        Transparency of lakes (the default is 0.5).
    labels : type
        Not implemented.

    """
    if cartopy is None:
        raise RuntimeError(
            "Mapping functions require `cartopy`. Install using conda install -c conda-forge cartopy "
        )

    # I could default to plt.gca() for ax, but does it work when I just pass
    # the axis object as positonal argument?
    ax.add_feature(cartopy.feature.LAND, color=land_color)
    ax.add_feature(cartopy.feature.COASTLINE, edgecolor=coast_color)
    ax.add_feature(cartopy.feature.LAKES, alpha=lake_alpha)
    # add option for gridlines and labelling


def same_y_range(axes):
    """Adjusts multiple axes so that the range of y values is the same everywhere, but not the actual values.

    Parameters
    ----------
    axes : np.array
        An array of matplotlib.axes objects produced by e.g. plt.subplots()

    """

    ylims = [ax.get_ylim() for ax in axes.flat]
    yranges = [lim[1] - lim[0] for lim in ylims]
    # find the max range
    yrange_max = np.max(yranges)
    # determine the difference from max range for other ranges
    y_range_missing = [yrange_max - rang for rang in yranges]

    # define new ylims by expanding with  (missing range / 2) at each end
    y_lims_new = [
        np.array(lim) + np.array([-1, 1]) * yrm / 2
        for lim, yrm in zip(ylims, y_range_missing)
    ]

    for ax, lim in zip(axes.flat, y_lims_new):
        ax.set_ylim(lim)


def center_lim(ax, which="y"):
    if which == "y":
        lim = np.array(ax.get_ylim())
        ax.set_ylim(np.array([-1, 1]) * abs(lim).max())
    elif which == "x":
        lim = np.array(ax.get_xlim())
        ax.set_xlim(np.array([-1, 1]) * abs(lim).max())
    elif which in ["xy", "yx"]:
        center_lim(ax, "x")
        center_lim(ax, "y")
    else:
        raise ValueError("`which` is not in (`x,`y`, `xy`) found %s" % which)


def depth_logscale(ax, yscale=400, ticks=None):
    if ticks is None:
        ticks = [0, 100, 250, 500, 1000, 2500, 5000]
    ax.set_yscale("symlog", linthreshy=yscale)
    ticklabels = [str(a) for a in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.invert_yaxis()


def shaded_line_plot(
    da,
    dim,
    ax=None,
    horizontal=True,
    spreads=None,
    alphas=[0.25, 0.4],
    spread_style="std",
    line_kwargs=dict(),
    fill_kwargs=dict(),
    **kwargs,
):
    """Produces a line plot with shaded intervals based on the spread of `da` in `dim`.

    Parameters
    ----------
    da : xr.DataArray
        The input data. Needs to be 2 dimensional, so that when `dim` is reduced, it is a line plot.

    dim : str
        Dimension of `da` which is used to calculate spread

    ax : matplotlib.axes
        Matplotlib axes object to plot on (the default is plt.gca()).

    horizontal : bool
        Determines if the plot is horizontal or vertical (e.g. x is plotted
        on the y-axis).

    spread : np.array, optional
        Values specifying the 'spread-values', dependent on `spread_style`. Defaults to shading the
        range of 1 and 2 standard deviations in `dim`

    alpha: np.array, optional
        Transparency values of the shaded ranges. Defaults to [0.5,0.15].

    spread_style : str
        Metric used to define spread on `dim`.
        Options:
            'std': Calculates standard deviation along `dim` and shading indicates multiples of std centered on the mean

            'quantile': Calculates quantile ranges. An input of `spread=[0.2,0.5]` would show an inner shading for
            the 40th-60th percentile, and an outer shading for the 25th-75th percentile, centered on the 50th quantile (~median).
            Must be within [0,100].

    line_kwargs : dict
        optional parameters for line plot.

    fill_kwargs : dict
        optional parameters for std fill plot.

    **kwargs
        Keyword arguments passed to both line plot and fill_between.

    Example
    ------


    """
    # check input
    if isinstance(spreads, float) or isinstance(spreads, int):
        spreads = [spreads]

    if isinstance(alphas, float):
        alphas = [alphas]

    if isinstance(dim, float):
        dim = [dim]

    # set axis
    if not ax:
        ax = plt.gca()

    # Option to plot a straight line when the dim is not present (TODO)

    # check if the data is 2 dimensional
    dims = da.mean(dim).dims
    if len(dims) != 1:
        raise ValueError(
            f"`da` must be 1 dimensional after reducing over {dim}. Found {dims}"
        )

    # assemble plot elements
    xdim = dims[0]
    x = da[xdim]

    # define the line plot values
    if spread_style == "std":
        y = da.mean(dim)
        if spreads is None:
            spreads = [1, 3]
    elif spread_style in ["quantile", "percentile"]:
        y = da.quantile(0.5, dim)
        if spreads is None:
            spreads = [0.5, 0.8]
    else:
        raise ValueError(
            f"Got unknown option ['{spread_style}'] for  `spread_style`. Supported options are : ['std', 'quantile']"
        )

    # set line kwargs
    line_defaults = {}
    line_defaults.update(line_kwargs)

    if horizontal:
        ll = ax.plot(x, y, **line_defaults)
    else:
        ll = ax.plot(y, x, **line_defaults)

    # now loop over the spreads:
    fill_defaults = {"facecolor": ll[-1].get_color(), "edgecolor": "none"}

    # Apply defaults but respect input
    fill_defaults.update(fill_kwargs)
    ff = []

    spreads = list(np.flip(spreads))
    alphas = list(np.flip(alphas))
    # np.flip(this ensures that the shadings are drawn from outer to inner otherwise they blend too much into each other

    for spread, alpha in zip(spreads, alphas):
        f_kwargs = {k: v for k, v in fill_defaults.items()}
        f_kwargs["alpha"] = alpha

        if spread_style == "std":
            y_std = da.std(dim)  # i could probably precompute that.
            y_spread = y_std * spread
            y_lower = y - (y_spread / 2)
            y_upper = y + (y_spread / 2)

        elif spread_style in ["quantile", "percentile"]:
            y_lower = da.quantile(0.5 - (spread / 2), dim)
            y_upper = da.quantile(0.5 + (spread / 2), dim)

        if horizontal:
            ff.append(ax.fill_between(x.data, y_lower.data, y_upper.data, **f_kwargs))
        else:
            ff.append(ax.fill_betweenx(x.data, y_lower.data, y_upper.data, **f_kwargs))
    return ll, ff


def plot_line_shaded_std(
    x, y, std_y, horizontal=True, ax=None, line_kwargs=dict(), fill_kwargs=dict()
):
    """Plot wrapper to draw line for y and shaded patch according to std_y.
    The shading represents one std on each side of the line...

    Parameters
    ----------
    x : numpy.array or xr.DataArray
        Coordinate.
    y : numpy.array or xr.DataArray
        line data.
    std_y : numpy.array or xr.DataArray
        std corresponding to y.
    horizontal : bool
        Determines if the plot is horizontal or vertical (e.g. x is plotted
        on the y-axis).
    ax : matplotlib.axes
        Matplotlib axes object to plot on (the default is plt.gca()).
    line_kwargs : dict
        optional parameters for line plot.
    fill_kwargs : dict
        optional parameters for std fill plot.

    Returns
    -------
    (ll, ff)
        Tuple of line and patch objects.

    """

    warnings.warn(
        "This is an outdated function. Use `shaded_line_plot` instead",
        DeprecationWarning,
    )

    line_defaults = {}

    # Set plot defaults into the kwargs
    if not ax:
        ax = plt.gca()

    # Apply defaults but respect input
    line_defaults.update(line_kwargs)

    if horizontal:
        ll = ax.plot(x, y, **line_defaults)
    else:
        ll = ax.plot(y, x, **line_defaults)

    fill_defaults = {
        "facecolor": ll[-1].get_color(),
        "alpha": 0.35,
        "edgecolor": "none",
    }

    # Apply defaults but respect input
    fill_defaults.update(fill_kwargs)

    if horizontal:
        ff = ax.fill_between(x, y - std_y, y + std_y, **fill_defaults)
    else:
        ff = ax.fill_betweenx(x, y - std_y, y + std_y, **fill_defaults)
    return ll, ff


def box_plot(box, ax=None, split_detection="True", **kwargs):
    """plots box despite coordinate discontinuities.
    INPUT
    -----
    box: np.array
        Defines the box in the coordinates of the current axis.
        Describing the box corners [x1, x2, y1, y2]
    ax: matplotlib.axis
        axis for plotting. Defaults to plt.gca()
    kwargs: optional
        anything that can be passed to plot can be put as kwarg
    """

    if len(box) != 4:
        raise RuntimeError(
            "'box' must be a 4 element np.array, \
            describing the box corners [x1, x2, y1, y2]"
        )
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    x_split = False
    y_split = False

    if ax is None:
        ax = plt.gca()

    if split_detection:
        if np.diff([box[0], box[1]]) < 0:
            x_split = True

        if np.diff([box[2], box[3]]) < 0:
            y_split = True

    if y_split and not x_split:
        ax.plot(
            [box[0], box[0], box[1], box[1], box[0]],
            [ylim[1], box[2], box[2], ylim[1], ylim[1]],
            **kwargs,
        )

        ax.plot(
            [box[0], box[0], box[1], box[1], box[0]],
            [ylim[0], box[3], box[3], ylim[0], ylim[0]],
            **kwargs,
        )

    elif x_split and not y_split:
        ax.plot(
            [xlim[1], box[0], box[0], xlim[1], xlim[1]],
            [box[2], box[2], box[3], box[3], box[2]],
            **kwargs,
        )

        ax.plot(
            [xlim[0], box[1], box[1], xlim[0], xlim[0]],
            [box[2], box[2], box[3], box[3], box[2]],
            **kwargs,
        )

    elif x_split and y_split:
        ax.plot([xlim[1], box[0], box[0]], [box[2], box[2], ylim[1]], **kwargs)

        ax.plot([xlim[0], box[1], box[1]], [box[2], box[2], ylim[1]], **kwargs)

        ax.plot([xlim[1], box[0], box[0]], [box[3], box[3], ylim[0]], **kwargs)

        ax.plot([xlim[0], box[1], box[1]], [box[3], box[3], ylim[0]], **kwargs)

    elif not x_split and not y_split:
        ax.plot(
            [box[0], box[0], box[1], box[1], box[0]],
            [box[2], box[3], box[3], box[2], box[2]],
            **kwargs,
        )


def dict2box(di, xdim="lon", ydim="lat"):
    return np.array([di[xdim].start, di[xdim].stop, di[ydim].start, di[ydim].stop])


def box_plot_dict(di, xdim="lon", ydim="lat", **kwargs):
    """plot box from xarray selection dict e.g.
    `{'xdim':slice(a, b), 'ydim':slice(c,d), ...}`"""

    # extract box from dict
    box = dict2box(di, xdim=xdim, ydim=ydim)
    # plot
    box_plot(box, **kwargs)


def draw_dens_contours_teos10(
    sigma="sigma0",
    add_labels=True,
    ax=None,
    density_grid=20,
    dens_interval=1.0,
    salt_on_x=True,
    slim=None,
    tlim=None,
    contour_kwargs={},
    c_label_kwargs={},
    **kwargs,
):
    """draws density contours on the current plot.
    Assumes that the salinity and temperature values are given as SA and CT.
    Needs documentation..."""
    if gsw is None:
        raise RuntimeError(
            "`gsw` is not available. Install with `conda install -c conda-forge gsw`"
        )

    if ax is None:
        ax = plt.gca()

    if sigma not in ["sigma%i" % s for s in range(5)]:
        raise ValueError(
            "Sigma function has to be one of `sigma0`...`sigma4` \
                         is: %s"
            % (sigma)
        )

    # get salt (default: xaxis) and temp (default: yaxis) limits
    if salt_on_x:
        if not (slim is None):
            slim = ax.get_xlim()
        if not (tlim is None):
            tlim = ax.get_ylim()
        x = np.linspace(*(slim + [density_grid]))
        y = np.linspace(*(tlim + [density_grid]))
    else:
        if not tlim:
            tlim = ax.get_xlim()
        if not slim:
            slim = ax.get_ylim()
        x = np.linspace(*(slim + [density_grid]))
        y = np.linspace(*(tlim + [density_grid]))

    if salt_on_x:
        ss, tt = np.meshgrid(x, y)
    else:
        tt, ss = np.meshgrid(x, y)

    sigma_func = getattr(gsw, sigma)

    sig = sigma_func(ss, tt)

    levels = np.arange(np.floor(sig.min()), np.ceil(sig.max()), dens_interval)

    c_kwarg_defaults = dict(
        levels=levels, colors="0.4", linestyles="--", linewidths=0.5
    )
    c_kwarg_defaults.update(kwargs)
    c_kwarg_defaults.update(contour_kwargs)

    c_label_kwarg_defaults = dict(fmt="%.02f")
    c_label_kwarg_defaults.update(kwargs)
    c_label_kwarg_defaults.update(c_label_kwargs)

    ch = ax.contour(x, y, sig, **c_kwarg_defaults)
    ax.clabel(ch, **c_label_kwarg_defaults)

    if add_labels:
        plt.text(
            0.05,
            0.05,
            "$\sigma_{%s}$" % (sigma[-1]),
            fontsize=14,
            verticalalignment="center",
            horizontalalignment="center",
            transform=ax.transAxes,
            color=c_kwarg_defaults["colors"],
        )


def tsdiagram(
    salt,
    temp,
    color=None,
    size=None,
    lon=None,
    lat=None,
    pressure=None,
    convert_teos10=True,
    ts_kwargs={},
    ax=None,
    fig=None,
    draw_density_contours=True,
    draw_cbar=True,
    add_labels=True,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    if fig is None:
        fig = plt.gcf()

    if convert_teos10:
        temp_label = "Conservative Temperature [$^{\circ}C$]"
        salt_label = "Absolute Salinity [$g/kg$]"
        if any([a is None for a in [lon, lat, pressure]]):
            raise ValueError(
                "when converting to teos10 variables, \
                             input for lon, lat and pressure is needed"
            )
        else:
            salt = gsw.SA_from_SP(salt, pressure, lon, lat)
            temp = gsw.CT_from_pt(salt, temp)
    else:
        temp_label = "Potential Temperature [$^{\circ}C$]"
        salt_label = "Practical Salinity [$g/kg$]"

    if add_labels:
        ax.set_xlabel(salt_label)
        ax.set_ylabel(temp_label)

    scatter_kw_defaults = dict(s=size, c=color)
    scatter_kw_defaults.update(kwargs)
    s = ax.scatter(salt, temp, **scatter_kw_defaults)
    if draw_density_contours:
        draw_dens_contours_teos10(ax=ax, **ts_kwargs)
    if draw_cbar and color is not None:
        if isinstance(color, str) or isinstance(color, tuple):
            pass
        elif (
            isinstance(color, list)
            or isinstance(color, np.ndarray)
            or isinstance(color, xr.DataArray)
        ):
            fig.colorbar(s, ax=ax)
        else:
            raise RuntimeError("`color` not recognized. %s" % type(color))
    return s


def linear_piecewise_scale(
    cut, scale, ax=None, axis="y", scaled_half="upper", add_cut_line=False
):
    """This function sets a piecewise linear scaling for a given axis to highlight e.g. processes in the upper ocean vs deep ocean.

    Parameters
    ----------
    cut : float
        value along the chosen axis used as transition between the two linear scalings.
    scale : float
        scaling coefficient for the chosen axis portion (determined by `axis` and `scaled_half`).
        A higher number means the chosen portion of the axis will be more compressed. Must be positive. 0 means no compression.
    ax : matplotlib.axis, optional
        The plot axis object. Defaults to current matplotlib axis
    axis : str, optional
        Which axis of the plot to act on.
        * 'y' (Default)
        * 'x'
    scaled_half: str, optional
        Determines which half of the axis is scaled (compressed).
        * 'upper' (default). Values larger than `cut` are compressed
        * 'lower'. Values smaller than `cut` are compressed


    Returns
    -------
    ax_scaled : matplotlib.axis

    """

    if ax is None:
        ax = plt.gca()

    if scale < 0:
        raise ValueError(f"`Scale can not be negative. Got value of {scale}")

    if scale == 0:
        # do nothing
        return ax
    else:
        if scaled_half == "upper":

            def inverse(x):
                return np.piecewise(
                    x,
                    [x <= cut, x > cut],
                    [lambda x: x + (scale * (x - cut)), lambda x: x],
                )

            def forward(x):
                return np.piecewise(
                    x,
                    [x <= cut, x > cut],
                    [lambda x: x + (scale * (x - cut)), lambda x: x],
                )

        elif scaled_half == "lower":

            def inverse(x):
                return np.piecewise(
                    x,
                    [x >= cut, x < cut],
                    [lambda x: x + (scale * (x - cut)), lambda x: x],
                )

            def forward(x):
                return np.piecewise(
                    x,
                    [x >= cut, x < cut],
                    [lambda x: x + (scale * (x - cut)), lambda x: x],
                )

        else:
            raise ValueError(
                f"`scaled_half` value not recognized. Must be ['upper', 'lower']. Got {scaled_half}"
            )

        if axis == "y":
            axlim = ax.get_ylim()
            ax.set_yscale("function", functions=(forward, inverse))
            ax.set_ylim(axlim)
        elif axis == "x":
            axlim = ax.get_xlim()
            ax.set_xscale("function", functions=(forward, inverse))
            ax.set_xlim(axlim)
        else:
            raise ValueError(
                f"`axis` value not recognized. Must be ['x', 'y']. Got {axis}"
            )

        return ax
