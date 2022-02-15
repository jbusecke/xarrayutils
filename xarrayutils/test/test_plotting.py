import pytest


plt = pytest.importorskip("matplotlib.pyplot")
from matplotlib.colors import to_rgb
import numpy as np

import xarray as xr
import matplotlib
from xarrayutils.plotting import (
    plot_line_shaded_std,
    same_y_range,
    shaded_line_plot,
    linear_piecewise_scale,
)


def test_plot_line_shaded_std():
    a = np.arange(10)
    noise = np.random.rand(len(a))
    ll, ff = plot_line_shaded_std(a, a, noise)
    # Test defaults
    assert ff.get_edgecolor().size == 0
    assert ff.get_alpha() == 0.35
    assert (to_rgb(ll[-1].get_color()) == ff.get_facecolor()[0][0:3]).all()


@pytest.mark.parametrize("dim", ["member", "time"])
@pytest.mark.parametrize("horizontal", [True, False])
@pytest.mark.parametrize("spreads", [[1, 2], [1], [2, 5, 8]])
@pytest.mark.parametrize("alphas", [[0.5], [0.3], [0.8, 0.5, 0.2]])
def test_shaded_line_plot(dim, horizontal, spreads, alphas):
    x = np.linspace(0, 2 * np.pi, 10)
    y = np.sin(x)
    y_full = np.stack([y + np.random.rand(len(y)) - 0.5 for e in range(6)])
    da = xr.DataArray(y_full, coords=[("member", range(6)), ("time", x)])
    # standard version
    plt.figure()
    ll, ff = shaded_line_plot(
        da, dim, spreads=spreads, alphas=alphas, horizontal=horizontal
    )

    assert isinstance(ll[0], matplotlib.lines.Line2D)
    assert len(ff) == min(len(spreads), len(alphas))

    for f, a in zip(ff, np.flip(alphas)):
        assert isinstance(f, matplotlib.collections.PolyCollection)
        assert f.get_alpha() == a

    # check values
    # find 1d dim
    other_dim = tuple(set(da.dims) - set([dim]))[0]

    if horizontal:
        np.testing.assert_allclose(da[other_dim].data, ll[0].get_xdata())
        np.testing.assert_allclose(da.mean(dim).data, ll[0].get_ydata())
        # would be nice to test the boundaries of the spread, but I dont know how to do that RN
    else:
        np.testing.assert_allclose(da[other_dim].data, ll[0].get_ydata())
        np.testing.assert_allclose(da.mean(dim).data, ll[0].get_xdata())
        # would be nice to test the boundaries of the spread, but I dont know how to do that RN

    # Test exceptions
    with pytest.raises(ValueError) as excinfo:
        shaded_line_plot(da, ["member"], spread_style="bogus")
    assert "Got unknown option ['bogus'] for  `spread_style`" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        shaded_line_plot(da, ["time", "member"])
    assert "1 dimensional after" in str(excinfo.value)


def test_same_y_range():
    fig, axarr = plt.subplots(ncols=2, nrows=2)

    axarr.flat[0].plot(np.random.rand(10))
    axarr.flat[1].plot((np.random.rand(10) * 5) - 16)
    axarr.flat[2].plot((np.random.rand(10)) - 16)
    axarr.flat[3].plot((np.random.rand(10) * 5))

    same_y_range(axarr)

    ylims = [ax.get_ylim() for ax in axarr.flat]
    yranges = [lim[1] - lim[0] for lim in ylims]
    assert all([np.isclose(a, yranges[0]) for a in yranges])


@pytest.mark.parametrize("cut", [10, 40, 50])
@pytest.mark.parametrize("scale", [pytest.param(-2, marks=pytest.mark.xfail), 0, 3, 10])
@pytest.mark.parametrize(
    "axis", ["x", "y", pytest.param("wrong", marks=pytest.mark.xfail)]
)
@pytest.mark.parametrize(
    "scaled_half", ["upper", "lower", pytest.param("wrong", marks=pytest.mark.xfail)]
)
def test_linear_piecewise_scale(cut, scale, axis, scaled_half):
    da_z = xr.DataArray(np.arange(100), dims=["x"])
    da_x = xr.DataArray(np.arange(50), dims=["z"])
    da_data = da_z * xr.ones_like(da_x)
    plt.contourf(da_x, da_z, da_data)

    linear_piecewise_scale(cut, scale, axis=axis, scaled_half=scaled_half)

    if axis == "x":
        if scale != 0:
            assert plt.gca().get_xscale() == "function"
        # this is not a great test. Need something more definitive...
    elif axis == "y":
        if scale != 0:
            assert plt.gca().get_yscale() == "function"
