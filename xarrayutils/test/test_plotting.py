from xarrayutils.plotting import plot_line_shaded_std, same_y_range
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
import numpy as np


def test_plot_line_shaded_std():
    a = np.arange(10)
    noise = np.random.rand(len(a))
    ll, ff = plot_line_shaded_std(a, a, noise)
    # Test defaults
    assert ff.get_edgecolor().size == 0
    assert ff.get_alpha() == 0.35
    assert (to_rgb(ll[-1].get_color()) == ff.get_facecolor()[0][0:3]).all()

def test_same_y_range():
    fig, axarr = plt.subplots(ncols=2, nrows=2)

    axarr.flat[0].plot(np.random.rand(10))
    axarr.flat[1].plot((np.random.rand(10)*5)-16)
    axarr.flat[2].plot((np.random.rand(10))-16)
    axarr.flat[3].plot((np.random.rand(10)*5))

    same_y_range(axarr)

    ylims = [ax.get_ylim() for ax in axarr.flat]
    yranges = [lim[1]-lim[0] for lim in ylims]
    assert all([np.isclose(a,yranges[0]) for a in yranges])

# TODO: test passed options...
