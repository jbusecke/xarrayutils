from xarrayutils.plotting import plot_line_shaded_std
from matplotlib.colors import to_rgb
import numpy as np


def test_plot_line_shaded_std():
    a = np.arange(10)
    noise = np.random.rand(len(a))
    ll, ff = plot_line_shaded_std(a, a, noise)
    # Test defaults
    assert ff.get_edgecolor().size == 0
    assert ff.get_alpha() == 0.35
    assert (to_rgb(ll[-1].get_color()) == ff.get_facecolor()[0][0:3]).all()


# TODO: test passed options...
