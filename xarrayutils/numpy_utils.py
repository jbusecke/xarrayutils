import numpy as np
import scipy.interpolate as spi


def interp_map_regular_grid(a, x, y, x_i, y_i, method="linear", debug=False, wrap=True):
    """Interpolates 2d fields from regular grid to another regular grid.

    wrap option: pads outer values/coordinates with other side of the array.
    Only works with lon/lat coordinates correctly.

    """
    # TODO these (interp_map*) should eventually end up in xgcm? Maybe not...
    # Pad borders to simulate wrap around coordinates
    # in global maps
    if wrap:

        x = x[[-1] + list(range(x.shape[0])) + [0]]
        y = y[[-1] + list(range(y.shape[0])) + [0]]

        x[0] = x[0] - 360
        x[-1] = x[-1] + 360

        y[0] = y[0] - 180
        y[-1] = y[-1] + 180

        a = a[:, [-1] + list(range(a.shape[1])) + [0]]
        a = a[[-1] + list(range(a.shape[0])) + [0], :]

    if debug:
        print("a shape", a.shape)
        print("x shape", x.shape)
        print("y shape", y.shape)
        print("x values", x[:])
        print("y values", y[:])
        print("x_i values", x_i[:])
        print("y_i values", y_i[:])

    xx_i, yy_i = np.meshgrid(x_i, y_i)
    f = spi.RegularGridInterpolator((x, y), a.T, method=method, bounds_error=False)
    int_points = np.vstack((xx_i.flatten(), yy_i.flatten())).T
    a_new = f(int_points)

    return a_new.reshape(xx_i.shape)


def interp_map_irregular_grid(a, x, y, x_i, y_i, method="linear", debug=False):
    """Interpolates fields from any grid to another grid
    !!! Careful when using this on regular grids.
    Results are not unique and it takes forever.
    Use interp_map_regular_grid instead!!!
    """
    xx, yy = np.meshgrid(x, y)
    xx_i, yy_i = np.meshgrid(x_i, y_i)

    # pad margins to avoid nans in the interpolation
    xx = xx[:, [-1] + list(range(xx.shape[1])) + [0]]
    xx = xx[[-1] + list(range(xx.shape[0])) + [0], :]

    xx[:, 0] = xx[:, 0] - 360
    xx[:, -1] = xx[:, -1] + 360

    yy = yy[:, [-1] + list(range(yy.shape[1])) + [0]]
    yy = yy[[-1] + list(range(yy.shape[0])) + [0], :]

    yy[0, :] = yy[0, :] - 180
    yy[-1, :] = yy[-1, :] + 180

    a = a[:, [-1] + list(range(a.shape[1])) + [0]]
    a = a[[-1] + list(range(a.shape[0])) + [0], :]

    if debug:
        print("a shape", a.shape)
        print("x shape", xx.shape)
        print("y shape", yy.shape)
        print("x values", xx[0, :])
        print("y values", yy[:, 0])

    points = np.vstack((xx.flatten(), yy.flatten())).T
    values = a.flatten()
    int_points = np.vstack((xx_i.flatten(), yy_i.flatten())).T
    a_new = spi.griddata(points, values, int_points, method=method)

    return a_new.reshape(xx_i.shape)
