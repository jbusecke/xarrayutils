
from __future__ import print_function
from future.utils import iteritems
import numpy as np
import scipy.interpolate as spi

"""
Lower Level implementation in numpy and dask
"""


def numpy_block_aggregate(a,blocks,func=np.nanmean):
    """
    Performs efficient block averaging in one or multiple dimensions.

    Parameters
    ----------
    a : array_like
    blocks : list
        List of tuples containing the axis number and interval to average over
    func : function
        Aggregation function. Needs to accept parameter axis (e.g. numpy.nanmean)
        Defaults to numpy.nanmean

    Returns
    -------
    a_block : array_like
        Averaged array

    Examples
    --------
    >>> from xarrayutils.utils import block_mean
    >>> import numpy as np
    >>> a = np.arange(0,12).reshape(2,6)
    >>> print a
    array([[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]])

    >>> # average over 2 by 3 blocks
    >>> b = block_aggregate(a,[(0,2),(1,3)])
    >>> print b
    array([[ 4.  7.]])

    >>> # sum over 2 by 3 blocks
    >>> c = block_aggregate(a,[(0,2),(1,3)],func=np.nansum)
    >>> print c
    array([[24 42]])

    """
    # TODO...this is not really useful..one can always convert to dask and use the
    #coarsen function from the main module
    for bb in blocks:
        axis = bb[0]
        interval = bb[1]
        shape = list(a.shape)

        # define a permutaton that rolls the dimensions so that axis is the last
        new_axis = range(len(a.shape))
        while new_axis[-1] != axis:
            new_axis = np.roll(new_axis,1)

        aligned = np.transpose(a,axes = new_axis)
        new_shape = list(aligned.shape)
        new_shape[-1] = -1
        new_shape = tuple(new_shape)+tuple([interval])
        split = aligned.reshape(*new_shape)
        # applied = np.nanmean(split,axis=len(shape))
        applied = np.apply_along_axis(func,len(shape),split)
        a = np.transpose(applied,axes = np.argsort(new_axis))

    return a

def interp_map_regular_grid(a,x,y,x_i,y_i,method='linear',debug=False,wrap=True):
    """Interpolates 2d fields from regular grid to another regular grid.

    wrap option: pads outer values/coordinates with other side of the array.
    Only works with lon/lat coordinates correctly.

    """
    # TODO these (interp_map*) should eventually end up in xgcm? Maybe not...
    # Pad borders to simulate wrap around coordinates
    # in global maps
    if wrap:

        x = x[[-1]+range(x.shape[0])+[0]]
        y = y[[-1]+range(y.shape[0])+[0]]

        x[0] = x[0]-360
        x[-1] = x[-1]+360

        y[0] = y[0]-180
        y[-1] = y[-1]+180

        a = a[:,[-1]+range(a.shape[1])+[0]]
        a = a[[-1]+range(a.shape[0])+[0],:]

    if debug:
        print('a shape',a.shape)
        print('x shape',x.shape)
        print('y shape',y.shape)
        print('x values',x[:])
        print('y values',y[:])
        print('x_i values',x_i[:])
        print('y_i values',y_i[:])

    xx_i,yy_i = np.meshgrid(x_i,y_i)
    f = spi.RegularGridInterpolator((x, y),a.T,method=method,bounds_error=False)
    int_points = np.vstack((xx_i.flatten(),yy_i.flatten())).T
    a_new = f(int_points)

    return a_new.reshape(xx_i.shape)

def interp_map_irregular_grid(a,x,y,x_i,y_i,method='linear',debug=False):
    """Interpolates fields from any grid to another grid
    !!! Careful when using this on regular grids.
    Results are not unique and it takes forever.
    Use interp_map_regular_grid instead!!!
    """
    xx,yy = np.meshgrid(x,y)
    xx_i,yy_i = np.meshgrid(x_i,y_i)

    # pad margins to avoid nans in the interpolation
    xx = xx[:,[-1]+range(xx.shape[1])+[0]]
    xx = xx[[-1]+range(xx.shape[0])+[0],:]

    xx[:,0] = xx[:,0]-360
    xx[:,-1] = xx[:,-1]+360

    yy = yy[:,[-1]+range(yy.shape[1])+[0]]
    yy = yy[[-1]+range(yy.shape[0])+[0],:]

    yy[0,:] = yy[0,:]-180
    yy[-1,:] = yy[-1,:]+180

    a = a[:,[-1]+range(a.shape[1])+[0]]
    a = a[[-1]+range(a.shape[0])+[0],:]

    if debug:
        print('a shape',a.shape)
        print('x shape',xx.shape)
        print('y shape',yy.shape)
        print('x values',xx[0,:])
        print('y values',yy[:,0])

    points = np.vstack((xx.flatten(),yy.flatten())).T
    values = a.flatten()
    int_points = np.vstack((xx_i.flatten(),yy_i.flatten())).T
    a_new = spi.griddata(points,values,int_points,method=method)

    return a_new.reshape(xx_i.shape)
