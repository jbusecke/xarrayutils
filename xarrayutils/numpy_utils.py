"""
Lower Level implementation in numpy and dask
"""

import numpy as np

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
