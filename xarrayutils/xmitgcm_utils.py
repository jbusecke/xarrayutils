from __future__ import print_function
from future.utils import iteritems
import numpy as np
import xarray as xr
import warnings
import dask.array as da_ar

from .utils import aggregate
"""
Code specific to xarrays created with xmitgcm

"""

def gradient1d(grid,data,axis):
    """Calculate gradient along single axis.
    PARAMETERS
    ----------
    grid : xgcm.Grid
    data : xarray.DataArray
        The data to interpolate
    axis: {'X', 'Y'}
        The name of the axis along which to calculate
    RETURNS
    -------
    da_i : xarray.DataArray
        gradient along axis
    """
    dx = get_dx(grid,data,axis)
    mask = get_hfac(grid,dx)
    return grid.diff(data, axis)/dx*mask

def gradient(grid,data,interpolate=False):
    """compute the gradient in x,y direction (optional interpolation between grid variables).

        PARAMETERS
        ----------
        grid : xgcm.Grid
        data : xarray.DataArray
            The data to interpolate
        axis: {'X', 'Y'}
            The name of the axis along which to calculate

        interpolate : bool, optional
            Should values be interpreted from grid face to center or vice versa.

        RETURNS
        -------
        da_i : xarray.DataArray
            gradient along axis

        TODO
        ----
            Currently only performs a first order forward gradient.
            It could be good to implement different order gradients later
        """

    grad_x = gradient1d(grid,data,'X')
    grad_y = gradient1d(grid,data,'Y')

    if interpolate:
        grad_x = grid.interp(grad_x,'X')
        grad_y = grid.interp(grad_y,'Y')

    return grad_x,grad_y

# Silly functions
def get_hfac(grid,da):
    """Figure out the correct hfac given array dimensions."""
    hfac = None
    if 'XC' in da.dims and 'YC' in da.dims and 'hFacC' in grid._ds:
        hfac = grid._ds.hFacC
    if 'XC' in da.dims and 'YG' in da.dims and 'hFacS' in grid._ds:
        hfac = grid._ds.hFacS
    if 'XG' in da.dims and 'YC' in da.dims and 'hFacW' in grid._ds:
        hfac = grid._ds.hFacW
    return hfac

def get_dx(grid,da,axis):
    """Figure out the correct hfac given array dimensions."""
    dx = None
    if axis == 'X':
        if 'XC' in da.dims and 'YC' in da.dims and 'dxC' in grid._ds:
            dx = grid._ds.dxC
        if 'XG' in da.dims and 'YC' in da.dims and 'dxG' in grid._ds:
            hfac = grid._ds.dxG

        # Is this right or is there a different dxC for the vorticity cell?
        if 'XC' in da.dims and 'YG' in da.dims and 'dxC' in grid._ds:
            dx = grid._ds.dxC
        if 'XG' in da.dims and 'YG' in da.dims and 'dxG' in grid._ds:
            hfac = grid._ds.dxG

    elif axis == 'Y':
        if 'XC' in da.dims and 'YC' in da.dims and 'dyC' in grid._ds:
            dx = grid._ds.dyC
        if 'XC' in da.dims and 'YG' in da.dims and 'dyG' in grid._ds:
            hfac = grid._ds.dyG

        # Is this right or is there a different dxC for the vorticity cell?
        if 'XG' in da.dims and 'YC' in da.dims and 'dyC' in grid._ds:
            dx = grid._ds.dyC
        if 'XG' in da.dims and 'YG' in da.dims and 'dyG' in grid._ds:
            hfac = grid._ds.dyG
    return dx

def matching_coords(grid,dims):
    #Fill in all coordinates from grid that match the new dims
    c = []
    for kk in grid.coords.keys():
        check = list(grid[kk].dims)
        if all([a in dims for a in check]):
            c.append(kk)

    c_dict = dict([])
    for ii in c:
        c_dict[ii] = grid[ii]
    return c_dict

# Discontinued functions
def interpolate_from_W_to_C(grid,x):
    raise RuntimeError('not supported anymore, use xgcm')

def interpolate_from_S_to_C(grid,x):
    raise RuntimeError('not supported anymore, use xgcm')

def interpolateGtoC(grid,x,dim='x',debug=False):
    raise RuntimeError('not supported anymore, use xgcm')

def raw_diff(grid,x,dim,method='pad',wrap_ref=360.0,shift_grid=False):
    raise RuntimeError('not supported anymore, use xgcm')



    # def gradient1d(grid,ar,dim='i'):
    #     if 'i' == dim:
    #             dx = 'dxG'
    #             swap_dim = 'i_g'
    #             add_coords = []
    #     elif 'j' == dim:
    #             dx = 'dyG'
    #             swap_dim = 'j_g'
    #             add_coords = []
    #     elif 'i_g' == dim:
    #             dx = 'dxC'
    #             swap_dim = 'i'
    #             add_coords = []
    #     elif 'j_g' == dim:
    #             dx = 'dyG'
    #             swap_dim = 'j'
    #             add_coords = []
    #
    #     if '_g' in dim:
    #         # This might have to be expanded with the vertical suffixes
    #         shift_idx = np.array([-1,0])
    #     else:
    #         shift_idx = np.array([0,1])
    #
    #     new_dims = list(ar.dims)
    #     new_dims[new_dims.index(dim)] = swap_dim
    #
    #     c = matching_coords(grid,new_dims)
    #
    #     dx_data = grid[dx].data
    #     diff_x_raw = ar.roll(**{dim:shift_idx[0]}).data-ar.roll(**{dim:shift_idx[1]}).data
    #     # This needs to be implemented with custom diff (using ;wrap option)
    #     # Also this needs to land on the new
    #
    #     grad_x = xr.DataArray(diff_x_raw/dx_data,dims=new_dims,coords=c)
    #     return grad_x

    # def reassign_grid(grid,x,old,new,debug=False):
    #     dims = list(x.dims)
    #     dims[dims.index(old)] = new
    #     if new in list(grid.dims):
    #         rename_switch = False
    #     else:
    #         rename_switch = True
    #
    #     coords = dict([])
    #     for ii in dims:
    #         if ii == new and rename_switch:
    #             coords[ii] = grid[old].rename({old:new})
    #         else:
    #             coords[ii] = grid[ii]
    #
    #     return xr.DataArray(x.data,coords=coords,dims=dims)
