"""
Code specific to xarrays created with xmitgcm

"""

import numpy as np
import xarray as xr

def matching_coords(grid,dims):
    #Fill in all coordinates from grid that match the new dims
    new_coords = []
    for kk in grid.coords.keys():
        check = list(grid[kk].dims)
        if all([a in dims for a in check]):
            new_coords.append(kk)

    new_coords_dict = dict([])
    for ii in new_coords:
        new_coords_dict[ii] = grid[ii]
    return new_coords_dict

def gradient1d(grid,ar,dim='i'):
    if 'i' == dim:
            dx = 'dxG'
            swap_dim = 'i_g'
            add_coords = []
    elif 'j' == dim:
            dx = 'dyG'
            swap_dim = 'j_g'
            add_coords = []
    elif 'i_g' == dim:
            dx = 'dxC'
            swap_dim = 'i'
            add_coords = []
    elif 'j_g' == dim:
            dx = 'dyG'
            swap_dim = 'j'
            add_coords = []

    if '_g' in dim:
        # This might have to be expanded with the vertical suffixes
        shift_idx = np.array([-1,0])
    else:
        shift_idx = np.array([0,1])

    new_dims = list(ar.dims)
    new_dims[new_dims.index(dim)] = swap_dim

    new_coords = matching_coords(grid,new_dims)

    diff_x_raw = ar.roll(**{dim:shift_idx[0]})-ar.roll(**{dim:shift_idx[1]})
    grad_x = xr.DataArray(diff_x_raw.data/grid[dx].data,dims=new_dims,coords=new_coords)
    return grad_x


def gradient(grid,ar,recenter=False,debug=False):
    '''
    grid is the complete dataset and a is the dataarray to perform the grad on
    '''

    if debug:
        print ar.dims
    # auto assign the correct gradient in each dimension
    dims = np.array(ar.dims)

    x_dim = dims[np.array(['i' in a[0] for a in dims])][0]
    y_dim = dims[np.array(['j' in a[0] for a in dims])][0]
    # z_dim = dims[np.array(['k' in a[0] for a in dims])]
    # this needs to be somewhat variable...maybe fill the nonexisting gradients
    # with nans? Or have varible output?

    grad_x = gradient1d(grid,ar,dim=x_dim)
    grad_y = gradient1d(grid,ar,dim=y_dim)

    if recenter:
        grad_x = interpolate_from_W_to_C(grid,grad_x)
        grad_y = interpolate_from_S_to_C(grid,grad_y)

    return grad_x,grad_y

def interpolate_from_W_to_C(a,b):
    """
    Interp values from western boundary to cell center
    !!!These need to be generalized
    """
    ref = a.TRAC01
    if len(b.shape)==2:
        ref = ref.mean(dim='time')
    x1 = a.XG.data
    x2 = a.XG.roll(i_g=-1).data
    y1 = b.data
    y2 = b.roll(i_g=-1).data
    x = a.XC.data
    y = y1+(y2-y1)*(x-x1)/(x2-x1)
    out = xr.DataArray(y,dims=ref.dims,coords=ref.coords)
    return out

def interpolate_from_S_to_C(a,b):
    """
    Interp values from western boundary to cell center
    """
    ref = a.TRAC01
    if len(b.shape)==2:
        ref = ref.mean(dim='time')
    x1 = a.YG.data
    x2 = a.YG.roll(j_g=-1).data
    y1 = b.data
    y2 = b.roll(j_g=-1).data
    x = a.YC.data
    y = y1+(y2-y1)*(x-x1)/(x2-x1)

    out = xr.DataArray(y,dims=ref.dims,coords=ref.coords)
    return out
