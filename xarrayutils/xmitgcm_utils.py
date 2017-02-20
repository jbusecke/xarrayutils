"""
Code specific to xarrays created with xmitgcm

"""

import numpy as np
import xarray as xr
import warnings
import dask.array as da_ar
from utils import aggregate


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

    dx_data = grid[dx].data
    diff_x_raw = ar.roll(**{dim:shift_idx[0]}).data-ar.roll(**{dim:shift_idx[1]}).data
    # This needs to be implemented with custom diff (using ;wrap option)
    # Also this needs to land on the new
    grad_x = xr.DataArray(diff_x_raw/dx_data,dims=new_dims,coords=new_coords)
    return grad_x


def gradient(grid,ar,recenter=False,debug=False):
    '''
    wrapper function to compute the gradient in x,y,(z) direction

    grid is the dataset with all grid info
    a is a dataarray to perform the grad on

    Currently only performs a first order forward gradient.
    It could be good to implement different order gradients later
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

def interpolate_from_W_to_C(grid,x):
    """
    Interp values from western boundary to cell center
    !!!These are just wrappers for interpolateGtoC
    """
    return interpolateGtoC(grid,x,dim='x')

def interpolate_from_S_to_C(grid,x):
    """
    Interp values from western boundary to cell center
    !!!These are just wrappers for interpolateGtoC
    """
    return interpolateGtoC(grid,x,dim='y')

def interpolateGtoC(grid,x,dim='x'):
    if dim == 'x':
        old_dim = 'i_g'
        swap_dim = 'i'
        x_name = 'X'
        method_x = 'wrap'
        method_y = 'roll'
    elif dim == 'y':
        old_dim = 'j_g'
        swap_dim = 'j'
        x_name = 'Y'
        method_x = 'pad'
        method_y = 'roll'

    dx = raw_diff(grid,grid[x_name+'G'],old_dim,method=method_x)
    dy = raw_diff(grid,x,old_dim,method=method_y)

    dx_new = grid[x_name+'G']-\
        reassign_grid(grid,grid[x_name+'C'],swap_dim,old_dim)

    y1 = x

    out = y1+dy*dx_new/dx
    return reassign_grid(grid,out,old_dim,swap_dim)



def raw_diff(grid,x,dim,method='pad',wrap_ref=360.0,shift_grid=False):

    if method == 'pad':
        #shift automatically pads the 'shifted around' values with nan
        x2 = x.shift(**{dim:-1}).copy()
        x1 = x.shift(**{dim:0}).copy()
    elif method in ['roll','wrap']:
        x2 = x.roll(**{dim:-1}).copy()
        x1 = x.roll(**{dim:0}).copy()
    else:
        raise RuntimeError("'method' not recognized")

    if method == 'wrap':
        warnings.warn("WARNING: option 'wrap' loads the dask array into memory")
        x2.load()
        x2[{dim:-1}] = x2[{dim:-1}]+wrap_ref
        # This is highly problematic when swap_dims is activated in xmitgcm/open_mdsdataset
        # Because one cannot replace index variables!

    diff_raw = x2.data-x1.data

#     new_coords = dict([])
#     for ii in x.coords.keys():
#         if ii==dim:
#             if shift_grid:

#                 new_coords[ii+'_']
#             else:
#                 new_coords[ii] = grid[ii]
#         else:
#             new_coords[ii] = grid[ii]

    new_coords = dict([])
    for ii in x.coords.keys():
            new_coords[ii] = grid[ii]



    diff = xr.DataArray(diff_raw,coords=x.coords,dims=x.dims)
    # This purposely does not switch the dims from grid to center at this point,
    # That needs to be set in the calling routine.
    # if the method is not wrap and it makes sense the border has to be set to nan...

    return diff

def reassign_grid(grid,x,old,new,debug=False):
    dims = list(x.dims)
    if debug:
        print 'swapping '+old+' into '+new
    dims[dims.index(old)] = new

    coords = dict([])
    for ii in dims:
        coords[ii] = grid[ii]
    return xr.DataArray(x.data,coords=coords,dims=dims)

def inferGfromC(grid,x,dim,method='wrap'):
    dx = raw_diff(grid,x,dim,method=method)
    x_out = x-(dx/2)
    return reassign_grid(grid,x_out,dim,dim+'_g')
# Test: Convert the actual XC into XG and check if they match
# test =  inferGfromC(grid_diag,ds_diag.YC,'j',method='roll')
# np.all(np.all(np.isclose(test.data,ds_diag.YG.data),axis=1)[0:-1]


def grid_aggregate(grid,bins):
    new_dims = dict([])

    # subset dims
#     for dd in list(grid.dims):
    for dd in ['i','i_g','j','j_g','time']:
            new_dims[dd] = grid[dd]

    for bb in bins:
        if bb[0] in new_dims.keys():
            new_dims[bb[0]] = new_dims[bb[0]][::bb[1]]

    for bb in bins:
        if bb[0]+'_g' in new_dims.keys():
            new_dims[bb[0]+'_g'] = new_dims[bb[0]+'_g'][::bb[1]]

    out = xr.Dataset(new_dims)

    new_coords = dict([])

    temp = dict([])
    for ff in ['XC','YC']:
        temp[ff] = xr.DataArray(da_ar.from_array(grid[ff].data,chunks=grid[ff].shape),
                                coords=grid[ff].coords,dims=grid[ff].dims)

    # this needs to be automated but for now...lets do it manually
    new_coords['XC'] = aggregate(temp['XC'],bins,np.mean)
    new_coords['YC'] = aggregate(temp['YC'],bins,np.mean)

    # print new_coords
    new_coords['XG'] = inferGfromC(out,new_coords['XC'],'i',method='wrap')
    new_coords['YG'] = inferGfromC(out,new_coords['YC'],'j',method='roll')

    # print new_coords
    temp = raw_diff(out,new_coords['XC'],'i',method='wrap')
    new_coords['dxC'] = reassign_grid(out,temp,'i','i_g')

    temp = raw_diff(out,new_coords['YC'],'j',method='roll')
    new_coords['dyC'] = reassign_grid(out,temp,'j','j_g')

    temp = raw_diff(out,new_coords['XG'],'i_g',method='wrap')
    new_coords['dxG'] = reassign_grid(out,temp,'i_g','i')

    temp = raw_diff(out,new_coords['YG'],'j_g',method='roll')
    new_coords['dyG'] = reassign_grid(out,temp,'j_g','j')

#     for nn in new_coords.keys():
    out = out.assign_coords(**new_coords)
    return out
