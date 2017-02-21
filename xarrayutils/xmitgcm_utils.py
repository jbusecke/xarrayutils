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
    c = []
    for kk in grid.coords.keys():
        check = list(grid[kk].dims)
        if all([a in dims for a in check]):
            c.append(kk)

    c_dict = dict([])
    for ii in c:
        c_dict[ii] = grid[ii]
    return c_dict

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

    c = matching_coords(grid,new_dims)

    dx_data = grid[dx].data
    diff_x_raw = ar.roll(**{dim:shift_idx[0]}).data-ar.roll(**{dim:shift_idx[1]}).data
    # This needs to be implemented with custom diff (using ;wrap option)
    # Also this needs to land on the new
    grad_x = xr.DataArray(diff_x_raw/dx_data,dims=new_dims,coords=c)
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
        method_y = 'roll'
        dx = grid['dxG']
    elif dim == 'y':
        old_dim = 'j_g'
        swap_dim = 'j'
        x_name = 'Y'
        method_y = 'roll'
        dx = grid['dyG']

    dy = raw_diff(grid,x,old_dim,method=method_y)

    # for now lets jsut assume the new dx is half of the old...
    dx_new = dx/2
    # dx_ll_new = grid[x_name+'C'].data-grid[x_name+'G'].data
    # print x
    # dx_ll_new = xr.DataArray(dx_ll_new,coords = x.coords,dims = x.dims)
    #
    # if dim == 'x':
    #     dx_new,_ = dll_dist(grid,dx_ll_new,dx_ll_new,grid['XG'],grid['YG'],\
    #                 lon_dim='i',lat_dim='j')
    # elif dim == 'y':
    #     _,dx_new = dll_dist(grid,dx_ll_new,dx_ll_new,grid['XG'],grid['YG'],\
    #                 lon_dim='i',lat_dim='j')

    dx     = dx.data
    dx_new = dx_new.data
    dy     = dy.data
    y1     = x.data

    out = y1+dy*dx_new/dx
    out = xr.DataArray(out,coords=x.coords,dims=x.dims)

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

    # Do i need the grid input here? Not really since
    # I am keeping the dims and coords the same

    # c = dict([])
    # for ii in x.coords.keys():
    #         c[ii] = grid[ii]

    # So lets try this...
    c = dict([])
    for ii in x.coords.keys():
            c[ii] = x.coords[ii]



    diff = xr.DataArray(diff_raw,coords=x.coords,dims=x.dims)
    # This purposely does not switch the dims from grid to center at this point,
    # That needs to be set in the calling routine.
    # if the method is not wrap and it makes sense the border has to be set to nan...

    return diff

def reassign_grid(grid,x,old,new,debug=False):
    dims = list(x.dims)
    dims[dims.index(old)] = new
    if new in list(grid.dims):
        rename_switch = False
    else:
        rename_switch = True

    coords = dict([])
    for ii in dims:
        if ii == new and rename_switch:
            coords[ii] = grid[old].rename({old:new})
        else:
            coords[ii] = grid[ii]

    return xr.DataArray(x.data,coords=coords,dims=dims)

def inferGfromC(grid,x,dim,method='wrap'):
    dx = raw_diff(grid,x,dim,method=method)
    x_out = x-(dx/2)
    out = reassign_grid(grid,x_out,dim,dim+'_g')

    # this could probably be accomplished with the interpolation once done
    return out
# Test: Convert the actual XC into XG and check if they match
# test =  inferGfromC(grid_diag,ds_diag.YC,'j',method='roll')
# np.all(np.all(np.isclose(test.data,ds_diag.YG.data),axis=1)[0:-1]

def rebuild_grid(grid):
        c = dict([])
        c['XC'] = grid['XC']
        c['YC'] = grid['YC']

        c['XG'] = inferGfromC(grid,c['XC'],'i',method='wrap')
        c['YG'] = inferGfromC(grid,c['YC'],'j',method='roll')

        temp = raw_diff(grid,c['XC'],'i',method='wrap')
        c['dxC'],_ = dll_dist(grid,temp,temp,c['XG'],c['YC'])

        temp = raw_diff(grid,c['YC'],'j',method='roll')
        _,c['dyC'] = dll_dist(grid,temp,temp,c['XC'],c['YG'])

        temp = raw_diff(grid,c['XG'],'i_g',method='wrap')
        c['dxG'],_ = dll_dist(grid,temp,temp,c['XC'],c['YG'])

        temp = raw_diff(grid,c['YG'],'j_g',method='roll')
        _,c['dyG'] = dll_dist(grid,temp,temp,c['XG'],c['YC'])

        grid = grid.assign_coords(**c)
        return grid


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

    temp = dict([])
    for ff in ['XC','YC']:
        temp[ff] = xr.DataArray(da_ar.from_array(grid[ff].data,chunks=grid[ff].shape),
                                coords=grid[ff].coords,dims=grid[ff].dims)

    c = dict([])
    # this needs to be automated but for now...lets do it manually
    c['XC'] = aggregate(temp['XC'],bins,np.mean)
    c['YC'] = aggregate(temp['YC'],bins,np.mean)

    out = out.assign_coords(**c)

    out = rebuild_grid(out)

    return out

def dll_dist(grid,dlon,dlat,lon,lat,lon_dim='i',lat_dim='j'):
    # First attempt with super cheap approach...
    #     111km for each deg lat and then scale that by cos(lat) for lon
    dll_factor = 111000.0

    # x_coords = dict([])
    # for xx in list(lon.dims):
    #     x_coords[xx] = grid[xx]
    # y_coords = dict([])
    # for yy in list(lon.dims):
    #     y_coords[yy] = grid[yy]
    dx = dlon.data*np.cos(np.deg2rad(lat.data))*dll_factor
    dx = xr.DataArray(dx,coords=lon.coords,dims=lon.dims)
    dy = dlat.data*dll_factor
    dy = xr.DataArray(dy,coords=lat.coords,dims=lat.dims)
    return dx,dy
