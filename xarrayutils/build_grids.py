from __future__ import print_function
import xgcm
import xarray as xr
import dask.array as da
import numpy as np
import warnings

from . utils import aggregate

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

def inferGfromC(grid,x,dim,method='wrap'):
    dx = raw_diff(grid,x,dim,method=method)
    x_out = x-(dx/2)
    out = reassign_grid(grid,x_out,dim,dim+'_g')

    # this could probably be accomplished with the interpolation once done
    return out
# Test: Convert the actual XC into XG and check if they match
# test =  inferGfromC(grid_diag,ds_diag.YC,'j',method='roll')
# np.all(np.all(np.isclose(test.data,ds_diag.YG.data),axis=1)[0:-1]

def get_dims_from_comodo_axes(ds,axis):
    dims = ds.dims.keys()
    pick_dims = []
    for dd in dims:
        if ds[dd].attrs.keys():
            if 'standard_name' in ds[dd].attrs.keys():
                if 'axis' in ds[dd].attrs.keys():
                    if axis.lower()+'_grid_index' in ds[dd].attrs['standard_name']:
                        if axis in ds[dd].attrs['axis']:
                            pick_dims.append(dd)
    return pick_dims


def grid_aggregate(grid,axis_bins):
    """aggregate a grid dataset compatible with xgcm

    PARAMETERS
    ----------
    grid : xarray.Dataset (Attributes compatible with xgcm required)
    axis_bins : list of tuples with axis specifier (e.g. 'X',see xgcm) and
        Aggregation interval

    RETURNS
    -------
    grid_c : xarray.Dataset
        aggregated dataset

    TODO
    ----
        generalise the treatment of coordinates. So far this is very specific to
        mitgcm output
    """
    bins = []
    for tt in axis_bins:
        bins = bins+[(a,tt[1]) for a in get_dims_from_comodo_axes(grid,tt[0])]

    new_dims = dict([])
    for dd in grid.dims.keys():
        new_dims[dd] = grid[dd]

    for bb in bins:
        if bb[0] in new_dims.keys():
            new_dims[bb[0]] = new_dims[bb[0]][::bb[1]]

    out = xr.Dataset(new_dims)

    temp = dict([])
    for ff in grid.coords.keys():
        temp[ff] = xr.DataArray(da.from_array(grid[ff].data,chunks=grid[ff].shape),
                                coords=grid[ff].coords,dims=grid[ff].dims)

    c = dict([])
    # this needs to be automated but for now...lets do it manually
    c['XC'] = aggregate(temp['XC'],[a for a in bins if a in temp['XC'].dims],np.mean)
    c['YC'] = aggregate(temp['YC'],[a for a in bins if a in temp['YC'].dims],np.mean)
    # I am not sure how to aggregate the other things...

    out = out.assign_coords(**c)
    out = rebuild_grid(out)
    return out

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

def rebuild_grid(grid):
        # These are not general enough...need to rewrite them for any xgcm file
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
        grid['i'].attrs = {'axis': 'X',
                          'standard_name': 'x_grid_index',
                          'long_name': 'x-dimension of the grid'}
        grid['i_g'].attrs = {'axis': 'X',
                          'standard_name': 'x_grid_index_at_u_location',
                          'long_name': 'x-dimension of the grid',
                          'c_grid_axis_shift': -0.5}

        grid['j'].attrs = {'axis': 'Y',
                          'standard_name': 'y_grid_index',
                          'long_name': 'y-dimension of the grid'}
        grid['j_g'].attrs = {'axis': 'Y',
                          'standard_name': 'y_grid_index_at_u_location',
                          'long_name': 'y-dimension of the grid',
                          'c_grid_axis_shift': -0.5}
        # patch the attributes to the
        return grid


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
