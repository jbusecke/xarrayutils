import xgcm
def inferGfromC(grid,x,dim,method='wrap'):
    dx = raw_diff(grid,x,dim,method=method)
    x_out = x-(dx/2)
    out = reassign_grid(grid,x_out,dim,dim+'_g')

    # this could probably be accomplished with the interpolation once done
    return out
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
