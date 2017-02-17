"""
Code specific to xarrays created with xmitgcm

"""

import numpy as np
import xarray as xr

def gradient(grid,a,location='center',recenter=False):
    'a is the complete dataset and b is the dataarray to perform the grad on'
    # grid = b
    if location=='grid':
        shift_idx = [-1,0]
        ref_x = grid.XC
        ref_y = grid.YC
        dx = grid.dxG
        dy = grid.dyG
    elif location=='center':
        shift_idx = [0,1]
        ref_x = grid.XG
        ref_y = grid.YG
        dx = grid.dxC
        dy = grid.dyC

    if len(b.shape)==2:
        raise RuntimeError('this should never happen...!!!')
        ref_x = ref_x.mean(dim='time')
        ref_y = ref_y.mean(dim='time')

    diff_x_raw = a.roll(i=shift_idx[0]).data-b.roll(i=shift_idx[1]).data
    diff_x = xr.DataArray(diff_x_raw,dims=ref_x.dims,coords=ref_x.coords)


    diff_y_raw = b.roll(j=shift_idx[0]).data-b.roll(j=shift_idx[1]).data
    diff_y = xr.DataArray(diff_y_raw,dims=ref_y.dims,coords=ref_y.coords)

    grad_x = diff_x/dx
    grad_y = diff_y/dy

    if recenter:
        grad_x = interpolate_from_W_to_C(a,grad_x)
        grad_y = interpolate_from_S_to_C(a,grad_y)

    return grad_x,grad_y

def interpolate_from_W_to_C(a,b):
    """
    Interp values from western boundary to cell center
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
