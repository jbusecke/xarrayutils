import xarray as xr
import numpy as np
import pytest
from xarrayutils.vertical_coordinates import conservative_remap

def random_broadcast(da):
    # add random noise with additionial dimensions to input array
    raw_noise = np.random.rand(2,6,12)
    noise = xr.DataArray(raw_noise, dims=['test_a', 'test_b', 'test_c'])
    return ((noise-0.5) * 4) + da

@pytest.mark.parametrize('mask', [True, False])
@pytest.mark.parametrize('multi_dim', [True, False])
@pytest.mark.parametrize('dask', [True, False])
def test_conservative_remap_1D(mask, multi_dim, dask):
    data = xr.DataArray(np.array([30,12.3,5]), dims=['z'])
    z_bounds_source = xr.DataArray(np.array([4.5,9, 23, 45.6]), dims=['z_bounds'])
    z_bounds_target = xr.DataArray(np.array([0,2,4,10,11,13.4, 23, 55.6, 80, 100]), dims=['z_bounds'])

    if multi_dim:
        # Add more dimension and shift values for each depth profile by a random amount
        data = random_broadcast(data)
        z_bounds_source = random_broadcast(z_bounds_source)
        z_bounds_target = random_broadcast(z_bounds_target)

    if dask:
        if multi_dim:
            chunks = {'test_c':1}
        else:
            chunks = {}
        data = data.chunk(chunks)
        z_bounds_source = z_bounds_source.chunk(chunks)
        z_bounds_target = z_bounds_target.chunk(chunks)

    # Calculate cell thickness and rename
    dz_source = z_bounds_source.diff('z_bounds').rename({'z_bounds':'z'})
    dz_target = z_bounds_target.diff('z_bounds').rename({'z_bounds':'remapped'})

    data_new = conservative_remap(data, z_bounds_source, z_bounds_target, mask=mask)

    raw = (data*dz_source).sum('z')
    remapped = (data_new*dz_target).sum('remapped')
    print(raw)
    print(remapped)
    xr.testing.assert_allclose(raw, remapped)
