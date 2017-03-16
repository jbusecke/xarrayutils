import pytest
import xarray as xr
import numpy as np
import os
from xarrayutils.build_grids import rebuild_grid
from numpy.testing import assert_allclose

from . datasets import datagrid_dimtest

@pytest.mark.parametrize("test_coord",
    ['i','j','i_g','j_g','XC','XG','YC','YG','dxC','dxG','dyC','dyG'])
    # TODO This should be able to read all coord variable from the dataset
    # so its not hardcoded, but I cant get it to work
def test_rebuild_grid(datagrid_dimtest,test_coord):
    coords          = datagrid_dimtest.coords.keys()
    coords_stripped = [ x for x in coords if x not in ['i','j','XC','YC'] ]
    stripped        = datagrid_dimtest.drop(coords_stripped)
    reconstructed   = rebuild_grid(stripped,x_wrap=8.0,y_wrap=6.0)
    assert reconstructed[test_coord].dims == datagrid_dimtest[test_coord].dims
    assert_allclose(reconstructed[test_coord].data,
        datagrid_dimtest[test_coord].data)
