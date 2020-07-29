import pytest

xgcm = pytest.importorskip("xgcm")
from xarrayutils.xmitgcm_utils import get_hfac, get_dx
from .datasets import datagrid_w_attrs


@pytest.mark.parametrize(
    "test_coord", ["XC", "XG", "YC", "YG", "dxC", "dxG", "dyC", "dyG"]
)
# TODO I am not sure if this checks all possible combos
# TODO This should be able to read all coord variable from the dataset
# so its not hardcoded, but I cant get it to work
def test_get_dx_dims(datagrid_w_attrs, test_coord):
    data = datagrid_w_attrs
    grid = xgcm.Grid(data)
    dx = get_dx(grid, data[test_coord], "X")
    dy = get_dx(grid, data[test_coord], "Y")
    assert dx.dims == data[test_coord].dims
    assert dy.dims == data[test_coord].dims


# TODO write the same test for get_hfac
