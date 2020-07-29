import pytest

xgcm = pytest.importorskip("xgcm")
from xarrayutils.build_grids import rebuild_grid
from numpy.testing import assert_allclose
from .datasets import datagrid_dimtest, datagrid_dimtest_ll


@pytest.mark.parametrize(
    "test_coord",
    ["i", "j", "i_g", "j_g", "XC", "XG", "YC", "YG", "dxC", "dxG", "dyC", "dyG"],
)
# TODO This should be able to read all coord variable from the dataset
# so its not hardcoded, but I cant get it to work
def test_rebuild_grid(datagrid_dimtest, test_coord):
    a = datagrid_dimtest
    coords = a.coords.keys()
    coords_stripped = [x for x in coords if x not in ["i", "j", "XC", "YC"]]
    stripped = a.drop(coords_stripped)
    b = rebuild_grid(stripped, x_wrap=360.0, y_wrap=180.0, ll_dist=False)
    assert b[test_coord].dims == a[test_coord].dims
    assert_allclose(b[test_coord].data, a[test_coord].data)


@pytest.mark.parametrize(
    "test_coord",
    ["i", "j", "i_g", "j_g", "XC", "XG", "YC", "YG", "dxC", "dxG", "dyC", "dyG"],
)
# TODO This should be able to read all coord variable from the dataset
# so its not hardcoded, but I cant get it to work
def test_rebuild_grid_ll(datagrid_dimtest_ll, test_coord):
    a = datagrid_dimtest_ll
    coords = a.coords.keys()
    coords_stripped = [x for x in coords if x not in ["i", "j", "XC", "YC"]]
    stripped = a.drop(coords_stripped)
    b = rebuild_grid(stripped, x_wrap=360.0, y_wrap=180.0, ll_dist=True)
    assert b[test_coord].dims == a[test_coord].dims
    assert_allclose(b[test_coord].data, a[test_coord].data)
