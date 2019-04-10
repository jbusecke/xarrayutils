from xgcm import Grid
import xarray as xr
import numpy as np
import pytest
from xarrayutils.xgcm_utils import _infer_gridtype


def test_infer_gridtype():
    xt = np.arange(4)
    xu = xt + 0.5
    yt = np.arange(4)
    yu = yt + 0.5

    # Need to add a tracer here to get the tracer dimsuffix
    tr = xr.DataArray(np.random.rand(4, 4), coords=[("xt", xt), ("yt", yt)])

    u_b = xr.DataArray(np.random.rand(4, 4), coords=[("xu", xu), ("yu", yu)])
    v_b = xr.DataArray(np.random.rand(4, 4), coords=[("xu", xu), ("yu", yu)])

    u_c = xr.DataArray(np.random.rand(4, 4), coords=[("xu", xu), ("yt", yt)])
    v_c = xr.DataArray(np.random.rand(4, 4), coords=[("xt", xt), ("yu", yu)])

    coords = {
        "X": {"center": "xt", "right": "xu"},
        "Y": {"center": "yt", "right": "yu"},
    }

    ds_b = xr.Dataset({"u": u_b, "v": v_b, "tracer": tr})
    grid_b = Grid(ds_b, coords=coords)

    ds_c = xr.Dataset({"u": u_c, "v": v_c, "tracer": tr})
    grid_c = Grid(ds_c, coords=coords)

    # This should fail
    ds_fail = xr.Dataset({"u": u_b, "v": v_c, "tracer": tr})
    grid_fail = Grid(ds_fail, coords=coords)
    print(_infer_gridtype(grid_b, ds_b.u, ds_b.v))
    assert _infer_gridtype(grid_b, ds_b.u, ds_b.v) == "B"
    assert _infer_gridtype(grid_c, ds_c.u, ds_c.v) == "C"
    with pytest.raises(RuntimeError, match=r"Gridtype not recognized *"):
        _infer_gridtype(grid_fail, ds_fail.u, ds_fail.v)

    # This is not supported yet
    coords2 = {
        "X": {"center": "xt", "outer": "xu"},
        "Y": {"center": "yt", "outer": "yu"},
    }
    ds_fail2 = xr.Dataset({"u": u_b, "v": v_c, "tracer": tr})
    grid_fail2 = Grid(ds_fail2, coords=coords2)

    with pytest.raises(RuntimeError):  # , match=r'`inner` or `outer` *'
        _infer_gridtype(grid_fail2, ds_fail2.u, ds_fail2.v)
