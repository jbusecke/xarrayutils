from datetime import datetime, timedelta
import pandas as pd
import os
import xarray as xr
import dask.array as dsa
import matplotlib.pyplot as plt
from . utils import concat_dim_da
from . weighted_operations import weighted_mean, weighted_sum


# def convert_boundary_flux(da,da_full,top=True):
#     dummy = xr.DataArray(dsa.zeros_like(da_full.data),
#                     coords = da_full.coords,
#                     dims = da_full.dims)
#     if top:
#         da.coords['st_ocean'] = da_full['st_ocean'][0]
#         dummy_cut = dummy.isel(st_ocean=slice(1,None))
#         out = xr.concat([da,dummy_cut],dim='st_ocean')
#     else:
#         da.coords['st_ocean'] = da_full['st_ocean'][-1]
#         dummy_cut = dummy.isel(st_ocean=slice(0,-1))
#         out = xr.concat([dummy_cut,da],dim='st_ocean')
#     return out


def cm26_flist(ddir, name, years=None, yearformat='%04i0101'):
    if years:
        name = name.replace('*', yearformat)
        out = [os.path.join(ddir, name % yy) for yy in years]
    else:
        out = os.path.join(ddir, name)
    return out


def cm26_convert_boundary_flux(da, da_full, top=True):
    dummy = xr.DataArray(dsa.zeros_like(da_full.data),
                         coords=da_full.coords,
                         dims=da_full.dims)
    if top:
        da.coords['st_ocean'] = da_full['st_ocean'][0]
        dummy_cut = dummy.isel(st_ocean=slice(1, None))
        out = xr.concat([da, dummy_cut], dim='st_ocean')
    else:
        da.coords['st_ocean'] = da_full['st_ocean'][-1]
        dummy_cut = dummy.isel(st_ocean=slice(0, -1))
        out = xr.concat([dummy_cut, da], dim='st_ocean')
    return out


def convert_units(ds, name, new_unit, factor):
    ds = ds.copy()
    ds[name] = ds[name]*factor
    ds[name].attrs['units'] = new_unit
    return ds


def shift_lon(ds, londim, shift=360, crit=0, smaller=True, sort=True):
    ds = ds.copy()
    if smaller:
        ds[londim].data[ds[londim] < crit] = \
            ds[londim].data[ds[londim] < crit] + shift
    else:
        ds[londim].data[ds[londim] > crit] = \
            ds[londim].data[ds[londim] > crit] + shift

    if sort:
        ds = ds.sortby(londim)
    return ds


def mask_tracer(ds, mask_ds, levels, name):
    out = []
    for ll in levels:
        out.append(ds.where(mask_ds <= ll))
    out = xr.concat(out, concat_dim_da(levels, name))
    return out

# Doesnt work with the timing
# def metrics_wrapper(ds, odir, oname, xdim='xt_ocean',
#                     ydim='yt_ocean', zdim='st_ocean',
#                     omz_var='o2',
#                     omz_thresholds=[30, 60, 100, 1000]):
#
#     print('========mask omz===========')
#     # mask different values of
#     ds_box = mask_tracer(ds, ds[omz_var], omz_thresholds, 'omz_thresholds')
#
#     print('========track weights===========')
#     # Add a 'dummy' array of ones like the oxygen, to track the total weight (can later be subtracted to get mean)
#     ds_box = ds_add_track_dummy(ds_box, omz_var)
#
#     print('========calculate metrics===========')
#     % time metrics = metrics_ds(ds_box, xdim, ydim, zdim, area_w='area_t', \
#                                 volume_w='volume', compute_average=False)
#
#     print('========save metrics===========')
#     %time metrics_save(metrics, odir, '%s_%s_sum.nc' %(oname,bb))


def metrics_ds(ds, xdim, ydim, zdim, area_w='area_t', volume_w='volume',
               compute_average=False, drop_control=False):
    """applies all needed metrics to a dataset and puts out a dict"""
    metrics = dict()
    if compute_average:
        metrics['x_section'] = weighted_mean(ds, ds[area_w],
                                             dim=[ydim], dimcheck=False)
        metrics['y_section'] = weighted_mean(ds, ds[area_w],
                                             dim=[xdim], dimcheck=False)
        metrics['profile'] = weighted_mean(ds, ds[area_w],
                                           dim=[xdim, ydim])
        metrics['timeseries'] = weighted_mean(ds, ds[volume_w],
                                              dim=[xdim, ydim, zdim])
        if drop_control:
            for ff in metrics.keys():
                metrics[ff] = metrics[ff].drop('ones')

    else:
        metrics['x_section'] = weighted_sum(ds, ds[area_w],
                                            dim=[ydim], dimcheck=False)
        metrics['y_section'] = weighted_sum(ds, ds[area_w],
                                            dim=[xdim], dimcheck=False)
        metrics['profile'] = weighted_sum(ds, ds[volume_w],
                                          dim=[xdim, ydim])
        metrics['timeseries'] = weighted_sum(ds, ds[volume_w],
                                             dim=[xdim, ydim, zdim])

        for ff in metrics.keys():
            if ff == 'timeseries':
                rename = 'integrated_volume'
            else:
                rename = 'integrated_area'
            metrics[ff] = metrics[ff].rename({'ones': rename})

    return metrics


def metrics_control_plots(metrics, xdim='xt_ocean', ydim='yt_ocean',
                          zdim='st_ocean', tdim='time'):

    n_data_var = len(metrics['timeseries'].data_vars)
    # n_time = len(metrics['timeseries'][tdim])

    plt.figure(figsize=[14, 5*n_data_var])
    # summed values
    for ii, ip in enumerate(metrics['timeseries'].data_vars):
        plt.subplot(n_data_var, 2, ii+1)
        data = (metrics['timeseries'])
        if 'omz_thresholds' in metrics['timeseries'].dims.keys():
            for oo in metrics['timeseries'].omz_thresholds:
                metrics['timeseries'][ip].sel(omz_thresholds=oo).plot()
        else:
            metrics['timeseries'].plot()
    # mean values (volume should be 1!)
    for ii, ip in enumerate(metrics['timeseries'].data_vars):
        plt.subplot(n_data_var, 2, ii+1+len(metrics['timeseries'].data_vars))
        data = (metrics['timeseries'] /
                metrics['timeseries']['integrated_volume'])
        if 'omz_thresholds' in metrics['timeseries'].dims.keys():
            for oo in metrics['timeseries'].omz_thresholds:
                data[ip].sel(omz_thresholds=oo).plot()
        else:
            data.plot()

    # ################ Sections ###############
    for ii, ip in enumerate(metrics['x_section'].data_vars):

        kwarg_dict = {'robust': True,
                      'yincrease': False}
        if 'omz_thresholds' in metrics['timeseries'].dims.keys():
            kwarg_dict['col'] = 'omz_thresholds'
        for sec in ['x', 'y']:
            plt.figure()
            metrics[sec+'_section'][{tdim: 0}][ip].plot(**kwarg_dict)

            plt.figure()
            l = (metrics[sec+'_section'] /
                 metrics[sec+'_section']['integrated_area'])
            l[{tdim: 0}][ip].plot(**kwarg_dict)

    # plt.imshow(ds_box[plot_var].isel(TIME=0,DEPTH_center=3))
    # plt.figure()
    # ds_box[plot_var].isel(TIME=0,DEPTH_center=3).plot()

    # y_section[plot_var].isel(TIME=0).plot()


def metrics_save(metrics, odir, fname, mf_save=False, **kwargs):
    for kk in metrics.keys():
        if mf_save:
            years, datasets = zip(*metrics[kk].groupby('time.year'))
            paths = [os.path.join(odir,
                '%04i_%s_%s.nc' %(y, fname, kk)) for y in years]
            xr.save_mfdataset(datasets, paths)
        else:
            metrics[kk].to_netcdf(os.path.join(odir, '%s_%s.nc' % (fname, kk)),
                                  **kwargs)


def metrics_load(metrics, odir, fname, **kwargs):
    for kk in metrics.keys():
        metrics[kk] = xr.open_dataset(os.path.join(odir,
                                                   '%s_%s.nc' % (fname, kk)),
                                      **kwargs)


def ds_add_track_dummy(ds, refvar):
    ones = ds[refvar].copy()
    ones = (ones*0)+1
    return ds.assign(ones=ones)


def metrics_wrapper():
    pass


def time_add_refyear(ds, timedim='time', refyear=2000):
    ds = ds.copy()
    # Fix the time axis (I added 1900 years, since otherwise the stupid panda
    # indexing does not work)
    ds[timedim].data = pd.to_datetime(datetime(refyear+1, 1, 1, 0, 0, 0) +
                                      ds[timedim].data*timedelta(1))
    ds.attrs['refyear_shift'] = refyear
    # Weirdly I have to add a year here or the timeline is messed up.

    return ds


def add_grid_geometry(ds, rho_dzt, area):
    ds_new = ds.copy()
    rho_dzt = rho_dzt.copy()
    area = area.copy()
    ds_new = ds_new.assign_coords(area_t=area)
    # Infer vertical spacing (divided by constant rho=1035, since the model
    # uses the boussinesque appr.
    # [Griffies, Tutorial on budgets])
    ds_new = ds_new.assign_coords(dzt=(rho_dzt/1035.0))
    ds_new = ds_new.assign_coords(volume=ds_new['dzt']*ds_new['area_t'])
    ds_new = ds_new.assign_coords(rho_dzt=rho_dzt)
    return ds_new


def cm26_readin_annual_means(name, run,
                             rootdir='/work/Julius.Busecke/CM2.6_staged/'):

    global_file_kwargs = dict(
        decode_times=False,
    )

    # choose the run directory
    if run == 'control':
        rundir = os.path.join(rootdir, 'CM2.6_A_Control-1860_V03')
        years = range(100, 201)
    elif run == 'forced':
        rundir = os.path.join(rootdir, 'CM2.6_A_V03_1PctTo2X')
        years = range(121, 201)

    if name == 'minibling_fields':
        path = os.path.join(rundir, 'annual_averages')
        name = '*.field.nc'
        yearformat = '%04i'
        file_kwargs = dict(drop_variables=['area_t',  'geolat_t',
                                           'geolon_t', 'average_T1',
                                           'average_T2', 'average_DT',
                                           'time_bounds', 'nv', 'chl'],
                           chunks={'time': 1, 'st_ocean': 1})
    elif name == 'physics':
        path = os.path.join(rundir, 'annual_averages')
        name = 'ocean.*.ann.nc'
        yearformat = '%04i'
        file_kwargs = dict(drop_variables=['area_t',  'geolat_t',
                                           'geolon_t', 'average_T1',
                                           'average_T2', 'average_DT',
                                           'time_bounds', 'nv',
                                           'xu_ocean', 'yu_ocean',
                                           'sw_ocean', 'xu_ocean',
                                           'yu_ocean', 'sw_ocean',
                                           'st_edges_ocean', 'sw_edges_ocean',
                                           'geolon_c', 'geolat_c', 'u',
                                           'v', 'eta_u', 'ty_trans',
                                           'salt_int_rhodz', 'sea_level',
                                           'sea_levelsq', 'sfc_hflux_coupler',
                                           'tau_x', 'tau_y', 'temp_int_rhodz',
                                           'wt', 'frazil_2d',
                                           'net_sfc_heating',
                                           'pme_river', 'river'],
                           chunks={'time': 1, 'st_ocean': 1})
    elif name == 'osat':
        path = os.path.join(rundir, 'annual_averages/o2_sat')
        if run == 'control':
            name = '*.control_o2_sat.nc'
        else:
            name = '*.forced_o2_sat.nc'
        yearformat = '%04i'
        file_kwargs = dict(chunks={'st_ocean': 1},
                           concat_dim='time')
    else:
        raise RuntimeError('name not recognized')
    # for debugging
    print

    flist = cm26_flist(path, name, years=years, yearformat=yearformat)
    file_kwargs.update(global_file_kwargs)
    return xr.open_mfdataset(flist, **(file_kwargs))
