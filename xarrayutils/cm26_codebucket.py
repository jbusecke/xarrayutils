from datetime import datetime, timedelta
import pandas as pd
import os
import xarray as xr
import numpy as np
from dask.array import zeros_like
from . utils import concat_dim_da
from . weighted_operations import weighted_mean, weighted_sum


def cm26_flist(ddir, name, years=None, yearformat='%04i0101'):
    if years:
        name = name.replace('*', yearformat)
        out = [os.path.join(ddir, name % yy) for yy in years]
    else:
        out = os.path.join(ddir, name)
    return out


def cm26_convert_boundary_flux(da, da_full, top=True):
    dummy = xr.DataArray(zeros_like(da_full.data),
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


def time_add_refyear(ds, timedim='time', refyear=2000):
    ds = ds.copy()
    # Fix the time axis (I added 1900 years, since otherwise the stupid panda
    # indexing does not work)
    ds[timedim].data = pd.to_datetime(datetime(refyear+1, 1, 1, 0, 0, 0) +
                                      ds[timedim].data*timedelta(1))
    ds.attrs['refyear_shift'] = refyear
    # Weirdly I have to add a year here or the timeline is messed up.

    return ds


def cm26_readin_annual_means(name, run,
                             rootdir='/work/Julius.Busecke/CM2.6_staged/',
                             print_flist=False,
                             autoclose=False,
                             years=None):

    global_file_kwargs = dict(
        decode_times=False,
        autoclose=autoclose
    )

    # choose the run directory
    if run == 'control':
        rundir = os.path.join(rootdir, 'CM2.6_A_Control-1860_V03')
    elif run == 'forced':
        rundir = os.path.join(rootdir, 'CM2.6_A_V03_1PctTo2X')

    if not years:
        years = range(121, 201)

    if name == 'minibling_fields':
        path = os.path.join(rundir, 'annual_averages/minibling_fields')
        name = '*.field.nc'
        yearformat = '%04i'
        file_kwargs = dict(drop_variables=['area_t',  'geolat_t',
                                           'geolon_t', 'average_T1',
                                           'average_T2', 'average_DT',
                                           'time_bounds', 'nv', 'chl'],
                           chunks={'time': 1, 'st_ocean': 1})
    elif name == 'physics':
        path = os.path.join(rundir, 'annual_averages/ocean')
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
    elif name == 'minibling_src':
        path = os.path.join(rundir, 'annual_averages/budgets')
        name = '*.src.nc'
        yearformat = '%04i'
        file_kwargs = dict(drop_variables=['area_t',  'geolat_t',
                                           'geolon_t', 'average_T1',
                                           'average_T2', 'average_DT',
                                           'time_bounds', 'nv', 'chl',
                                           'o2_btf', 'po4_btf', 'dic_btf',
                                           'dic_stf', 'o2_stf'],
                           chunks={'time': 1, 'st_ocean': 1})

    else:
        raise RuntimeError('name not recognized')
    # for debugging
    print

    flist = cm26_flist(path, name, years=years, yearformat=yearformat)
    if print_flist:
        print(flist)
    file_kwargs.update(global_file_kwargs)

    return xr.open_mfdataset(flist, **(file_kwargs))


def cm26_reconstruct_annual_grid(ds, grid_path=None, load=None):

    if grid_path is None:
        grid_path = '/work/Julius.Busecke/CM2.6_staged/static/CM2.6_grid_spec.nc'
    ds = ds.copy()
    chunks_raw = {'gridlon_t': 3600,
                  'gridlat_t': 2700,
                  'gridlon_c': 3600,
                  'gridlat_c': 2700,
                  'st_ocean': 1,
                  'sw_ocean': 1}
    ds_grid = xr.open_dataset(grid_path,
                              chunks=chunks_raw).rename({
                                            'gridlon_t': 'xt_ocean',
                                            'gridlat_t': 'yt_ocean'
                                            })

    # Problem. The gridfile has ever so slightly different values for the
    # dimensions. Xarray excludes these values
    # when they are multiplied. For now I will wrap all of the grid variables
    # in new dims and coordinates. But there should be a more elegant
    # method for this.

    # If I do this 'trick' with the ones, I make sure that dzt has the same
    # dimensions as the data_vars
    template = ds['o2'][{'time': 1, 'st_ocean':1}].drop(['time', 'st_ocean'])
    area = xr.DataArray(ds_grid['area_t'].data,
                        dims=template.dims,
                        coords=template.coords)

    # activates loading of presaved dzt value
    load_kwargs = dict(decode_times=False, concat_dim='time',
                       chunks={'st_ocean': 1})
    if load == 'control':
        odir = '/work/Julius.Busecke/CM2.6_staged/CM2.6_A_Control-1860_V03/annual_averages/grid_fields'
        ds_dzt = xr.open_mfdataset(os.path.join(odir, '*dzt_control.nc'),
                                   **load_kwargs)
        dz = ds_dzt['dzt']

    elif load == 'forced':
        odir = '/work/Julius.Busecke/CM2.6_staged/CM2.6_A_V03_1PctTo2X/annual_averages/grid_fields'
        ds_dzt = xr.open_mfdataset(os.path.join(odir, '*dzt_forced.nc'),
                                   **load_kwargs)
        dz = ds_dzt['dzt']

    elif load is None:

        # attempted fix to deal with the mismatch between eta, wet and tracer
        # fields (mask the full dimension one array in space with wet)
        oceanmask = ds_grid['wet']
        ones = (ds['temp']*0+1).where(oceanmask)
        ht = xr.DataArray(ds_grid['ht'].data,
                          dims=template.dims,
                          coords=template.coords)
        eta = ds['eta_t']
        dz_star = xr.DataArray(ds['st_edges_ocean'].diff('st_edges_ocean').data,
                               dims=['st_ocean'],
                               coords={'st_ocean': ds['st_ocean']})
        dz = ones*dz_star*(1+(eta/ht.data))
        dz = dz.chunk({'st_ocean': 1})

    ds = ds.assign_coords(dzt=dz)
    ds = ds.assign_coords(area_t=area)
    ds = ds.assign_coords(volume_t=ds['dzt']*ds['area_t'])
    return ds


def regiondict():
    return {'Pacific': 3}


def region2masknum(regionstr):
    reg_dict = regiondict()
    regions = list(reg_dict.keys())
    if regionstr not in regions:
        raise RuntimeError('region not recognized must be one of [' +
                           ' '.join(regions)+']')
    return reg_dict[regionstr]


def masknum2region(masknum):
    reg_dict = regiondict()
    regionnums = [value for key, value in reg_dict.items()]
    if masknum not in regionnums:
        raise RuntimeError('number not recognized. Not in [' +
                           ' '.join([str(a) for a in regionnums])+']')
    return [k for k, v in reg_dict.items() if v == masknum][0]


def cm26_loadall_run(run,
                     rootdir='/work/Julius.Busecke/CM2.6_staged/',
                     normalize_budgets=True,
                     reconstruct_grids=True,
                     grid_load=True,
                     drop_vars=None,
                     integrate_vars=None,
                     compute_aou=True,
                     diff_vars=None,
                     autoclose=True,
                     region=None):
    """Master read in function for CM2.6. Merges all variables into one
    dataset. If specified, 'normalize_budgets divides by dzt.
    'budget_drop' defaults to all non o2 variables from src file
    to save time."""

    if region is not None:
        if isinstance(region, str):
            print('conversion')
            region = region2masknum(region)
            regionstr = masknum2region(region)

    if 'detrended' in run:
        read_kwargs = dict(decode_times=False, concat_dim='time',
                           chunks={'st_ocean': 1},
                           autoclose=autoclose,
                           drop_variables=['area_t', 'dzt',  'volume_t'])
        if run == 'control_detrended':
            rundir = os.path.join(rootdir, 'CM2.6_A_Control-1860_V03/annual_averages/detrended')
        elif run == 'forced_detrended':
            rundir = os.path.join(rootdir, 'CM2.6_A_V03_1PctTo2X/annual_averages/detrended')

        print('test region')
        print(region)
        print('test regionstr')
        print(regionstr)
        print(os.path.join(rundir, '*_%s_%s.nc' % (run, regionstr)))
        if region is None:
            fid = os.path.join(rundir, '*_%s.nc' % (run))
        else:
            fid = os.path.join(rundir, '*_%s_%s.nc' % (run, regionstr))
        ds = xr.open_mfdataset(fid, **read_kwargs)

        # Deactivate options that only apply to the non detrended data
        normalize_budgets=False

    else:
        ds_minibling_field = cm26_readin_annual_means('minibling_fields',
                                                      run,
                                                      rootdir=rootdir,
                                                      autoclose=autoclose)
        ds_physics = cm26_readin_annual_means('physics',
                                              run,
                                              rootdir=rootdir,
                                              autoclose=autoclose)
        ds_osat = cm26_readin_annual_means('osat',
                                           run,
                                           rootdir=rootdir,
                                           autoclose=autoclose)
        ds_minibling_src = cm26_readin_annual_means('minibling_src',
                                                    run,
                                                    rootdir=rootdir,
                                                    autoclose=autoclose)

        # Brute force the minibling time into all files
        # ######## THEY DONT HAVE THE SAME TIMESTAMP MOTHERFUCK....
        ds_physics.time.data = ds_minibling_field.time.data
        ds_osat.time.data = ds_minibling_field.time.data

        # TODO: Build test if the time is equal .
        # for now just watch the timesteps
        ds = xr.merge([ds_minibling_field, ds_physics,
                      ds_osat, ds_minibling_src])

    if drop_vars:
        ds = ds.drop(drop_vars)

    # Calculate timestep (TODO: Make this more accurate by using the time data)
    dt = dt = 364*24*60*60
    if integrate_vars:
        for vv in integrate_vars:
            ds[vv+'_integrated'] = (ds[vv]*dt).cumsum('time')

    if diff_vars:
        for vv in diff_vars:
            ds[vv+'_diff'] = ds[vv].diff('time')/dt
            # ds[vv+'_diff'].data = ds[vv+'_diff'].data/dt

    if compute_aou:
        ds['aou'] = ds['o2_sat']-ds['o2']

    if reconstruct_grids:
        if grid_load:
            if 'control' in run:
                ds = cm26_reconstruct_annual_grid(ds, load='control')
            elif 'forced' in run:
                ds = cm26_reconstruct_annual_grid(ds, load='forced')
            else:
                raise RuntimeError('Could not load time variable grid files.\
            Check runname')
        else:
            ds = cm26_reconstruct_annual_grid(ds)

    # TODO: Possibly I should give a list as possible input
    if normalize_budgets:
        convert_vars = list(ds_minibling_src.data_vars.keys())
        for vv in convert_vars:
            # print('%s is beiung divided by rho_dzt' % vv)
            # ds[vv] = ds[vv]/1035.0/ds['dzt']
            # Fast version without checking
            ds[vv].data = ds[vv].data/1035.0/ds['dzt'].data

    if region:
        ds = cm26_cut_region(ds, region)
        ds = remove_nan_domain(ds, dim=['xt_ocean', 'yt_ocean'])

    return ds


def _tracer_coords(obj, bin_var='o2', w_var='volume_t', bins=10):
    # Label extracted bins with upper limit
    if isinstance(bins, int):
        labels = None
    else:
        labels = bins[1:]
    # save unweighted values as bin reference
    ref = obj[bin_var].copy()
    # Weight values
    for vv in obj.data_vars.keys():
        obj[vv] = obj[vv]*obj[w_var]
    # The 'w_var' is now useless (it should be the square of the weight)
    # Bin and sum
    binned = obj.groupby_bins(ref, bins, labels=labels,
                              include_lowest=True).sum()
    return binned


def tracer_coords(obj, bin_var='o2', weight='volume_t',
                  timedim='time', bins=10, rename_ones=True):
    obj = obj.copy()
    ones = obj[bin_var]*0+1
    obj = obj.assign(ones=ones)
    # This is not completely elegant...I dont want these inputs to be keyword
    out = obj.groupby(timedim).apply(_tracer_coords, bin_var=bin_var,
                                     w_var=weight, bins=bins)
    if rename_ones:
        # Now now remove the weight and rename the dummy array
        out = out.rename({'ones': weight+'_integrated'})
    return out


def save_years_wrapper(ds, odir, name, start_year, timesteps_per_yr=1,
                       timedim='time', **kwargs):
    if not os.path.isdir(odir):
        os.mkdir(odir)

    years = list(range(start_year, start_year+len(ds[timedim])))
    datasets = [ds[{timedim: a}] for a in range(len(ds[timedim]))]
    paths = [os.path.join(odir, '%04i.'+name) % y for y in years]
    xr.save_mfdataset(datasets, paths, **kwargs)


def cm26_cut_region(obj, region, cut_domain=True,
                    regionfile=None, rename_dict=None):
    """Masks dataset/dataarray according to cm2.6 regionmask and cuts to region
    Region is defined by number as follows
    0 = Land
    1 = Southern Ocean
    2 = Atlantic
    3 = Pacific
    4 = Arctic
    5 = Indian
    6 = Med
    7 = Black Sea
    8 = Labrador Sea
    9 = Baltic Sea
    10 = Red Sea
    """
    obj = obj.copy()

    if regionfile is None:
        regionfile = '/work/Julius.Busecke/CM2.6_staged/static/regionmask_cm26_020813.nc'

    # TODO: I should write an overarching function that renames all the
    # 'common' names for cm26 into a singel convention

    if rename_dict is None:
        rename_dict = dict(XT_OCEAN='xt_ocean', YT_OCEAN='yt_ocean',
                           XU_OCEAN='xu_ocean', YU_OCEAN='yu_ocean')

    regionmask = xr.open_dataset(regionfile).rename(rename_dict)
    # TODO: Extend this to u mask for matching variables
    # At the same time, I should find a way to adjust the tolerance
    # for coordinate comparison, so I dont have to swap them
    regionmask.coords['xt_ocean'].data = obj.coords['xt_ocean'].data
    regionmask.coords['yt_ocean'].data = obj.coords['yt_ocean'].data
    ##

    # Mask data (still full domain)
    obj = obj.where(regionmask['TMASK'].data == region)
    return obj


def remove_nan_domain(obj, dim, all_dim='st_ocean', margin=0):
        if isinstance(obj, xr.DataArray):
            test_slice = obj.isel(time=0)
        elif isinstance(obj, xr.Dataset):
            test_slice = obj[list(obj.data_vars)[0]].isel(time=0).drop('time')
        else:
            raise RuntimeError('obj input has to be xarray.Dataset \
                               or DataArray')

        nanmask = xr.ufuncs.isnan(test_slice).all(all_dim)

        if isinstance(dim, str):
            dim = [dim]

        for dd in dim:
            all_dims = [a for a in list(nanmask.dims) if a != dd]
            data_idx = np.where(~nanmask.all(all_dims))[0]
            test = {dd: slice(data_idx[0]-margin, data_idx[-1]+margin)}
            obj = obj[test]
        return obj
