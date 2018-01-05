from datetime import datetime, timedelta
import pandas as pd
import os
from os.path import join as pjoin
import xarray as xr
import numpy as np
import gsw
from dask.array import zeros_like
from . utils import concat_dim_da
from . weighted_operations import weighted_mean, weighted_sum


def load_obs(fid, read_kwargs=dict(), swap_dims=None, rename=None,
             drop=None, squeeze=None, dtype_convert=None, unit_conversion=None,
             shift_lon_dim='xt_ocean', shift_lon_kwargs=None):

    """
    Reads and 'massages' observational dataset to facilitate comparison
    with model dataa

    INPUT
    =====
    fid: {str, list} : path or list of paths to data.
    read_kwargs: {None, dict} Arguments passed to xarray.open_dataset
    swap_dims: {None, dict} Arguments passed to xarray.Dataset.swap_dims
    rename: {None, dict} Dictionary passed to xarray.Dataset.rename
    drop: {None, list} List passed to xarray.Dataset.drop
    dtype_convert: {None, e.g. np.float64} dtype that all variables (also
                coords, dims) are converted too. If not there can be weird
                issues with the lon shift (Mimoc np.float32 data could not be
                edited for instance)
    unit_conversion: {None, dict} Dict of
                     {varname: (conversion_factor, new_units)} to be applied
                     as follows: ds['varname] = ds['varname']*conversion_factor
    """
    if isinstance(fid, str):
        ds = xr.open_dataset(fid, **read_kwargs)
    elif isinstance(fid, list):
        ds = xr.open_mfdataset(fid, **read_kwargs)
    else:
        raise RuntimeError('fid has to be a single path or a list of paths')

    if swap_dims is not None:
        ds = ds.swap_dims(swap_dims)
    if rename is not None:
        ds = ds.rename(rename)
    if drop is not None:
        ds = ds.drop(drop,)
    if squeeze is not None:
        ds = ds.squeeze(squeeze, drop=True)

    if dtype_convert is not None:
        for cc in list(ds.data_vars):
            ds[cc].data = ds[cc].data.astype(dtype_convert)
        for cc in list(ds.coords):
            if not isinstance(ds[cc].data[0], str):
                ds[cc].data = ds[cc].data.astype(dtype_convert)

    if unit_conversion is not None:
        for kk in list(unit_conversion.keys()):
            ds[kk].data = ds[kk].data*unit_conversion[kk][0]
            ds[kk].attrs['units'] = unit_conversion[kk][1]

    if shift_lon_kwargs is not None:
            ds = shift_lon(ds, shift_lon_dim, **shift_lon_kwargs)
    return ds


def load_obs_dict(fid_dict=None, drop_dict=None, mimoc_fix=True, debug=False,
                  calc_teos=True):
    """ Load multiple datasets into a dictionary.
    Time average and combine to single dataset if 'combo' is activated.

    INPUT
    =====
    fid_dict: {None, str, list, dict} Defaults to a 'standard selection of
              datsets. Dict has to have the structure like
              {name:(fid, load_obs_kwargs)}, where fid is the file location
              and load_obs_kwargs is a dict with input keywords for load_obs.
              If list/str is passed, only those datasetd from the standard
              selection are loaded.
    drop_dict: {None, dict} Dict which gives drop variables for each dataset.
              In case only certain vars are desired.
    calc_teos: {bool} activates calculation of additional teos-10 variables.
              default on.
    """
    # Determine input
    if fid_dict is None:
        load_list = 'all'
    else:
        if isinstance(fid_dict, dict):
            load_list = 'all'
        elif isinstance(fid_dict, list):
            load_list = fid_dict
        elif isinstance(fid_dict, str):
            load_list = [fid_dict]
        else:
            raise RuntimeError("'fid_dict' has to be a dict, \
                               list of strings or str")
    if debug:
        print(load_list)

    def glodap_preprocess(ds):
        ref = list(ds.data_vars)[0]
        ds = ds.assign_coords(Depth=ds['Depth'], SnR=ds['SnR'], CL=ds['CL'])
        f_dim = concat_dim_da(['data', 'error', 'relerr', 'input_mean',
                               'input_std', 'input_n'], 'field')
        ds_new = xr.concat([ds[ref], ds[ref+'_error'], ds[ref+'_relerr'],
                            ds['Input_mean'], ds['Input_std'], ds['Input_N']],
                           dim=f_dim). \
            assign_attrs(units=ds[ref].attrs['units']). \
            to_dataset()
        return ds_new

    def woa_preprocessing(ds):
        variables = ['A', 'i', 'n', 'o', 'O', 'p', 'I', 's', 't']
        fields = ['an', 'mn', 'dd', 'sd', 'se', 'oa', 'ma', 'gp']
        field_names = ['obj_analyzed_data', 'mean', 'n_obs', 'std',
                       'err', 'oa', 'ma', 'gp']
        idx = slice(0, 4)
        f_dim = concat_dim_da(field_names[idx], 'field')
        datasets = []
        for vv in variables:
            datasets.append(xr.concat([ds[vv+'_'+a] for a in fields[idx]],
                                      dim=f_dim))
            datasets[-1].attrs = ds[vv+'_an'].attrs
        ds_new = xr.merge(datasets)
        return ds_new

    def glodap_flist():
        ddir = '/work/Julius.Busecke/shared_data/GLODAPv2.2016b_MappedClimatologies'
        fid = [pjoin(ddir, a) for a in ['GLODAPv2.2016b.Cant.nc',
                                        'GLODAPv2.2016b.NO3.nc',
                                        'GLODAPv2.2016b.OmegaA.nc',
                                        'GLODAPv2.2016b.OmegaC.nc',
                                        'GLODAPv2.2016b.oxygen.nc',
                                        'GLODAPv2.2016b.pHts25p0.nc',
                                        'GLODAPv2.2016b.pHtsinsitutp.nc',
                                        'GLODAPv2.2016b.PI_TCO2.nc',
                                        'GLODAPv2.2016b.PO4.nc',
                                        'GLODAPv2.2016b.salinity.nc',
                                        'GLODAPv2.2016b.silicate.nc',
                                        'GLODAPv2.2016b.TAlk.nc',
                                        'GLODAPv2.2016b.TCO2.nc',
                                        'GLODAPv2.2016b.temperature.nc']]
        return fid

    def woa_convert_conc(c, rho, molvol):
        """This conversion requires water density, which is not always known.
        For dissolved oxigen, 1 ml/l is approximately 43.554 umol/kg assuming
        a constant seawater density of 1025 kg/m^3 and a molar volume
        of O2 (22.4 l)."""

        out = c/rho/molvol
        out.attrs['units'] = 'mol/kg'
        return out

    def calc_teos10(ds):
        # TODO add units and long names
        # Create necessary variables with teos-10, perhaps I should

        ds['pr'] = xr.apply_ufunc(gsw.p_from_z, -ds['st_ocean'],
                                  ds['yt_ocean'],
                                  output_dtypes=[np.float64],
                                  dask='parallelized')
        ds['sa'] = xr.apply_ufunc(gsw.SA_from_SP, ds['salt'], ds['pr'],
                                  ds['xt_ocean'], ds['yt_ocean'],
                                  output_dtypes=[np.float64],
                                  dask='parallelized')
        if 'temp' in list(ds.data_vars):
            ds['ct'] = xr.apply_ufunc(gsw.CT_from_pt, ds['sa'], ds['temp'],
                                      output_dtypes=[np.float64],
                                      dask='parallelized')
        else:
            ds['ct'] = xr.apply_ufunc(gsw.CT_from_t, ds['sa'], ds['te'],
                                      ds['pr'],
                                      output_dtypes=[np.float64],
                                      dask='parallelized')
            ds['temp'] = xr.apply_ufunc(gsw.pt_from_CT, ds['sa'], ds['ct'],
                                        output_dtypes=[np.float64],
                                        dask='parallelized')
        ds['pot_rho_0'] = xr.apply_ufunc(gsw.sigma0, ds['sa'], ds['ct'],
                                         output_dtypes=[np.float64],
                                         dask='parallelized')+1000
        return ds

    ds_dict = dict()

    if not isinstance(fid_dict, dict):
        fid_dict = {
            'CM26_init': ('/work/Julius.Busecke/CM2.6_staged/init/WOA01-05_CM2.6_annual.nc',
                          dict(
                              read_kwargs={'chunks': {'depth': 1}},
                              rename={'depth': 'st_ocean',
                                      'longitude': 'xt_ocean',
                                      'latitude': 'yt_ocean'},
                              squeeze=['time'],
                          )),
            'AVISO': ('/work/Julius.Busecke/shared_data/aviso/zos_AVISO_L4_199210-201012.nc',
                      dict(
                        read_kwargs=dict(chunks={'time': 1}),
                        rename={'zos': 'eta_t', 'lon': 'xt_ocean',
                                'lat': 'yt_ocean'}
                          )),
            'WOA13': (['/work/Julius.Busecke/shared_data/woa/woa13.nc'],
                      dict(
                    read_kwargs=dict(decode_times=False, chunks={'depth': 1},
                                     preprocess=woa_preprocessing),
                    dtype_convert=np.float64,
                    rename={
                        'lon': 'xt_ocean',
                        'lat': 'yt_ocean',
                        'depth': 'st_ocean',
                        'time': 'month',
                        'A_an': 'aou',
                        'i_an': 'silicate',
                        'n_an': 'no3',
                        'I_an': 'dens',
                        'p_an': 'po4',
                        'O_an': 'o2_sat_perc',
                        'o_an': 'o2',
                        't_an': 'te',
                        's_an': 'salt'
                    },
                    )),
            'GLODAPv2': (glodap_flist(),
                         dict(
                            read_kwargs=dict(preprocess=glodap_preprocess,
                                             chunks={'depth_surface': 1}),
                            swap_dims=dict(depth_surface='Depth'),
                            rename=dict(Depth='st_ocean', lat='yt_ocean',
                                        lon='xt_ocean', PO4='po4', NO3='no3',
                                        oxygen='o2', salinity='salt',
                                        temperature='temp'),
                            unit_conversion={
                                'po4':  (1e-6, 'mols/kg'),
                                'no3': (1e-6, 'mols/kg'),
                                'o2': (1e-6, 'mols/kg'),
                                'Cant': (1e-6, 'mols/kg'),
                                'silicate': (1e-6, 'mols/kg'),
                            },
                        )),
            'MLD_deBoyer': ('/work/Julius.Busecke/shared_data/MLD_deBoyerMontegut/MLD_deBoyerMontegut_combined.nc',
                            dict(
                              read_kwargs=dict(chunks={'time': 1},
                                               decode_times=False),
                              dtype_convert=np.float64,
                              rename=dict(lon='xt_ocean', lat='yt_ocean',
                                          time='month'),
                              shift_lon_kwargs=dict(shift=360, crit=0,
                                                    smaller=True)
                              )),
            'MLD_Holte': ('/work/Julius.Busecke/shared_data/Argo_mixedlayers_monthlyclim_03192017.nc',
                          dict(
                             read_kwargs=dict(chunks={'iMONTH': 1}),
                             swap_dims={'iLON': 'lon', 'iLAT': 'lat',
                                        'iMONTH': 'month'},
                             rename=dict(lat='yt_ocean', lon='xt_ocean'),
                             shift_lon_kwargs=dict(shift=360, crit=0,
                                                   smaller=True)
                              )),
            'MIMOC': ('/work/Julius.Busecke/shared_data/mimoc/MIMOC_Z_GRID_v2.2_PT_S.nc',
                      dict(
                          read_kwargs=dict(chunks={'month_of_year': 1}),
                          dtype_convert=np.float64,
                          rename=dict(lat='yt_ocean', long='xt_ocean',
                                      month_of_year='month', PRES='st_ocean',
                                      SALINITY='salt',
                                      POTENTIAL_TEMPERATURE='temp'),
                          shift_lon_kwargs=dict(shift=360, crit=0,
                                                smaller=True)
                      )),
            'Bianchi': ('/work/Julius.Busecke/shared_data/O2_bianchi.nc',
                        dict(
                           read_kwargs=dict(chunks={'TIME': 1, 'DEPTH': 1}),
                           drop=['DEPTH_bnds', 'TIME_bnds'],
                           rename=dict(LATITUDE='yt_ocean',
                                       LONGITUDE='xt_ocean',
                                       TIME='month', DEPTH='st_ocean',
                                       O2_LINEAR='o2'),
                           unit_conversion={'o2': (1e-6, 'mol/kg')},
                           shift_lon_kwargs=dict(shift=360, crit=0,
                                                 smaller=True)
                           ))
                    }

        if load_list == 'all':
            load_list = list(fid_dict.keys())

        if debug:
            print(load_list)
            print(fid_dict)

        for kk in load_list:
            if debug:
                print(kk)
            ds_dict[kk] = load_obs(fid_dict[kk][0], **fid_dict[kk][1])

        # special fixes for default input
        if 'MIMOC' in list(ds_dict.keys()):
            ds_dict['MIMOC']['st_ocean'] = ds_dict['MIMOC']['PRESSURE'].\
                isel(month=0, drop=True)
            ds_dict['MIMOC'] = ds_dict['MIMOC'].drop('PRESSURE')
            if calc_teos:
                ds_dict['MIMOC'] = calc_teos10(ds_dict['MIMOC'])

        if 'WOA13' in list(ds_dict.keys()):
            ds_dict['WOA13']['dens'] = ds_dict['WOA13']['dens']+1000
            ds_dict['WOA13']['o2'] = woa_convert_conc(ds_dict['WOA13']['o2'],
                                                      ds_dict['WOA13']['dens'],
                                                      22.4)  # molweight oxygen
            ds_dict['WOA13']['aou'] = woa_convert_conc(ds_dict['WOA13']['aou'],
                                                       ds_dict['WOA13']['dens'],
                                                       22.4)
            ds_dict['WOA13']['o2_sat'] = ds_dict['WOA13']['o2'] + \
                (ds_dict['WOA13']['aou'])
            for nut in ['po4', 'no3']:
                ds_dict['WOA13'][nut] = ds_dict['WOA13'][nut] / \
                    ds_dict['WOA13']['dens']*1e-3
            if calc_teos:
                ds_dict['WOA13'] = calc_teos10(ds_dict['WOA13'])

        # This could be integrated into the read_kwargs above by adding a field
        if drop_dict is not None:
            for kk in list(drop_dict.keys()):
                ds_dict[kk] = ds_dict[kk].drop(drop_dict[kk])

        return ds_dict


def cm26_load_obs(ref_ds, combo=True, masking=True):
    # TODO: I should replace the ref_ds with the grid file once ready
    ds_dict = load_obs_dict()

    # Select fields for data that is given with errors...
    if combo:
        for ff in list(ds_dict.keys()):
            if ff == 'GLODAPv2':
                ds_dict[ff] = ds_dict[ff].sel(field='data', drop=True)
            if ff == 'WOA13':
                ds_dict[ff] = ds_dict[ff].sel(field='obj_analyzed_data',
                                              drop=True)

    # Convert lon to CM2.6 convention and reindex
    for kk in list(ds_dict.keys()):
        ds_dict[kk] = shift_lon(ds_dict[kk], 'xt_ocean', shift=-360,
                                crit=360-270, smaller=False)
        ds_dict[kk] = ds_dict[kk].reindex_like(ref_ds, method='nearest')

    if masking:
        mask = ref_ds[list(ref_ds.data_vars)[0]]
        non_mask_dims = [a for a in list(mask.dims) if a not in ['xt_ocean',
                                                                 'yt_ocean']]
        for nn in non_mask_dims:
            mask = mask[{nn: 0}]
        for kk in list(ds_dict.keys()):
            ds_dict[kk] = ds_dict[kk].where(~xr.ufuncs.isnan(mask))

    if combo:
        c = []
        for ff in list(ds_dict.keys()):
            if 'month' in list(ds_dict[ff].dims):
                cc = ds_dict[ff].mean('month', keep_attrs=True)
            elif 'time' in list(ds_dict[ff].dims):
                cc = ds_dict[ff].mean('time', keep_attrs=True)
            else:
                cc = ds_dict[ff]

            for cv in list(cc.data_vars):
                cc = cc.rename({cv: '%s_%s' % (ff, cv)})
            c.append(cc)

        ds_dict = xr.merge(c)
    return ds_dict


def cm26_flist(ddir, name, years=None, yearformat='%04i0101'):
    if years:
        name = name.replace('*', yearformat)
        out = [pjoin(ddir, name % yy) for yy in years]
    else:
        out = pjoin(ddir, name)
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
            paths = [pjoin(odir,
                '%04i_%s_%s.nc' %(y, fname, kk)) for y in years]
            xr.save_mfdataset(datasets, paths)
        else:
            metrics[kk].to_netcdf(pjoin(odir, '%s_%s.nc' % (fname, kk)),
                                  **kwargs)


def metrics_load(metrics, odir, fname, **kwargs):
    for kk in metrics.keys():
        metrics[kk] = xr.open_dataset(pjoin(odir,
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
                             years=None,
                             clean_coords=False,
                             read_kwargs=dict()):

    global_file_kwargs = dict(
        decode_times=False,
        autoclose=autoclose
    )

    global_file_kwargs.update(read_kwargs)

    # choose the run directory
    if run == 'control':
        rundir = pjoin(rootdir, 'CM2.6_A_Control-1860_V03')
    elif run == 'forced':
        rundir = pjoin(rootdir, 'CM2.6_A_V03_1PctTo2X')

    if not years:
        years = range(121, 201)

    if name == 'minibling_fields':
        path = pjoin(rundir, 'annual_averages/minibling_fields')
        name = '*.field.nc'
        yearformat = '%04i'
        file_kwargs = dict(drop_variables=['area_t',  'geolat_t',
                                           'geolon_t', 'average_T1',
                                           'average_T2', 'average_DT',
                                           'time_bounds', 'nv',
                                           'st_edges_ocean', 'chl'],
                           chunks={'time': 1, 'st_ocean': 1},)
    elif name == 'physics':
        path = pjoin(rundir, 'annual_averages/ocean')
        name = 'ocean.*.ann.nc'
        yearformat = '%04i'
        file_kwargs = dict(drop_variables=['area_t',  'geolat_t',
                                           'geolon_t', 'average_T1',
                                           'average_T2', 'average_DT',
                                           'time_bounds', 'nv',
                                           'geolon_c', 'geolat_c',
                                           'ty_trans', 'salt_int_rhodz',
                                           'sea_level', 'sea_levelsq',
                                           'sfc_hflux_coupler',
                                           'temp_int_rhodz',
                                           'frazil_2d', 'net_sfc_heating',
                                           'pme_river', 'river'],
                           chunks={'time': 1, 'st_ocean': 1, 'sw_ocean': 1},)
    elif name == 'osat':
        path = pjoin(rundir, 'annual_averages/o2_sat')
        if run == 'control':
            name = '*.control_o2_sat.nc'
        else:
            name = '*.forced_o2_sat.nc'
        yearformat = '%04i'
        file_kwargs = dict(chunks={'st_ocean': 1},
                           concat_dim='time')
    elif name == 'minibling_src':
        path = pjoin(rundir, 'annual_averages/budgets')
        name = '*.src.nc'
        yearformat = '%04i'
        file_kwargs = dict(drop_variables=['area_t',  'geolat_t',
                                           'geolon_t', 'average_T1',
                                           'average_T2', 'average_DT',
                                           'time_bounds', 'nv', 'dic_stf',
                                           'dic_btf', 'o2_stf', 'o2_btf',
                                           'po4_btf'],
                           chunks={'time': 1, 'st_ocean': 1})

    else:
        raise RuntimeError('name not recognized')
    # for debugging
    print

    flist = cm26_flist(path, name, years=years, yearformat=yearformat)
    if print_flist:
        print(flist)
    file_kwargs.update(global_file_kwargs)
    ds = xr.open_mfdataset(flist, **(file_kwargs))
    if clean_coords:
        drop_coords = [a for a in list(ds.coords) if a not in ['time']]
        ds = ds.drop(drop_coords)
    return ds


def cm26_reconstruct_annual_grid(ds, load=None):
    ds = ds.copy()
    # If I do this 'trick' with the ones, I make sure that dzt has the same
    # dimensions as the data_vars
    template = ds['temp'][{'time': 1,
                         'st_ocean': 1}].drop(['time', 'st_ocean'])
    area = ds['area_t']

    # activates loading of presaved dzt value
    load_kwargs = dict(decode_times=False, concat_dim='time',
                       chunks={'st_ocean': 1})
    if load == 'control':
        odir = '/work/Julius.Busecke/CM2.6_staged/CM2.6_A_Control-1860_V03/annual_averages/grid_fields'
        ds_dzt = xr.open_mfdataset(pjoin(odir, '*dzt_control.nc'),
                                   **load_kwargs)
        dz = ds_dzt['dzt']

    elif load == 'forced':
        odir = '/work/Julius.Busecke/CM2.6_staged/CM2.6_A_V03_1PctTo2X/annual_averages/grid_fields'
        ds_dzt = xr.open_mfdataset(pjoin(odir, '*dzt_forced.nc'),
                                   **load_kwargs)
        dz = ds_dzt['dzt']

    elif load is None:

        # attempted fix to deal with the mismatch between eta, wet and tracer
        # fields (mask the full dimension one array in space with wet)
        oceanmask = ds['wet']
        ones = (ds['temp']*0+1).where(oceanmask)
        ht = xr.DataArray(ds['ht'].data,
                          dims=template.dims,
                          coords=template.coords)
        eta = ds['eta_t']
        dz_star = xr.DataArray(ds['st_edges_ocean'].diff('st_edges_ocean').data,
                               dims=['st_ocean'],
                               coords={'st_ocean': ds['st_ocean']})
        dz = ones*dz_star*(1+(eta/ht.data))
        dz = dz.chunk({'st_ocean': 1})

    ds['dzt'] = dz
    ds['volume_t'] = ds['dzt']*ds['area_t']
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
                     grid_load=False,
                     compute_aou=True,
                     region=None,
                     read_kwargs=dict(),
                     debug=False):
    """Master read in function for CM2.6. Merges all variables into one
    dataset. If specified, 'normalize_budgets divides by dzt.
    'budget_drop' defaults to all non o2 variables from src file
    to save time."""

    if region is not None:
        if isinstance(region, str):
            region = region2masknum(region)
            regionstr = masknum2region(region)

    if 'detrended' in run:
        read_kwargs_default = dict(decode_times=False, concat_dim='time',
                               autoclose=True,
                               drop_variables=['area_t', 'dzt',  'volume_t',
                                               'geolon_t', 'geolat_t', 'ht',
                                               'kmt', 'dyt', 'wet', 'dxt'])
        read_kwargs_default.update(read_kwargs)
        if run == 'control_detrended':
            rundir = pjoin(rootdir, 'CM2.6_A_Control-1860_V03/annual_averages/detrended')
        elif run == 'forced_detrended':
            rundir = pjoin(rootdir, 'CM2.6_A_V03_1PctTo2X/annual_averages/detrended')

        if region is None:
            fid = pjoin(rundir, '*_%s.nc' % (run))
        else:
            fid = pjoin(rundir, '*_%s_%s.nc' % (run, regionstr))
        ds = xr.open_mfdataset(fid, **read_kwargs_default)

        # rechunk
        ds = ds.chunk({'st_ocean':1, 'sw_ocean':1, 'time':1})

        # Deactivate options that only apply to the non detrended data
        normalize_budgets=False
        region = None
        grid_load = True

    else:
        read_kwargs_default = dict(decode_times=False, concat_dim='time',
                               autoclose=True,
                               drop_variables=['area_t', 'dzt',  'volume_t',
                                               'geolon_t', 'geolat_t', 'ht',
                                               'kmt', 'dyt', 'wet', 'dxt',
                                               'st_edges_ocean',
                                               'sw_edges_ocean'])
        read_kwargs_default.update(read_kwargs)
        ds_minibling_field = cm26_readin_annual_means('minibling_fields',
                                                      run,
                                                      rootdir=rootdir,
                                                      read_kwargs=read_kwargs_default)
        ds_physics = cm26_readin_annual_means('physics',
                                              run,
                                              rootdir=rootdir,
                                              read_kwargs=read_kwargs_default)
        ds_osat = cm26_readin_annual_means('osat',
                                           run,
                                           rootdir=rootdir,
                                           read_kwargs=read_kwargs_default)
        ds_minibling_src = cm26_readin_annual_means('minibling_src',
                                                    run,
                                                    rootdir=rootdir,
                                                    read_kwargs=read_kwargs_default)
        if debug:
            print('raw read done')
            print('ds_minibling_field',list(ds_minibling_field.coords))
            print('ds_physics',list(ds_physics.coords))
            print('ds_osat',list(ds_osat.coords))
            print('ds_minibling_src',list(ds_minibling_src.coords))


        # Brute force the minibling time into all files
        # ######## THEY DONT HAVE THE SAME TIMESTAMP MOTHERFUCK....
        ds_physics.time.data = ds_minibling_field.time.data
        ds_osat.time.data = ds_minibling_field.time.data

        # TODO: Build test if the time is equal .
        # for now just watch the timesteps
        ds = xr.merge([ds_minibling_field, ds_physics,
                      ds_osat, ds_minibling_src])

    # now update all coords from one files
    grid_path = '/work/Julius.Busecke/CM2.6_staged/static/grid_complete.nc'
    chunks_raw = {'st_ocean': 1, 'sw_ocean': 1,
                  'xt_ocean': 3600, 'xu_ocean': 3600,
                  'yt_ocean': 2700, 'yu_ocean': 2700}
    ds_grid = xr.open_dataset(grid_path, chunks=chunks_raw)
    ds.update(ds_grid)

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
        convert_vars = [a for a in convert_vars if a in list(ds.data_vars)]
        for vv in convert_vars:
            print('%s is beiung divided by rho_dzt' % vv)
            # ds[vv] = ds[vv]/1035.0/ds['dzt']
            # Fast version without checking
            ds[vv].data = ds[vv].data/1035.0/ds['dzt'].data

    if region:
        ds = cm26_cut_region(ds, region)

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
    paths = [pjoin(odir, '%04i.'+name) % y for y in years]
    xr.save_mfdataset(datasets, paths, **kwargs)


def cm26_cut_region(obj, region, remove_nan_domain=True,
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

    if rename_dict is None:
        rename_dict = dict(XT_OCEAN='xt_ocean', YT_OCEAN='yt_ocean',
                           XU_OCEAN='xu_ocean', YU_OCEAN='yu_ocean')

    regionmask = xr.open_dataset(regionfile).rename(rename_dict)
    # TODO: Extend this to u mask for matching variables
    # At the same time, I should find a way to adjust the tolerance
    # for coordinate comparison, so I dont have to swap them
    regionmask.coords.update(obj.coords)
    # regionmask.coords['xt_ocean'].data = obj.coords['xt_ocean'].data
    # regionmask.coords['yt_ocean'].data = obj.coords['yt_ocean'].data
    ##
    umask = regionmask['UMASK'] == region
    tmask = regionmask['TMASK'] == region

    # Mask data (still full domain)
    for vv in list(obj.data_vars):
        if set(['xt_ocean', 'yt_ocean']).issubset(set(obj[vv].dims)):
            obj[vv] = obj[vv].where(tmask)
        elif set(['xu_ocean', 'yu_ocean']).issubset(set(obj[vv].dims)):
            obj[vv] = obj[vv].where(umask)
        else:
            print('Regionmask not applied to ""s"%' % vv)
            print(obj[vv])

    #Cheap implementation of a nan cut: Check where the regionmask (only t)
    # is nan along a dimension and cut that sucker
    if remove_nan_domain:
        mask = tmask
        margin = 0
        for x in ['xt_ocean', 'xu_ocean']:
            if x in obj.dims:
                data_idx = np.where(mask.any('yt_ocean'))[0]
                test = {x: slice(data_idx[0]-margin, data_idx[-1]+margin)}
                obj = obj[test]
        for y in ['yt_ocean', 'yu_ocean']:
            if y in obj.dims:
                data_idx = np.where(mask.any('xt_ocean'))[0]
                test = {y: slice(data_idx[0]-margin, data_idx[-1]+margin)}
                obj = obj[test]

    return obj
