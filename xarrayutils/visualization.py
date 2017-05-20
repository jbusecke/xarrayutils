import matplotlib as mpl
mpl.use('Agg')
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xmitgcm import open_mdsdataset
from dask.array import from_array
from dask.multiprocessing import get
from dask.diagnostics import ProgressBar

# import argparse


def mitgcm_Movie(ddir,
                 prefix=['tracer_snapshots'],
                 maskname='hFacC',
                 varname='TRAC01',
                 clim=[-1, 35]):
    ds = open_mdsdataset(ddir, prefix=prefix, swap_dims=False)
    ds = ds[varname].where(ds[maskname] == 1)
    run = os.path.basename(os.path.normpath(ddir))
    odir = ddir + '/movie'
    Movie(ds, odir, clim=clim, moviename=run)


def Movie(da, odir,
          varname=None,
          framedim='time',
          moviename='movie',
          plotstyle='simple',
          clim=None,
          cmap=None,
          bgcolor=np.array([1, 1, 1]) * 0.3,
          framewidth=1280,
          frameheight=720,
          dpi=100,
          dask=True,
          delete=True,
          ffmpeg=True,
          ):
    # Set defaults:
    if not ffmpeg and delete:
        raise RuntimeError('raw picture deletion makes only \
            sense if ffmpeg conversion is enabled')

    if not isinstance(da, xr.DataArray):
        raise RuntimeError('input has to be an xarray DataStructure, instead\
        is ' + str(type(da)))

    if not os.path.exists(odir):
        os.makedirs(odir)

    # Infer defaults from data
    if clim is None:
        print('clim will be inferred from data, this can take very long...')
        clim = [da.min(), da.max()]
    if cmap is None:
        cmap = plt.cm.RdYlBu_r

    # Annnd here we go
    print('+++ Execute plot function +++')
    if dask:
        data = da.data
        frame_axis = da.get_axis_num(framedim)
        drop_axis = [da.get_axis_num(a) for a in da.dims if not a == framedim]
        chunks = list(data.shape)
        chunks[frame_axis] = 1
        data = data.rechunk(chunks)
        with ProgressBar():
            data.map_blocks(FramePrint, chunks=[1],
                            drop_axis=drop_axis,
                            dtype=np.float64,
                            dask=dask,
                            frame_axis=frame_axis,
                            odir=odir,
                            cmap=cmap,
                            clim=clim,
                            framewidth=framewidth,
                            frameheight=frameheight,
                            dpi=dpi,
                            bgcolor=bgcolor
                            ).compute(get=get)
    # The .compute(get=get) line is some dask 'magic': it parallelizes the
    # print function with processes and not threads,which is a lot faster
    # for custom functions apparently!

    else:
        # do it with a simple for loop...can this really be quicker?
        print('This is slow! Do it in dask!')
        for ii in range(0, len(da.time)):
            start_time = time.time()
            da_slice = da[{framedim: ii}]
            # fig,ax,h = FramePrint(da_slice,
            FramePrint(da_slice,
                       frame=ii,
                       odir=odir,
                       cmap=cmap,
                       clim=clim,
                       framewidth=framewidth,
                       frameheight=dpi,
                       bgcolor=bgcolor
                       )
            if ii % 100 == 0:
                remaining_time = (len(da.time) - ii) * \
                    (time.time() - start_time) / 60
                print('FRAME---%04d---' % ii)
                print('Estimated time left : %d minutes' % remaining_time)

    print('+++ Convert frames to video +++')
    query = 'ffmpeg -y -i "frame_%05d.png" -c:v libx264 -preset veryslow \
        -crf 6 -pix_fmt yuv420p \
        -framerate 15 \
        "' + moviename + '.mp4"'

    with cd(odir):
        if ffmpeg:
            excode = os.system(query)
        if excode == 0 and delete:
            os.system('rm *.png')


def SimplePlot(data, ax, cmap=None,
               clim=None,
               bgcolor=np.array([1, 1, 1]) * 0.3
               ):
    if cmap is None:
        cmap = plt.cm.Blues
    if clim is None:
        print('clim not defined. Will be deduced from data. \
        This could have undesired effects for videos')
        clim = [data.min(), data.max()]

    cmap.set_bad(bgcolor, 1)
    pixels = np.squeeze(np.ma.array(data, mask=np.isnan(data)))
    h = ax.imshow(pixels, cmap=cmap, clim=clim,
                  aspect='auto', interpolation='none')
    ax.invert_yaxis()
    return h


def MovieFrame(framewidth, frameheight, dpi):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(framewidth / dpi,
                        frameheight / dpi)
    return fig


def FramePrint(da,
               odir=None,
               frame=None,
               cmap=None,
               clim=None,
               bgcolor=np.array([1, 1, 1]) * 0.3,
               facecolor=np.array([1, 1, 1]) * 0.3,
               framewidth=1920,
               frameheight=1080,
               dpi=100,
               dask=False,
               block_id=None,
               frame_axis=None,
               plot_func=SimplePlot,
               ):
    """Prints the plotted picture to file


    """

    if not odir:
        raise RuntimeError('need an output directory')

    fig = MovieFrame(framewidth, frameheight, dpi)
# TODO plotsyle options

    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # fig.add_axes(ax)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor(facecolor)
    ax.set_aspect(1, anchor='C')

    if not dask:
        data = da.data
    else:
        data = da
        frame = block_id[frame_axis]

    plot_func(data, ax,
              cmap=cmap,
              clim=clim,
              bgcolor=bgcolor)
    #
    fig.savefig(odir + '/frame_%05d.png' % frame, dpi=fig.dpi)
    plt.close('all')
    # return fig,ax,h,dummy
    return from_array(np.array([0]), [1])


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
