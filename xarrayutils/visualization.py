import matplotlib as mpl
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xmitgcm import open_mdsdataset
# import argparse
# from dask.diagnostics import ProgressBar
from dask.array import from_array
from dask.multiprocessing import get
mpl.use('Agg')


def mitgcm_Movie(ddir,
                 prefix=['tracer_snapshots'],
                 maskname='hFacC',
                 varname='TRAC01',
                 clim=[-1,  35]):
    ds = open_mdsdataset(ddir, prefix=prefix,
                         swap_dims=False)
    ds = ds[varname].where(ds[maskname] == 1)
    run = os.path.basename(os.path.normpath(ddir))
    odir = ddir+'/movie'
    Movie(ds, odir, clim=clim, moviename=run)


def Movie(da, odir,
          varname=None,
          framedim='time',
          moviename='movie',
          plotstyle='simple',
          clim=None,
          cmap=None,
          bgcolor=np.array([1, 1, 1])*0.3,
          framewidth=1280,
          frameheight=720,
          dpi=100,
          dask=True,
          ):
    # Set defaults:

    if not isinstance(da, xr.DataArray):
        raise RuntimeError('input has to be an xarray DataStructure, instead\
        is '+str(type(da)))

    if not os.path.exists(odir):
        os.makedirs(odir)

    # Infer defaults from data
    if not clim:
        print('clim will be inferred from data, this can take very long...')
        clim = [da.min(),da.max()]
    if not cmap:
        cmap = plt.cm.RdYlBu_r

    # Annnd here we go
    print('+++ Execute plot function +++')
    if dask:
        data   = da.data
        frame_axis = da.get_axis_num(framedim)
        drop_axis = [da.get_axis_num(a) for a in da.dims if not a == framedim]
        chunks = list(data.shape)
        chunks[frame_axis] = 1
        data = data.rechunk(chunks)
        # with ProgressBar():
        dummy = data.map_blocks(FramePrint,chunks = [1],
                                drop_axis = drop_axis,
                                dtype=np.float64,
                                dask=dask,
                                frame_axis = frame_axis,
                                odir        = odir,
                                cmap        = cmap,
                                clim        = clim,
                                framewidth  = framewidth,
                                frameheight = frameheight,
                                dpi         = dpi,
                                bgcolor     = bgcolor
                                # this line is some dask 'magic': it parallelizes
                                # the print function with processes and not threads,
                                # which is a lot faster for custom functions
                                # apparently!
                                ).compute(get=get)

    else:
        # do it with a simple for loop...can this really be quicker?
        for ii in range(0,len(da.time)):
            start_time = time.time()
            da_slice = da[{framedim:ii}]
            # fig,ax,h = FramePrint(da_slice,
            dummy = FramePrint(da_slice,
                                    frame=ii,
                                    odir        = odir,
                                    cmap        = cmap,
                                    clim        = clim,
                                    framewidth  = framewidth,
                                    frameheight = frameheight,
                                    dpi         = dpi,
                                    bgcolor     = bgcolor
                                    )
            if ii % 100 == 0:
                remaining_time = (len(da.time)-ii)*(time.time() - start_time)/60
                print('FRAME---%04d---' %ii)
                print('Estimated time left : %d minutes' %remaining_time)



    print('+++ Convert frames to video +++')
    query = 'ffmpeg -y -i "frame_%05d.png" -c:v libx264 -preset veryslow \
        -crf 6 -pix_fmt yuv420p \
        -framerate 20 \
        "'+moviename+'.mp4"'

    with cd(odir):
        os.system(query)
        os.system('rm *.png')

def FramePrint(da,odir=None,
                    frame=None,
                    cmap=None,
                    clim = None,
                    bgcolor = np.array([1,1,1])*0.3,
                    facecolor = np.array([1,1,1])*0.3,
                    framewidth  = 1920,
                    frameheight = 1080,
                    dpi         = 100,
                    dask        = False,
                    block_id    = None,
                    frame_axis  = None
                    ):
    """Prints the plotted picture to file


    """

    if not odir:
        raise RuntimeError('need an output directory')

    fig = MovieFrame(framewidth,frameheight,dpi)
    # TODO plotsyle options
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_facecolor(facecolor)
    ax.set_aspect(1, anchor = 'C')
    fig.add_axes(ax)

    if not dask:
        data   = da.data
    else:
        data   = da
        frame  = block_id[frame_axis]

    h = SimplePlot(data,ax,
                    cmap=cmap,
                    clim=clim,
                    bgcolor=bgcolor)
    #
    fig.savefig(odir+'/frame_%05d.png' %frame, dpi=fig.dpi)
    plt.close('all')
    # return fig,ax,h,dummy
    return from_array(np.array([0]),[1])

def SimplePlot(data,ax,cmap = None,
                        clim = None,
                        bgcolor = np.array([1,1,1])*0.3
                        ):
    if not cmap:
        cmap = plt.cm.Blues
    if not clim:
        print('clim not defined. Will be deduced from data. \
        This could have undesired effects for videos')
        clim = [data.min(),data.max()]

    cmap.set_bad(bgcolor, 1)
    pixels = np.squeeze(np.ma.array(data, mask=np.isnan(data)))
    h = ax.imshow(pixels,cmap=cmap,clim=clim,aspect='auto',interpolation='none')
    ax.invert_yaxis()
    return h

def MovieFrame(framewidth,frameheight,dpi):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(framewidth/dpi,
                        frameheight/dpi)
    return fig

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='xarray Movie script')
#
#     parser.add_argument('ddir',help='input array')
#
#     parser.add_argument('-odir','--outdir',
#         help='output directory',default=None)
#     parser.add_argument('-v','--varname',
#         help='diagnostic name',default='TRAC01')
#     parser.add_argument('-cl','--clim',
#         help='color limit',default=None)
#     parser.add_argument('-di','--framedim',
#         help='video time dim',default='time')
#     parser.add_argument('-n','--moviename',
#         help='output filename',default='movie')
#     parser.add_argument('-st','--plotstyle',
#         help='plotting style',default='simple_fullscreen')
#
#     args = parser.parse_args()
#
#     # ## show values ##
#     # print ("Grid Directory: %s" % args.grid_dir)
#     # print ("Delayed Time Directory: %s" % args.dt_dir)
#     # print ("Near Real Time Directory: %s" % args.nrt_dir)
#     # print ("Output Directory: %s" % args.out_dir)
#     # print ("Delayed Time Name: %s" % args.fid_dt)
#     # print ("Real Time Name: %s" % args.fid_nrt)
#     # print ("--- ---")
