import os
from copy import copy

import numpy as np

from plotting import scatter3dPlot, sampleXYZData, imagePlot
from utils.utils import cart2sph, sph2cart, splitXYZ, aggregateXYZ
from gpus import loadLeicaDataFromGpus, smoothGPUs

# where is the code we'll be running?
GPU_PATH = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing"

def weightSmooth(fpath, xyz):

    N = 512

    # convert to shperical
    # s = 1.0
    # print "working with %f percent of data" % s
    x, y, z = splitXYZ(xyz)
    # x, y, z = sampleXYZData(x, y, z, s)

    scatter3dPlot(x, y, z, "org sample of data", sample=1.0)

    xOrg = copy(x)
    yOrg = copy(y)
    zOrg = copy(z)

    # sigma2 = <r>**2 - <r**2>
    ################# FIRST: calculate <r>

    smoothedXYZfiles = smooth(fpath)

    basename = os.path.basename(fpath)
    gpuPath = os.path.join(GPU_PATH, basename)
    print("loading GPU data from", gpuPath)
    xs, ys, zs = loadLeicaDataFromGpus(gpuPath)    

    scatter3dPlot(xs, ys, zs, "sample of smoothed data", sample=1.0)

    # we need the smoothed r, <r>, so convert this
    print("Converting to spherical ...")
    rs, els, azs = cart2sph(xs, ys, zs)


    ################ Second: calcualte <r**2>
    # convert the original xyz to spherical
    r, el, az = cart2sph(x, y, z)

    # square the r term, then go back to cartesian
    r2 = r**2
    x2, y2, z2 = sph2cart(el, az, r2)

    # write this to a file for smoothing
    xyz2 = aggregateXYZ(x2, y2, z2)
    f2 = basename + ".rs.csv"
    basepath = os.path.dirname(fpath)     
    fpath2 = os.path.join(basepath, f2)
    print("Saving r**2 data in xyz to file", fpath2)
    np.savetxt(fpath2, xyz2, delimiter=',')

    # now smooth this data
    smoothedXYZ2files = smooth(fpath2)
    
    # retrieve the xyz data
    gpuPath2 = os.path.join(GPU_PATH, f2)
    print("loading GPU data from", gpuPath2)
    xs2, ys2, zs2 = loadLeicaDataFromGpus(gpuPath2)

    # convert back to shperical to get <r**2>
    rs2, els2, azs2 = cart2sph(xs2, ys2, zs2)

    ####################: Finally: Calculate Variance
        # make sure small negative numerical errors are dealt with
    sigma2 = np.abs(rs2 - (rs**2))

    print("sigma squared: ", sigma2.shape, np.nanmin(sigma2), np.nanmax(sigma2), np.nanmean(sigma2), np.nanstd(sigma2))

    # not normalized weights
    Ws_Not_Norm= 1/sigma2

    # make sure NaNs induced by zeros in sigma2 are dealt with
    Ws_Not_Norm[sigma2 == 0.0] = 0.0

    # normalize weights, making sure Nans in sum are dealt with
    ws = (Ws_Not_Norm) / np.sum(Ws_Not_Norm[np.logical_not(np.isnan(sigma2))])

    print("Computed weights: ", ws.shape, ws)

    # plot weights
    sigma2.shape = (N, N)
    imagePlot(sigma2, "sigma^2")
    imagePlot(np.log(np.abs(sigma2)), "log sigma^2")
    wsc = copy(ws)
    wsc.shape = (N, N)
    imagePlot(wsc, "weights")
    imagePlot(np.log(np.abs(wsc)), "log weights")

    return smoothedXYZfiles, ws

def smooth(fpath, N=512, spherical=False):

    assert os.path.isfile(fpath)

    # we need to specify the aboslute path of our inputs
    abspath = os.path.abspath(fpath)

    # our output will be same file name, but with appendages
    outfile = os.path.basename(fpath)

    smoothGPUs(GPU_PATH,
               abspath,
               outfile,
               N,
               spherical=spherical)

    # make sure the output is where it should be
    outfiles = []
    for dim in ['x', 'y', 'z']:
        dimFile = "%s.%s.csv" % (outfile, dim)
        dimPath = os.path.join(GPU_PATH, dimFile)
        outfiles.append(dimPath)
        assert os.path.isfile(dimPath)
        print("GPUs created file: ", dimPath)

    return outfiles
