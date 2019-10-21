import os
import time
from copy import copy
import random

import numpy as np
import dask

from dask import config
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler
from dask.diagnostics import visualize

from plotting import imagePlot, scatterPlot, sampleXYZData, scatter3dPlot
from processPTX import getRawXYZ
from main import importCsv
from utils import cart2sph, sph2cart, aggregateXYZ
from parabolas import loadLeicaDataFromGpus

#config.set(scheduler='distributed')  # overwrite default with multiprocessing scheduler
#config.set(scheduler='processes')  # overwrite default with multiprocessing scheduler

# N=20, 11 secs
# N=50, 63 secs

def digitizeWinDask(x, y, z, N):

    print "len data: ", len(x), len(y), len(z)

    assert len(x) == len(y)
    assert len(y) == len(z)

    # figure out how to group our data
    xMin = np.min(x)
    xMax = np.max(x)
    yMin = np.min(y)
    yMax = np.max(y)

    # what's the size of each cell?
    xBins = np.linspace(xMin, xMax, N)
    yBins = np.linspace(yMin, yMax, N)

    xd = np.digitize(x, xBins)
    yd = np.digitize(y, yBins)

    lbins = np.zeros((N, N))
    total = 0
    bins = {}

    start = time.time()
    print "delayed stage ..."
    for xi in range(1, N+1):
        print xi
        for yi in range(1, N+1):
            t = dask.delayed(getNumInBins(xi, yi, xd, yd))
            #t = getNumInBins(xi, yi, copy(xd), copy(yd))
            # t = getNumInBins(xi, yi, xd, yd)
            #print t
            #xx = xd == xi
            #yy = yd == yi
            #t = len(xd[np.logical_and(xx, yy)])
            #lbins[xi-1][yi-1] = t
            bins[(xi-1, yi-1)] = t
            #total += t
            
    end = time.time()
    print "elapsed secs: ", end - start

    return xd, yd, bins, lbins

    start = time.time()
    print "persist stage ..."        
    dask.persist(*bins)
    end = time.time()
    print "elapsed secs: ", end - start

    start = time.time()
    print "compute stage ..."
    total = 0
    i = 0

    keys = sorted(bins.keys())
    for k in keys:
    # for k, v in bins.items():
        v = bins[k] 
        #print i
        i += 1
        xi, yi = k
        #tt = v.compute(scheduler="processes")
        tt = v.compute()
        lbins[xi][yi] = tt
        total += tt

        #rSms = rSms.compute(scheduler="processes")

    end = time.time()
    print "elapsed secs: ", end - start
    print "total: ", total
    #print bins
    #print lbins

    return xBins, yBins, bins, lbins

def getNumInBins(xi, yi, xd, yd):
    xx = xd == xi
    yy = yd == yi
    return len(xd[np.logical_and(xx, yy)])

def demoDigitize():

    N = 5

    x = np.array(range(10)) 
    y = np.array(range(10)) 

    # figure out how to group our data
    xMin = np.min(x)
    xMax = np.max(x)
    yMin = np.min(y)
    yMax = np.max(y)

    # what's the size of each cell?
    xBins = np.linspace(xMin, xMax, N+1)
    yBins = np.linspace(yMin, yMax, N+1)

    print "x: ", x

    print "xBins: ", xBins
    
    xd = xBins[1] - xBins[0]
    print "x step size: ", xd
    xd2 = (xMax - xMin) / float((N - 1))

    print "alt. x step:", xd2

    xd = np.digitize(x, xBins)
    print "x digitized: ", xd

    xdf = np.digitize(x, xBins, right=True)
    print "x (right-True) digitized: ", xdf

    dataBinsLn = np.zeros(N-1)
    for i in range(N-1):
        dataBinsLn[i] = len(x[xd == i+1])

    dataBins = {}
    for i in range(N-1):
        b = x[xd == i+1]
        dataBins[i] = b


    print "dataBinsLn: ", dataBinsLn  


    # print "dataBins:", dataBins

    keys = sorted(dataBins.keys())
    for i, k in enumerate(keys):
        v = dataBins[k]
        print "xBin[i]: ", i, xBins[i], xBins[i+1]
        print "mean: ", np.mean(v)

def digitizeSphericalData(N=None):
    "bins our data using numpy.digitize"

    if N is None:
        N = 30

    fn = "/home/sandboxes/jbrandt/Telescope27Mar2019/Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx.csv.sph.csv"

    # reallY: r, az, el
    z, x, y = getDishXYZ(fn)
    
    print "len data: ", len(x), len(y), len(z)

    assert len(x) == len(y)
    assert len(y) == len(z)

    # x = x[:100]
    # y = y[:100]

    # figure out how to group our data
    xMin = np.min(x)
    xMax = np.max(x)
    yMin = np.min(y)
    yMax = np.max(y)

    # what's the size of each cell?
    # first, split our space up.  We use N+1 because
    # we want N bins, and N+1 gives us the last fence post,
    # that is, the end point of the last bin.
    xBins = np.linspace(xMin, xMax, N)
    yBins = np.linspace(yMin, yMax, N)

    # however, we use this same 
    xDelta = xBins[1] - xBins[0]
    yDelta = yBins[1] - yBins[0]

    print "org xBins", xBins
    print "xDelta", xDelta

    xBins = xBins - (xDelta/2.)
    yBins = yBins - (yDelta/2.)

    xBins = np.append(xBins, xBins[-1] + xDelta)
    yBins = np.append(yBins, yBins[-1] + yDelta)
    #print "bins:"
    #print xBins
    #print yBins

    
    xd = np.digitize(x, xBins)
    yd = np.digitize(y, yBins)

    print "digitized"

    xt = yt = 0
    bins = {}

    ind = np.array(range(len(xd)))

    lbins = np.zeros((N, N))
    # lbins = np.zeros((N-1, N-1))
    total = 0
    # for xi in range(1, N+1):
    for xi in range(N):
        print xi
        # for yi in range(1, N+1):
        for yi in range(N):
            #print xi, yi
            # import ipdb; ipdb.set_trace()
            #xij = [x[i] for i in range(len(x)) if xd[i] == xi and yd[i] == yi]
            # yij = [y[i] for i in yd if i == yi]
            #yij = [y[i] for i in range(len(y)) if xd[i] == xi and yd[i] == yi]
            #bins[(xi, yi)] = (xij, yij)
            #assert len(xij) == len(yij)
            #lbins[xi-1][yi-1] = len(xij)
            xx = xd == xi + 1
            yy = yd == yi + 1
            thisBin = np.logical_and(xx, yy)
            # t = len(xd[np.logical_and(xx, yy)])
            # xThisBin = x[thisBin]

            # assert np.all(np.logical_and(xThisBin >= xBins[xi], xThisBin < xBins[xi+1]))
            # lenXx2 = xThisBin[np.logical_and(xThisBin >= xBins[xi], xThisBin < xBins[xi+1])]

            # print "passed assertion in x for ", xi, xBins[xi], xBins[xi+1]
            # print "mean in this bin: ", np.mean(xThisBin)
            # if len(xThisBin) > 0:
            #     print "mean in this bin: ", np.min(xThisBin), np.max(xThisBin),np.mean(xThisBin)

            # yThisBin = y[thisBin]
            # zThisBin = z[thisBin]
            # bins[(xi, yi)] = (xThisBin, yThisBin, zThisBin)
            bins[(xi, yi)] = ind[thisBin]
            t = len(xd[thisBin])
            lbins[xi][yi] = t
            total += t
            # error check
            #xt += len(xij)
            #yt += len(yij)

    return xd, yd, bins, lbins

# N=20, 6 secs
# N=30, 14 secs
# N=50, 43 secs
# N=100, 160 secs
# N=256, 16 minutes
def digitizeWindow(x, y, z, N):
    "bins our data using numpy.digitize"

    print "len data: ", len(x), len(y), len(z)

    assert len(x) == len(y)
    assert len(y) == len(z)

    # x = x[:100]
    # y = y[:100]

    # figure out how to group our data
    xMin = np.min(x)
    xMax = np.max(x)
    yMin = np.min(y)
    yMax = np.max(y)

    # what's the size of each cell?
    # first, split our space up.  We use N+1 because
    # we want N bins, and N+1 gives us the last fence post,
    # that is, the end point of the last bin.
    xBins = np.linspace(xMin, xMax, N)
    yBins = np.linspace(yMin, yMax, N)

    # however, we use this same 
    xDelta = xBins[1] - xBins[0]
    yDelta = yBins[1] - yBins[0]

    print "org xBins", xBins
    print "xDelta", xDelta

    xBins = xBins - (xDelta/2.)
    yBins = yBins - (yDelta/2.)

    xBins = np.append(xBins, xBins[-1] + xDelta)
    yBins = np.append(yBins, yBins[-1] + yDelta)
    #print "bins:"
    #print xBins
    #print yBins

    
    xd = np.digitize(x, xBins)
    yd = np.digitize(y, yBins)

    

    #print "xd", xd
    #print "yd", yd


    xt = yt = 0
    bins = {}

    lbins = np.zeros((N, N))
    # lbins = np.zeros((N-1, N-1))
    total = 0
    # for xi in range(1, N+1):
    for xi in range(N):
        print xi
        # for yi in range(1, N+1):
        for yi in range(N):
            #print xi, yi
            # import ipdb; ipdb.set_trace()
            #xij = [x[i] for i in range(len(x)) if xd[i] == xi and yd[i] == yi]
            # yij = [y[i] for i in yd if i == yi]
            #yij = [y[i] for i in range(len(y)) if xd[i] == xi and yd[i] == yi]
            #bins[(xi, yi)] = (xij, yij)
            #assert len(xij) == len(yij)
            #lbins[xi-1][yi-1] = len(xij)
            xx = xd == xi + 1
            yy = yd == yi + 1
            thisBin = np.logical_and(xx, yy)
            # t = len(xd[np.logical_and(xx, yy)])
            xThisBin = x[thisBin]

            assert np.all(np.logical_and(xThisBin >= xBins[xi], xThisBin < xBins[xi+1]))
            # lenXx2 = xThisBin[np.logical_and(xThisBin >= xBins[xi], xThisBin < xBins[xi+1])]

            # print "passed assertion in x for ", xi, xBins[xi], xBins[xi+1]
            # print "mean in this bin: ", np.mean(xThisBin)
            # if len(xThisBin) > 0:
            #     print "mean in this bin: ", np.min(xThisBin), np.max(xThisBin),np.mean(xThisBin)

            yThisBin = y[thisBin]
            zThisBin = z[thisBin]
            bins[(xi, yi)] = (xThisBin, yThisBin, zThisBin)
            t = len(xd[thisBin])
            lbins[xi][yi] = t
            total += t
            # error check
            #xt += len(xij)
            #yt += len(yij)

    print lbins
    print "total: ", total
#    print "total bin len: ", xt, yt


    # for k, v in bins.items():
    keys = sorted(bins.keys())
    for k in keys:
        v = bins[k]
        # print ""
        # print k
        xi, yi = k
        xij, yij, zij = v
        # print "for x, y bins:", xBins[xi], yBins[yi]
        # if xi < N:
        #     print "for x bins:", xBins[xi], xBins[xi+1]
        #     print "for y bins:", yBins[yi], yBins[yi+1]
        # else:
        #     print "for x, y bins:", xBins[xi], yBins[yi]
        #     print "for x, y bins:", xBins[xi], yBins[yi]
        # print "for x bins:", xBins[xi], xBins[xi+1]
        # print "for y bins:", yBins[yi], yBins[yi+1]

        # print "lengths: ", len(xij), len(yij), len(zij)
        # print "means: ", np.mean(xij), np.mean(yij), np.mean(zij)

    return xBins, yBins, bins, lbins

def testDigitizeWindow():
    print "loading dish data ..."
    x, y, z = getDishXYZ()
    print "starting digitization ..."

    # x = np.array(range(3)) + 0.5
    # y = np.array(range(3)) + 0.5
    # xm, ym = np.meshgrid(x, y)
    # print x, y
    # print xm, ym
    # z = xm * ym

    # digitizeWindow(xm.flatten(), ym.flatten(), z.flatten(), 3)
    #digitizeWindow(x, y, z, 10)
    xd, yd, bins, lbins = digitizeWinDask(x, y, z, 3)

def demoDigitizeWindowScan(file=None, N=None):
    "Demonstrates that we can use digitize to take means and represent the dish"

    if N is None:
        N = 30

    if file is None:
        file = "Scan-9-sample.ptx.csv"

        
    x, y, z = getDishXYZ(fpath = "Scan-9-sample.ptx.csv")
    xd, yd, bins, lbins = digitizeWindow(x, y, z, N)

    imagePlot(lbins, "point density")

    # create the x y we need for 3d plots using our bins;
    # need this indexing option or our data looks like it 
    # got rotated 90 degrees.
    xm, ym = np.meshgrid(xd[:-1], yd[:-1], indexing='ij')

    scatter3dPlot(xm, ym, lbins, "point density")

    # calculate the means of each bin
    zm = np.zeros((N-1, N-1))
    for i in range(N-1):
        for j in range(N-1):
            xs, ys, zs = bins[(i,j)]
            zm[i][j] = np.mean(zs)

    imagePlot(zm, "means of z")

    scatter3dPlot(xm, ym, zm, "means of z")

    scatter3dPlot(x, y, z, "original data", sample=0.1)

def getWeight(az, el, azLoc, elLoc, sigAz, sigEl, j, k):
    return 2*np.pi*np.exp((-(np.cos(elLoc[j,k])**2)*(az-azLoc[j,k])**2/(2.*sigAz**2)-(el-elLoc[j,k])**2/(2.*sigEl**2 )))
    # return 2*np.pi*np.exp((-(np.cos(elLoc[j,k])**2)*(az-azLoc[j,k])**2/(2.*((r/a)*sigAz)**2)-(el-elLoc[j,k])**2/(2.*sigEl**2 )))

def getWeightNoCos(az, el, azLoc, elLoc, sigAz, sigEl, j, k):
    # return 2*np.pi*np.exp((az-azLoc[j,k])**2/(2.*sigAz**2)-(el-elLoc[j,k])**2/(2.*sigEl**2 ))
    azTerm = (az-azLoc[j,k])**2/(2.*sigAz**2)
    elTerm = (el-elLoc[j,k])**2/(2.*sigEl**2 )
    # print j, k, azTerm, elTerm, azTerm - elTerm
    return 2*np.pi*np.exp(azTerm - elTerm)


def testSmoothDigitalWins():

    # x, y, z = getDishXYZ(fpath = "Scan-9-sample.ptx.csv")
    # N = 10
    # xs, ys, zs = smoothDigitalWindows(x, y, z, N)

    x = range(1,5)
    y = range(1,5)
    xm, ym = np.meshgrid(x, y)
    z = xm * ym

    N = 4
    # xd, yd, bins, lbins = digitizeWindow(xm.flatten(), ym.flatten(), z.flatten(), N)
    
    # print "xd: ", xd
    # print "lbins: ", lbins
    # print "bins: ", bins

    smoothDigitalWinsXYZ(xm.flatten(), ym.flatten(), z.flatten(), N)

def testSmoothXYZvsGPUs(N=None):

    if N is None:
       N = 10

    # get the data
    fpath = "/home/sandboxes/jbrandt/Telescope27Mar2019/Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx.csv"
    x, y, z = getDishXYZ(fpath = fpath)    

    # first, do our new style bin smoothing
    xs, ys, zs = smoothDigitalWinsXYZ(x, y, z, N, sigWidth=1.0)

    # [pmargani@vegas-hpc10 /home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing]$
    # ./gpu_smooth --input /home/sandboxes/jbrandt/Telescope27Mar2019/Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx.csv 
    # --output Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx.csv 
    # --gridx 10 --gridy 10 --sigAz 1.0 --sigEl 1.0 -v --no-conv --no-cos --no-xyz-to-spherical --no-spherical-to-xyz

    smoothedFn = "data/binSmoothing/Scan-9.%d.xyz.ptx.csv" % N
    xg, yg, zg = loadLeicaDataFromGpus(smoothedFn)

    # necessary for comparison
    xst = xs.transpose()
    yst = ys.transpose()
    zst = zs.transpose()

    # make sure all is within bonds
    xDiff = xst.flatten() - xg
    print "xDiff mean, std", np.nanmean(xDiff), np.nanstd(xDiff)
    assert np.max(np.abs(xDiff)) < 1e-5

    yDiff = yst.flatten() - yg
    print "yDiff mean, std", np.nanmean(yDiff), np.nanstd(yDiff)
    assert np.max(np.abs(yDiff)) < 1e-5

    zDiff = zst.flatten() - zg
    print "zDiff mean, std", np.nanmean(zDiff), np.nanstd(zDiff)
    # assert np.nanmax(np.abs(zDiff)) < 2.0
    # assert np.nanmean(np.abs(zDiff)) < 0.2

    return (xg, yg, zg), (xst, yst, zst)

def testSmoothBinVsGPUs(N=None):

    if N is None:
       N = 10

    # get the data
    fpath = "/home/sandboxes/jbrandt/Telescope27Mar2019/Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx.csv"
    x, y, z = getDishXYZ(fpath = fpath)    

    # first, do our new style bin smoothing
    xs, ys, zs = smoothDigitalWindows(x, y, z, N)

    # [pmargani@vegas-hpc10 /home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing]$
    # ./gpu_smooth --input /home/sandboxes/jbrandt/Telescope27Mar2019/Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx.csv 
    # --output Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx.csv 
    # --gridx 10 --gridy 10 --sigAz 1.0 --sigEl 1.0 -v --no-conv --no-cos --no-xyz-to-spherical --no-spherical-to-xyz

    smoothedFn = "data/binSmoothing/Scan-9.%d.ptx.csv" % N
    xg, yg, zg = loadLeicaDataFromGpus(smoothedFn)

    # necessary for comparison
    xst = xs.transpose()
    yst = ys.transpose()
    zst = zs.transpose()

    # make sure all is within bonds
    xDiff = xst.flatten() - xg
    print "xDiff mean, std", np.nanmean(xDiff), np.nanstd(xDiff)
    # assert np.max(np.abs(xDiff)) < 1e-5

    yDiff = yst.flatten() - yg
    print "yDiff mean, std", np.nanmean(yDiff), np.nanstd(yDiff)
    # assert np.max(np.abs(yDiff)) < 1e-5

    zDiff = zst.flatten() - zg
    print "zDiff mean, std", np.nanmean(zDiff), np.nanstd(zDiff)
    # assert np.nanmax(np.abs(zDiff)) < 2.0
    # assert np.nanmean(np.abs(zDiff)) < 0.2

    return (xg, yg, zg), (xst, yst, zst)
    
def getLocalBin(bins, i, j):
    return bins[(i, j)]

    # print "getLocalBins", i, j
    # what are the max i and j?
    keys = bins.keys()
    ln = np.sqrt(len(keys))
    x = np.array([])
    y = np.array([])
    z = np.array([])
    for ii in range(i-1,i+2):
        for jj in range(j-1, j+2):
            if ii >= 0 and jj >= 0 and ii < ln and jj < ln:
                # print " using: ", ii, jj
                xi, yi, zi = bins[(ii, jj)]
                x = np.concatenate((x, xi))
                y = np.concatenate((y, yi))
                z = np.concatenate((z, zi))
    return x, y, z

def smoothDigitalWinsXYZ(x, y, z, N, sigWidth=None):

    print "smoothDigitalWindows", N

    print "digitize ..."
    s = time.time()
    xd, yd, bins, lbins = digitizeWindow(x, y, z, N)
    print "seconds: ", time.time() - s

    # xLoc, yLoc = np.meshgrid(xd[:-1], yd[:-1], indexing='ij')
    # Make sure we center each guassian in the MIDPOINT of each bin 
    # from above.  xd, yd represent the left hand side of each bin.
    xDelta = xd[1] - xd[0]
    yDelta = yd[1] - yd[0]
    xdd = xd + (xDelta/2.)
    ydd = yd + (yDelta/2.)
    xLoc, yLoc = np.meshgrid(xdd[:-1], ydd[:-1], indexing='ij')

    print "xDelta, xd", xDelta, xd
    print "xLoc: ", np.nanmin(xLoc), np.nanmax(xLoc), xLoc[1,0] - xLoc[0,0]
    print "yLoc: ", np.nanmin(yLoc), np.nanmax(yLoc), yLoc[0,1] - yLoc[0,0]

    # sigAz = sigEl = 0.001
    if sigWidth is None:
        sigAz = sigEl = 1.00
    else:    
        sigAz = sigEl = sigWidth

    # init our smoothing result
    zSm = np.ndarray(shape=(N,N))

    s = time.time()
    print "smooth ..."
    for i in range(N):
        for j in range(N):
            # x, y, z = bins[(i, j)]
            x, y, z = getLocalBin(bins, i, j)
            # print i, j, x
            # instead of passing in all the data, lats, lngs,
            # we just pass it in the LOCAL data, az, el.
            w = getWeightNoCos(x, y, xLoc, yLoc, sigAz, sigEl, i, j)
            norm=sum(w)
            if norm==0:
                norm=1
                zSm[i,j]=np.nan #0 #min( r )
            else:
                w = w / norm
                zSm[i,j] = sum(z * w)


    print "seconds: ", time.time() - s

    print "z smoothed:", zSm
    return xLoc, yLoc, zSm

def smoothDigitalWindows(x, y, z, N, cartesian=True):

    print "smoothDigitalWindows", N

    if cartesian:
        print "cart2sph ..."
        s = time.time()
        rs, lats, lngs = cart2sph(x, y, z, verbose=False)
        print "seconds: ", time.time() - s
    else:
        rs = x
        lats = y
        lngs = z

    print "digitize ..."
    s = time.time()
    latd, lngd, bins, lbins = digitizeWindow(lats, lngs, rs, N)
    print "seconds: ", time.time() - s

    latDelta = latd[1] - latd[0]
    lngDelta = lngd[1] - lngd[0]
    latdd = latd + (latDelta/2.)
    lngdd = lngd + (lngDelta/2.)
    azLoc, elLoc = np.meshgrid(latdd[:-1], lngdd[:-1], indexing='ij')

    # azLoc, elLoc = np.meshgrid(latd[:-1], lngd[:-1], indexing='ij')

    print "azLoc: ", np.nanmin(azLoc), np.nanmax(azLoc), azLoc[1] - azLoc[0]
    print "elLoc: ", np.nanmin(elLoc), np.nanmax(elLoc), elLoc[1] - elLoc[0]

    sigAz = sigEl = 0.001

    # init our smoothing result
    rSm = np.ndarray(shape=(N,N))

    s = time.time()
    print "smooth ..."
    for i in range(N):
        for j in range(N):
            az, el, r = bins[(i, j)]
            # instead of passing in all the data, lats, lngs,
            # we just pass it in the LOCAL data, az, el.
            w = getWeight(az, el, azLoc, elLoc, sigAz, sigEl, i, j)
            norm=sum(w)
            if norm==0:
                norm=1
                rSm[i,j]=np.nan #0 #min( r )
            else:
                w = w / norm
                rSm[i,j] = sum(r * w)
    print "seconds: ", time.time() - s

    print "sph2cart ..."
    s = time.time()
    xyz = sph2cart(azLoc.flatten(), elLoc.flatten(), rSm.flatten(), verbose=False)
    print "seconds: ", time.time() - s

    # return (azLoc, elLoc, rSm)
    # if cartesian:
    #     print "sph2cart ..."
    #     s = time.time()
    #     xyz = sph2cart(azLoc.flatten(), elLoc.flatten(), rSm.flatten(), verbose=False)
    #     print "seconds: ", time.time() - s
    # else:
    #     xyz = (azLoc, elLoc, rSm)

    return xyz

def selfWindow(x, y, z, N):

    assert len(x) == len(y)
    assert len(y) == len(z)

    # figure out how to group our data
    xMin = np.min(x)
    xMax = np.max(x)
    yMin = np.min(y)
    yMax = np.max(y)

    # what's the size of each cell?
    xRange = np.linspace(xMin, xMax, N)
    yRange = np.linspace(yMin, yMax, N)

    xSize = (xMax - xMin) / N
    ySize = (yMax - yMin) / N

    print "x, y sizes: ", xSize, ySize

    r = np.sqrt(xSize**2 + ySize**2) / 2.

    print "radius of each group", r

    # here's our collection nodes
    rs = {}

    # mark the first one
    rs[(x[0], y[0])] = [(x[0], y[0], z[0])]

    for i in range(1, len(x)):
        # import ipdb; ipdb.set_trace()
        xi = x[i]
        yi = y[i]
        zi = z[i]

        # find distance between all nodes
        inBin = False
        for xn, yn in rs.keys():
            rn = np.sqrt((xn - xi)**2 + (yn - yi)**2)
            if rn <= r:
                rs[(xn, yn)].append((xi, yi, zi))
                inBin = True
                break

        # if we aren't in a bin, make a new one
        if not inBin:
            rs[(xi, yi)] = [(xi, yi, zi)]


    # rss = np.zeros((N, N))
    t = 0
    for k, v in rs.items():
        print k, len(v)
        t += len(v)
        # rss[int(k[0]), int(k[1])] = len(v)
        
    print "total: ", t
    
    rss = makeDistImg(rs, N)    
    imagePlot(rss, "rss")
        
    keys = rs.keys() 
    xns = [x for x, y in keys]   
    yns = [y for x, y in keys]
    zns = [len(v) for k, v in rs.items()]   
    scatterPlot(xns, yns, "xns, yns")
    scatter3dPlot(xns, yns, zns, "xns, yns, zns")

    return rs

def makeDistImg(rs, N):

    xs = [x for x, y in rs.keys()]
    xmax = np.max(xs)
    xmin = np.min(xs)

    ys = [y for x, y in rs.keys()]
    ymax = np.max(ys)
    ymin = np.min(ys)

    rss = np.zeros((N, N))
    for k, v in rs.items():
        x, y = k
        xn = int((x - xmin) / N)
        yn = int((y - ymin) / N)
        rss[xn, yn] = len(v)

    return rss

def getRegularXY(m):
    xs = np.array(range(m), dtype=float)
    ys = np.array(range(m), dtype=float)

    xm, ym = np.meshgrid(xs, ys)
    return xm, ym

def getRandomXY(m, rng):

    xs = np.random.random(m) * rng
    ys = np.random.random(m) * rng

    print "random xs:", xs
    print "random ys:", ys
    xm, ym = np.meshgrid(xs, ys)

    return xm, ym

def getLeicaXYZ():

    path = "/home/sandboxes/jbrandt/Telescope27Mar2019"
    f = "Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx"
    fpath = os.path.join(path, f)

    with open(fpath, 'r') as f:
        ls = f.readlines()

    x, y, z, i = getRawXYZ(ls)    

    return x, y, z

def createSamplePTXFile():

    path = "/home/sandboxes/jbrandt/Telescope27Mar2019"
    f = "Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx.csv"
    fpath = os.path.join(path, f)

    with open(fpath, 'r') as f:
        ls = f.readlines()

    print "opened original file"
    dataLs = ls

    # fSample = "Sample-%s" % f
    fSample = "Scan-9-sample.ptx.csv"

    prcnt = 10.
    numSamples = int(len(dataLs) / prcnt)
    # idx = random.sample(range(len(dataLs)), numSamples)

    print "getting sample lines"
    # sampleLs = [l for i, l in enumerate(dataLs) if i in idx]
    sampleLs = random.sample(dataLs, numSamples)

    print "writing sample file"
    with open(fSample, 'w') as f:
        f.writelines(sampleLs)

    
def getDishXYZ(fpath=None):

    if fpath is None:
        path = "/home/sandboxes/jbrandt/Telescope27Mar2019"
        f = "Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx.csv"
        fpath = os.path.join(path, f)

    return importCsv(fpath)

    
def selfWindowTest():


    m = 100
    # getRegularXY(m)
    # xm, ym = getRandomXY(m, m)
    
    # zs = xm * ym
    xm, ym, zs = getDishXYZ()
    xm, ym, zs = sampleXYZData(xm, ym, zs, 1.0)
    print "num points: ", len(xm)
    
    # visualize our test input
    scatterPlot(xm.flatten(), ym.flatten(), "org data", sample=10.0)

    rs = selfWindow(xm.flatten(), ym.flatten(), zs.flatten(), m)

def windowData(x, y, z, N):

    print("windowing by: ", N)
    #print "x: "
    #print x
    #print "y: "
    #print y
    #print "z: "
    #print z

    xMin = np.min(x)
    xMax = np.max(x)
    yMin = np.min(y)
    yMax = np.max(y)

    xRange = np.linspace(xMin, xMax, N)
    yRange = np.linspace(yMin, yMax, N)

    xLoc, yLoc = np.meshgrid(xRange, yRange) 

    xys = np.array(zip(x, y))

    #print "xys: ", xys

    ws = {}
    wi = 0
    for ix in range(N-1):
        for iy in range(N-1):
            #print "ix, iy: ", ix, iy
            x0 = xRange[ix]
            x1 = xRange[ix+1]
            y0 = yRange[iy]
            y1 = yRange[iy+1]
            rng = ((x0, x1), (y0, y1))
            #print "rng: ", rng
            #xyi = [i for i, xy in enumerate(xys) if xy[0] >= x0 and xy[0] < x1 and xy[1] >= y0 and xy[1] < y1]
            xyi = [i for i, xy in enumerate(xys) if xy[0] >= x0 and xy[0] <= x1 and xy[1] >= y0 and xy[1] <= y1]
            #print "xyi: ", xyi
            xysf = xys[xyi]
            #xw = np.where(np.logical_and(x <= x1, x >x0))
            #yw = np.where(np.logical_and(y <= y1, y >y0))
            #print "xw: ", xw
            #print "yw: ", yw
            #zw = list(set(list(xw[0])).intersection(set(list(yw[0]))))
            #ws[wi] = (rng, (x[xw], y[yw], z[zw]))
            ws[wi] = ((ix, iy), rng, (xysf, z[xyi]))
            wi += 1

    return ws

def windowDataOld(x, y, z, N):

    print("windowing by: ", N)
    #print "x: "
    #print x
    #print "y: "
    #print y
    #print "z: "
    #print z

    xMin = np.min(x)
    xMax = np.max(x)
    yMin = np.min(y)
    yMax = np.max(y)

    xRange = np.linspace(xMin, xMax, N)
    yRange = np.linspace(yMin, yMax, N)

    xLoc, yLoc = np.meshgrid(xRange, yRange) 

    xys = zip(x, y)

    ws = {}
    wi = 0
    for ix in range(N-1):
        for iy in range(N-1):
            #print "ix, iy: ", ix, iy
            x0 = xRange[ix]
            x1 = xRange[ix+1]
            y0 = yRange[iy]
            y1 = yRange[iy+1]
            rng = ((x0, x1), (y0, y1))
            #print "rng: ", rng
            #xyi = [i for i, xy in enumerate(xyz)]
            xw = np.where(np.logical_and(x <= x1, x >x0))
            yw = np.where(np.logical_and(y <= y1, y >y0))
            #print "xw: ", xw
            #print "yw: ", yw
            zw = list(set(list(xw[0])).intersection(set(list(yw[0]))))
            print("zw: ", zw)
            ws[wi] = (rng, (x[xw], y[yw], z[zw]))
            #ws[wi] = (rng, (xysf, z[zw]))
            wi += 1
    return ws

def getWeight(az, el, azLoc, elLoc, sigAz, sigEl, j, k):
    return 2*np.pi*np.exp( (- (az - azLoc[j,k])**2 /( 2.*sigAz**2 )-(el-elLoc[j,k])**2 /(2.*sigEl**2 )))
    #return 2*np.pi*np.exp((-(np.cos(elLoc[j,k])**2)*(az-azLoc[j,k])**2/(2.*sigAz**2)-(el-elLoc[j,k])**2/(2.*sigEl**2 )))

def getWeightHere(az, el, azLoc, elLoc, sigAz, sigEl):
    return 2*np.pi*np.exp( (- (az - azLoc)**2 /( 2.*sigAz**2 )-(el-elLoc)**2 /(2.*sigEl**2 )))
    
def assignWeight(w, r):
    norm=sum(w)
    if norm==0:
        norm=1
        v=np.nan #0 #min( r )
        #v=0 #0 #min( r )
    else:
        w = w / norm
        v = sum(r * w)   
    return v

def smooth(az, el, r, n, sigEl=None, sigAz=None):
    "smooth our data"

    azRange = np.linspace(min(az), max(az), n)
    elRange = np.linspace(min(el), max(el), n)

    azLoc, elLoc = np.meshgrid(azRange, elRange)

    if sigEl is None:
        sigEl=0.001
    if sigAz is None:    
        sigAz=0.001;

    #import ipdb; ipdb.set_trace()
    # init our smoothing result
    #rSm = np.ndarray(shape=(n,n))
    rSm = np.zeros((n, n))
    rSms = []
    for j in range(n):
        #print "J:", j
        for k in range(n):
            #print "K:", k
            #w=2*np.pi*np.exp( (- (az - azLoc[j,k])**2 /( 2.*sigAz**2 )-(el-elLoc[j,k])**2 /(2.*sigEl**2 )))
            #norm=sum(w)
            #if norm==0:
            #    norm=1
            #    rSm[j,k]=np.nan #0 #min( r )
            #else:
            #    w = w / norm
            #    rSm[j,k] = sum(r * w)
            w = getWeight(az, el, azLoc, elLoc, sigAz, sigEl, j, k)
            #print w
            tmp = assignWeight(w, r)
            #norm=sum(w)
            #print "norm: ", norm
            #w = w / norm
            #print "w: ", w
            #print "r: ", r
            #tmp = sum(r*w)
            #print "tmp: ", tmp
            #print "rsm[j, k]: ", rSm[j, k]

            rSm[j, k] = tmp 
            
    return (azLoc, elLoc, rSm)  

def smoothWin(ws, sigAz=None, sigEl=None):


    if sigEl is None:
        sigEl=0.001
    if sigAz is None:    
        sigAz=0.001;

    n = len(ws.keys())
    #rSm = np.zeros((n, n))
    rSm = np.zeros((n,))

    for k, v in ws.items():
        ixy, rng, values = v
        xys, zs = values
        x = [xy[0] for xy in xys]
        y = [xy[1] for xy in xys]
        xRng = rng[0]
        yRng = rng[1]
        xCntr = xRng[0] + ((xRng[1] - xRng[0]) / 2.)
        yCntr = yRng[0] + ((yRng[1] - yRng[0]) / 2.)
        w = getWeightHere(x, y, xCntr, yCntr, sigAz, sigEl)
        rSm[int(k)] = assignWeight(w, zs)

    return rSm

def mainTest():

    m = 5
    xs = np.array(range(m), dtype=float)
    ys = np.array(range(m), dtype=float)

    xm, ym = np.meshgrid(xs, ys)

    print("xm: ")
    print(xm)
    print("ym: ")
    print(ym)

    zs = xm * ym
    
    #np.zeros((m,m))

    #ws = windowData(xs, ys, 10)
    #print ws

    n = 3
    xLoc, yLoc, zsmooth = smooth(xm.flatten(), ym.flatten(), zs.flatten(), n, sigAz=1., sigEl=1.)

    print("data: ")
    print(zs)

    print("smoothed to: ")
    print(zsmooth)

    ws = windowData(xm.flatten(), ym.flatten(), zs.flatten(), n+1)
    print(ws.keys())
    for k, v in ws.items():
        ixy, rng, values = v
        _, zzz = values
        print(rng, np.mean(zzz))

    zWinSmooth = smoothWin(ws, sigAz=1.0, sigEl=1.0)    

    print("zWinSmooth: ")
    print(zWinSmooth)

def main():

    m = 100
    xs = np.array(range(m), dtype=float)
    ys = np.array(range(m), dtype=float)

    xm, ym = np.meshgrid(xs, ys)

    zs = xm * ym
    
    n = 10
    xLoc, yLoc, zsmooth = smooth(xm.flatten(), ym.flatten(), zs.flatten(), n, sigAz=1., sigEl=1.)

    print("smoothed to: ")
    print(zsmooth)

    ws = windowData(xm.flatten(), ym.flatten(), zs.flatten(), n+1)
    print(ws.keys())
    for k, v in ws.items():
        ixy, rng, values = v
        _, zzz = values
        #print rng, np.mean(zzz)

    zWinSmooth = smoothWin(ws, sigAz=1.0, sigEl=1.0)    

    print("zWinSmooth: ")
    zWinSmooth.shape = (n, n)
    print(zWinSmooth)

if __name__=='__main__':
    # selfWindowTest()
    # testDigitizeWindow()
    # demoDigitize()
    # createSamplePTXFile()    
    # testSmoothDigitalWins()
    # testSmoothDigitalWins()
    testSmoothXYZvsGPUs()
