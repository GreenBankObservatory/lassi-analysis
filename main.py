import csv
import os
from copy import copy

import numpy as np
from astropy.coordinates import cartesian_to_spherical
from astropy.coordinates import spherical_to_cartesian
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d, Axes3D
#import opticspy
from scipy import interpolate

from dask import delayed
from dask import config
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler
from dask.diagnostics import visualize

from processPTX import splitXYZ

from plotting import *

from parabolas import fitLeicaScan

import settings

GPU_PATH = settings.GPU_PATH

# config.set(scheduler='distributed')  # overwrite default with multiprocessing scheduler

def process(fn, n, outputName, noise=False):

    # 10: seconds
    # 50: 2.5 mins
    # 100: 8 mins
    # n = 20

    print "importing CSV data ..."
    x, y, z = importCsv(fn)

    print x.shape, y.shape, z.shape

    print x[0], y[0], z[0]
    print type(x[0])
    print type(x)

    print "Converting to spherical coords ..."
    r, az, el = cart2sph(x, y, z)


    # print "min/max r", r.min(), r.max()

    # print "unwrapping az values ..."
    # az = unwrap(az)

    print "smoothing data ..."
    azLoc, elLoc, rSmooth = smooth(az, el, r, n)

    print "azLoc", azLoc
    print "elLoc", elLoc
    if noise:
        print "adding noise to radial values ..."
        rSmooth = rSmooth + 1*np.random.randn(n,n)

    #print rSmooth
    # fig = plt.figure()
    # ax = fig.add_subplot(111) #, projection='3d')
    # ax = Axes3D(fig)

    # plots!
    print "plotting ..."
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(azLoc, elLoc, rSmooth)
    plt.show()

    plt.imshow(rSmooth, interpolation="nearest", origin="upper")
    plt.colorbar()
    plt.show()


    # back to x, y, z
    print "converting back to cartesian ..."
    xs, ys, zs = sph2cart(azLoc, elLoc, rSmooth)

    # save for later analysis
    np.savez("smoothXYZ%d" % n, x=xs, y=ys, z=zs)

    print "plotting x y z ..."
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xs, ys, zs)
    plt.show()

    print "plotting x y z as scatter to see spacing"
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xs, ys, zs)
    plt.show()

    print "smoothing x y z"
    sxs, sys, szs = interpXYZ(xs, ys, zs, n)

    # print "zernikes of xyz surface"
    # zs2 = copy(zs)
    # zs2[np.isnan(zs2)] = 0.
    # zernikeFit(zs2)

    
    # print "smoothing x y z"
    # xss, yss, zss = smoothXYZ(xs, ys, zs, n, sigX=0.25, sigY=0.25)

    # print "plotting final smoothed x y z"
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(xss, yss, zss)
    # plt.show()

    # print "plotting final smoothed x y z as scatter"
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(xss, yss, zss)
    # plt.show()

    # find the zernike fit to this!
    # first convert NaN's to zeros
    # zss[np.isnan(zs)] = 0.
    zs[np.isnan(zs)] = 0.

    print "zernikes of smoothed xyz"
    zernikeFit(zss)

    # save processing results
    # np.savez(outputName, x=xss, y=yss, z=zss)
    # np.savez(outputName, x=xs, y=ys, z=zs)

def smoothWithWeights(fpath, N=None, useWeights=True):
    """
    tests idea of producing weights from smoothing
    and applying them in the fitting
    """
    if N is None:
        N=512

    print "Loading data from file ..."
    data = np.loadtxt(fpath, delimiter=',')
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    # work with a sample of the data
    down = 0.1
    seed = 1971
    print "Downsampling data by %f" % down
    x, y, z = sampleXYZData(x, y, z, down, seed=seed)

    scatter3dPlot(x, y, z, "downsampled by %f" % down)

    print "Converting to spherical coords ..."
    # r, az, el = cart2sph(x, y, z)
    r, el, az = cart2sph(x, y, z)

    print "smoothing data ..."
    azLoc, elLoc, rSmooth = smoothDask(az, el, r, N)
    print "smoothing squared data ..."
    azLoc, elLoc, rSmooth2 = smoothDask(az, el, r**2, N)

    nrsmooth2 = rSmooth2[np.logical_not(np.isnan(rSmooth2))]

    print "num pos rsmooth2", len(nrsmooth2[nrsmooth2 > 0.])
    print "num neg rsmooth2", len(nrsmooth2[nrsmooth2 < 0.])   


    # TEacher says we can cheet
    sigmat = rSmooth2 - (rSmooth**2)
    mask = sigmat < 0.
    print sigmat[mask], rSmooth2[mask], rSmooth[mask]**2

    sigma2 = np.abs(rSmooth2 - (rSmooth**2))


    print "Size of sigma2 vs. num zeros", sigma2.shape, len(sigma2[sigma2 == 0.0])

    nsigma2 = sigma2[np.logical_not(np.isnan(sigma2))]

    print "num pos sigma2", len(nsigma2[nsigma2 > 0.])
    print "num neg sigma2", len(nsigma2[nsigma2 < 0.])

    surface3dPlot(azLoc, elLoc, sigma2, "sigma squared")


    print "plotting ..."
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(azLoc, elLoc, rSmooth)
    # plt.show()
    surface3dPlot(azLoc, elLoc, rSmooth, "Spherical smoothed")

    # plt.imshow(rSmooth, interpolation="nearest", origin="upper")
    # plt.colorbar()
    # plt.show()
    imagePlot(rSmooth, "R smoothed")
    imagePlot(rSmooth2, "R**2 smoothed")

    # back to x, y, z
    print "converting back to cartesian ..."
    # xs, ys, zs = sph2cart(azLoc, elLoc, rSmooth)
    xs, ys, zs = sph2cart(elLoc, azLoc, rSmooth)

    scatter3dPlot(xs.flatten(), ys.flatten(), zs.flatten(), "smoothed xyz")

    # return xs, ys, zs, sigma2
    Ws_Not_Norm= 1/sigma2

    # make sure NaNs induced by zeros in sigma2 are dealt with
    Ws_Not_Norm[sigma2 == 0.0] = 0.0


    #ws = (1/sigma2) / np.sum(1/sigma2[np.logical_not(np.isnan(sigma2))])
    ws = (Ws_Not_Norm) / np.sum(Ws_Not_Norm[np.logical_not(np.isnan(sigma2))])
 
    print "weights", ws, np.isnan(ws).all()

    if useWeights:
        xs, ys, diffs = fitLeicaScan(None, xyz=(xs, ys, zs), weights=ws)
    else:    
        print "Not actually using the weights we calculated!"
        xs, ys, diffs = fitLeicaScan(None, xyz=(xs, ys, zs))

    return xs, ys, diffs, sigma2, ws
    
    
def getCenterData(x, y, z, n, m):

    start = n / m
    end = n - m

    return x[start:end, start:end], y[start:end], z[start:end]

def interpXYZ(x, y, z, n, xrange=None, yrange=None, checkLevels=False, center=False):

    x2 = copy(x)
    y2 = copy(y)
    z2 = copy(z)

    if xrange is None:
        xMin = np.nanmin(x2)
        xMax = np.nanmax(x2)
    else:
        xMin = xrange[0]    
        xMax = xrange[1]

    if yrange is None:        
        yMin = np.nanmin(y2)
        yMax = np.nanmax(y2)
    else:
        yMin = yrange[0]
        yMax = yrange[1]

    # zmin = np.nanmin(z2)
    # zmax = np.nanmax(z2)

    # We'll replace the NaN's with zero's and flatten our coordinate 
    # matrices to fit an interpolation function to all our data
    #x2[np.isnan(x2)] = 0.
    #y2[np.isnan(x2)] = 0.
    #z2[np.isnan(x2)] = 0.

    #x2[np.isnan(y2)] = 0.
    #y2[np.isnan(y2)] = 0.
    #z2[np.isnan(y2)] = 0.

    #x2[np.isnan(z2)] = 0.
    #y2[np.isnan(z2)] = 0.
    #z2[np.isnan(z2)] = 0.

    x2f = x2.flatten()
    y2f = y2.flatten()
    z2f = z2.flatten()

    cond = [not b for b in np.isnan(x2f)]
    x3 = np.extract(cond, x2f)

    cond = [not b for b in np.isnan(y2f)]
    y3 = np.extract(cond, y2f)

    cond = [not b for b in np.isnan(z2f)]
    z3 = np.extract(cond, z2f)

    print "Removed %d NaNs from %d data points" % (len(x2f) - len(x3), len(x2f))

    assert len(x3) == len(y3)
    assert len(y3) == len(z3)

    f = interpolate.interp2d(x3, y3, z3, kind='cubic')
    #f = interpolate.interp2d(x2.flatten(),
    #                         y2.flatten(),
    #                         z2.flatten(),
    #                         kind='cubic')

    # TBD: just use the center region to avoid edge cases and 0's and Nans!

    d = 4.

    xD = xMax - xMin
    xStart = xMin + ((1/d)*xD)
    xEnd = xMin + (((d-1)/d)*xD)
        
    yD = yMax - yMin
    yStart = yMin + ((1/d)*yD)
    yEnd = yMin + (((d-1)/d)*yD)

    if center:
        xnew = np.linspace(xStart, xEnd, n)
        ynew = np.linspace(yStart, yEnd, n)
    else:    
        xnew = np.linspace(xMin, xMax, n)
        ynew = np.linspace(yMin, yMax, n)

    # use the intropolation function to get our new surface
    znew = f(xnew, ynew)

    # we'll need the evenly spaced meshgrid too
    mx, my = np.meshgrid(xnew, ynew)

    # Here, we simply define bad points as being ones that are outside
    # the original range of the original data
    if checkLevels:
        lgtMax = znew > 2.*np.nanmax(z)
        if lgtMax.any():
            print "Replacing values greater then original with NaNs"
            znew[lgtMax] = np.nan
        lstMin = znew < np.nanmin(z)/2.
        if lstMin.any():
            print "Replacing values less then original with NaNs"
            znew[lstMin] = np.nan    

    return mx, my, znew

def zernikeFit(z):

    fitlist,C1 = opticspy.zernike.fitting(z,
                                          12,
                                          remain2D=1,
                                          barchart=1)
    print "fitlist: ", fitlist
    C1.listcoefficient()
    C1.zernikemap()
    # this works but takes forever!
    # C1.zernikesurface()

    return fitlist, C1

def getWeight(az, el, azLoc, elLoc, sigAz, sigEl, j, k):
    return 2*np.pi*np.exp((-(np.cos(elLoc[j,k])**2)*(az-azLoc[j,k])**2/(2.*sigAz**2)-(el-elLoc[j,k])**2/(2.*sigEl**2 )))
    # return 2*np.pi*np.exp((-(np.cos(elLoc[j,k])**2)*(az-azLoc[j,k])**2/(2.*((r/a)*sigAz)**2)-(el-elLoc[j,k])**2/(2.*sigEl**2 )))

def getWeightXYZ(x, y, xLoc, yLoc, sigX, sigY, j, k):
    #return 2*np.pi*np.exp((az-azLoc[j,k])**2/(2.*sigAz**2)-(el-elLoc[j,k])**2/(2.*sigEl**2 ))
    return 2*np.pi*np.exp((-(x-xLoc[j,k])**2 /( 2.*sigX**2 )-(y-yLoc[j,k])**2 /(2.*sigY**2 )))

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

def smoothXYZNew(x, y, z, n, sigX=None, sigY=None):

    if sigX is None:
        sigX=0.001
    if sigY is None:    
        sigY=0.001;

    xRange = np.linspace(min(x), max(x), n)
    yRange = np.linspace(min(y), max(y), n)

    xLoc, yLoc = np.meshgrid(xRange, yRange)


    zSm = np.ndarray(shape=(n,n))
    print "starting smoothing"
    for j in range(n):
        print "J:", j
        for k in range(n):
            w=2*np.pi*np.exp( (- (x - xLoc[j,k])**2 /( 2.*sigX**2 )-(y-yLoc[j,k])**2 /(2.*sigY**2 )))
            norm=sum(w)
            if norm==0:
                norm=1
                zSm[j,k]=np.nan #0 #min( r )
            else:
                w = w / norm
                zSm[j,k] = sum(z * w)

    return (xLoc, yLoc, zSm)    

def smoothSlow(az, el, r, n, sigEl=None, sigAz=None):
    "smooth our data"

    azRange = np.linspace(min(az), max(az), n)
    elRange = np.linspace(min(el), max(el), n)

    azLoc, elLoc = np.meshgrid(azRange, elRange)

    if sigEl is None:
        sigEl=0.001
    if sigAz is None:    
        sigAz=0.001;

    # init our smoothing result
    rSm = np.ndarray(shape=(n,n))
    rSms = []
    for j in range(n):
        # print "J:", j
        for k in range(n):
            #w=2*np.pi*np.exp( (- (az - azLoc[j,k])**2 /( 2.*sigAz**2 )-(el-elLoc[j,k])**2 /(2.*sigEl**2 )))
            #w=2*np.pi*np.exp((-(np.cos(elLoc[j,k])**2)*(az-azLoc[j,k])**2/(2.*sigAz**2)-(el-elLoc[j,k])**2/(2.*sigEl**2 )))
            w = getWeight(az, el, azLoc, elLoc, sigAz, sigEl, j, k)
            norm=sum(w)
            if norm==0:
                norm=1
                rSm[j,k]=np.nan #0 #min( r )
            else:
                w = w / norm
                rSm[j,k] = sum(r * w)

    return (azLoc, elLoc, rSm)    

def smooth(az, el, r, n, sigEl=None, sigAz=None):
    "smooth our data"

    # dAz=(max(az)-min(az))/(n-1)
    # dEl=(max(el)-min(el))/(n-1)
    # azRange=range(min(az), max(az), dAz)
    # elRange=range(min(el), max(el), dEl)
    azRange = np.linspace(min(az), max(az), n)
    elRange = np.linspace(min(el), max(el), n)

    azLoc, elLoc = np.meshgrid(azRange, elRange)

    if sigEl is None:
        sigEl=0.001
    if sigAz is None:    
        sigAz=0.001;

    # init our smoothing result
    rSm = np.ndarray(shape=(n,n))
    rSms = []
    for j in range(n):
        # print "J:", j
        for k in range(n):
            w=delayed(getWeight)(az, el, azLoc, elLoc, sigAz, sigEl, j, k)
            
            # w=delayed(getWeight)(copy(az),
            #                      copy(el), 
            #                      copy(azLoc),
            #                      copy(elLoc),
            #                      sigAz, sigEl, j, k)
            # w=2*np.pi*np.exp( (- (az - azLoc[j,k])**2 /( 2.*sigAz**2 )-(el-elLoc[j,k])**2 /(2.*sigEl**2 )))
            # norm=sum(w)
            # if norm==0:
            #     norm=1
            #     rSm[j,k]=np.nan #0 #min( r )
            # else:
            #     w = w / norm
            #     rSm[j,k] = sum(r * w)
            v = delayed(assignWeight)(w, copy(r))
            rSms.append((j, k, v))

    rSms = delayed(identity)(rSms)

    with ProgressBar(), Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler() as cprof:    
        rSms = rSms.compute(scheduler="processes")

    for j, k, v in rSms:
        rSm[j][k] = v

    return (azLoc, elLoc, rSm)

def daskWeight(n, az, el, r, azLoc, elLoc, sigAz, sigEl, j):
    rSms = []
    for k in range(n):
        w=getWeight(az, el, azLoc, elLoc, sigAz, sigEl, j, k)
        
        v = assignWeight(w, r)
        rSms.append((j, k, v))
    return rSms

def daskWeightXYZ(n, az, el, r, azLoc, elLoc, sigAz, sigEl, j):
    rSms = []
    for k in range(n):
        w=getWeightXYZ(az, el, azLoc, elLoc, sigAz, sigEl, j, k)
        
        v = assignWeight(w, r)
        rSms.append((j, k, v))
    return rSms

def smoothDask(az, el, r, n, sigEl=None, sigAz=None):
    "smooth our data with Dask"

    azRange = np.linspace(min(az), max(az), n)
    elRange = np.linspace(min(el), max(el), n)

    azLoc, elLoc = np.meshgrid(azRange, elRange)

    if sigEl is None:
        sigEl=0.001
    if sigAz is None:    
        sigAz=0.001;

    # init our smoothing result
    rSm = np.ndarray(shape=(n,n))
    rSms = []
    for j in range(n):
        rr =delayed(daskWeight)(n, az, el, r, azLoc, elLoc, sigAz, sigEl, j)
        # for rrr in rr:
            # rSms.append(rrr)
        rSms.append(rr)    
        # print "J:", j
        # for k in range(n):
            # w=delayed(getWeight)(az, el, azLoc, elLoc, sigAz, sigEl, j, k)
            
            # w=delayed(getWeight)(copy(az),
            #                      copy(el), 
            #                      copy(azLoc),
            #                      copy(elLoc),
            #                      sigAz, sigEl, j, k)
            # w=2*np.pi*np.exp( (- (az - azLoc[j,k])**2 /( 2.*sigAz**2 )-(el-elLoc[j,k])**2 /(2.*sigEl**2 )))
            # norm=sum(w)
            # if norm==0:
            #     norm=1
            #     rSm[j,k]=np.nan #0 #min( r )
            # else:
            #     w = w / norm
            #     rSm[j,k] = sum(r * w)
            # v = delayed(assignWeight)(w, copy(r))
            # rSms.append((j, k, v))

    rSms = delayed(identity)(rSms)

    with ProgressBar(), Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler() as cprof:    
        rSms = rSms.compute(scheduler="processes")

    for rSmss in rSms:
        for j, k, v in rSmss:
            rSm[j][k] = v

    return (azLoc, elLoc, rSm)

def smoothXYZ(x, y, z, n, sigX=None, sigY=None):
    "smooth our data"

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    # dAz=(max(az)-min(az))/(n-1)
    # dEl=(max(el)-min(el))/(n-1)
    # azRange=range(min(az), max(az), dAz)
    # elRange=range(min(el), max(el), dEl)
    # xNoNan = x[[not b for b in np.isnan(x)]]
    # yNoNan = y[[not b for b in np.isnan(y)]]
    # xRange = np.linspace(xNoNan.min(), xNoNan.max(), n)
    # yRange = np.linspace(yNoNan.min(), yNoNan.max(), n)
    x2 = x[[not b for b in np.isnan(x)]]
    y2 = y[[not b for b in np.isnan(y)]]
    z2 = z[[not b for b in np.isnan(z)]]

    x = x2
    y = y2
    z = z2

    xRange = np.linspace(x.min(), x.max(), n)
    yRange = np.linspace(y.min(), y.max(), n)

    xLoc, yLoc = np.meshgrid(xRange, yRange)

    if sigX is None:
        sigX=0.1
    if sigY is None:    
        sigY=0.1;

    # init our smoothing result
    zSm = np.ndarray(shape=(n,n))

    print "Starting smoothing"
    for j in range(n):
        print "J:", j
        for k in range(n):
            # rSms.append(rrr)
            w=2*np.pi*np.exp((-(x-xLoc[j,k])**2 /( 2.*sigX**2 )-(y-yLoc[j,k])**2 /(2.*sigY**2 )))
            #w=2*np.pi*np.exp((az-azLoc[j,k])**2/(  2.*sigAz**2)-(el-elLoc[j,k])**2/(2.*sigEl**2 ))
            norm=sum(w)
            if norm==0:
                norm=1
                zSm[j,k]=np.nan #0 #min( r )
            else:
                #import ipdb; ipdb.set_trace()
                w = w / norm
                v = sum(z * w)
                zSm[j,k] = v

    print "xLoc", xLoc
    print "yLoc", yLoc
    print "zSm", zSm
    return (xLoc, yLoc, zSm)

def loadLeicaDataFromGpus(fn):
    "Crudely loads x, y, z csv files into numpy arrays"

    xyzs = {}
    dims = ['x', 'y', 'z']
    for dim in dims:
        data = []
        fnn = "%s.%s.csv" % (fn, dim)
        with open(fnn, 'r') as f:
            ls = f.readlines()
        for l in ls:
            ll = l[:-1]
            if ll == 'nan':
                data.append(np.nan)
            else:
                data.append(float(ll))
        xyzs[dim] = np.array(data)
    return xyzs['x'], xyzs['y'], xyzs['z']

# def smoothXYZGpu(x, y, z, n, sigX=None, sigY=None, filename=None):
def smoothXYZGpu(x, y, z, n, sigXY=None, filename=None):
    "use GPU code to do the simple XYZ smoothing"

    if sigXY is None:
        sigXY = 0.1


    # first get data into file format expected by GPU code:
    # x, y, z per line
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    x = x[[not b for b in np.isnan(x)]]
    y = y[[not b for b in np.isnan(y)]]
    z = z[[not b for b in np.isnan(z)]]

    assert len(x) == len(y)
    assert len(y) == len(z)

    # TBF: how to zip 3 together?
    xyz = []
    for i in range(len(x)):
        xyz.append((x[i], y[i], z[i]))
    xyz = np.array(xyz)

    # where's our input data?
    abspath = os.path.abspath(os.path.curdir)
    if filename is None:
        fn = "test"
    else:
        fn = filename
            
    inFile = os.path.join(abspath, "data", fn)

    np.savetxt(inFile, xyz, delimiter=",") 

    # call the GPU code
    # where is the code we'll be running?
    # gpuPath = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing"
    gpuPath = GPU_PATH
    outFile = fn
    smoothGPUs(gpuPath, inFile, outFile, n, noCos=True, sigAzEl=sigXY)

    # make sure the output is where it should be
    for dim in ['x', 'y', 'z']:
        dimFile = "%s.%s.csv" % (outFile, dim)
        dimPath = os.path.join(gpuPath, dimFile)
        assert os.path.isfile(dimPath)

    # extract the results from the resultant files
    outPath = os.path.join(gpuPath, outFile)
    return loadLeicaDataFromGpus(outPath)

    
def smoothXYZDask(x, y, z, n, sigX=None, sigY=None):
    "smooth our data"

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    x2 = x[[not b for b in np.isnan(x)]]
    y2 = y[[not b for b in np.isnan(y)]]
    z2 = z[[not b for b in np.isnan(z)]]

    x = x2
    y = y2
    z = z2

    xRange = np.linspace(x.min(), x.max(), n)
    yRange = np.linspace(y.min(), y.max(), n)

    xLoc, yLoc = np.meshgrid(xRange, yRange)

    if sigX is None:
        sigX=0.1
    if sigY is None:    
        sigY=0.1;

    # init our smoothing result
    zSm = np.ndarray(shape=(n,n))
    zSms = []
    print "Starting smoothing"
    for j in range(n):
        #print "J:", j
        rr =delayed(daskWeightXYZ)(n, x, y, z, xLoc, yLoc, sigX, sigY, j)
        # for rrr in rr:
            # rSms.append(rrr)
        zSms.append(rr)    
        #for k in range(n):
        #    w=2*np.pi*np.exp( (- (x - xLoc[j,k])**2 /( 2.*sigX**2 )-(y-yLoc[j,k])**2 /(2.*sigY**2 )))
        #    norm=sum(w)
        #    if norm==0:
        #        norm=1
        #        zSm[j,k]=np.nan #0 #min( r )
        #    else:
        #        w = w / norm
        #        zSm[j,k] = sum(z * w)

    zSms = delayed(identity)(zSms)

    with ProgressBar(), Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler() as cprof:    
        zSms = zSms.compute(scheduler="processes")

    for zSmss in zSms:
        for j, k, v in zSmss:
            zSm[j][k] = v

    return (xLoc, yLoc, zSm)

def unwrap(azs):
    "make sure all az values are between 0 and 2pi"

    # must be a more efficient way of doing this
    for i, az in enumerate(azs):
        if az < 0:
            azs[i] = az + (2*np.pi)

    return azs
            
def cart2sph(x, y, z):
    "Wrapper around cartesian_to_spherical"
    rs, lats, lngs = cartesian_to_spherical(x, y, z)

    # convert from astropy Quantities to simple numpy
    rs = np.array([r.value for r in rs])
    lats = np.array([l.value for l in lats])
    lngs = np.array([l.value for l in lngs])

    print "min/max lats (radians)", lats.min(), lats.max()
    print "lngs range (radians)", -np.pi/2, np.pi/2
    print "min/max lngs (radians)", lngs.min(), lngs.max()
    print "lats range (radians)", 0, 2*np.pi

    return rs, lats, lngs

def sph2cart(az, el, r):

    print "min/max az (radians)", az.min(), az.max()
    print "el range (radians)", -np.pi/2, np.pi/2
    print "min/max el (radians)", el.min(), el.max()
    print "az range (radians)", 0, 2*np.pi


    xs, ys, zs = spherical_to_cartesian(r, az, el)

    # convert from astropy Quantities to simple numpy
    xs = np.array([x.value for x in xs])
    ys = np.array([y.value for y in ys])
    zs = np.array([z.value for z in zs])

    return xs, ys, zs 


def importCsv(filename):
    "Import x,y,z values from CSV file"


    fieldnames = ['x', 'y', 'z']

    xs = []
    ys = []
    zs = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)

        for row in reader:
            xs.append(float(row['x']))
            ys.append(float(row['y']))
            zs.append(float(row['z']))


    return np.array(xs), np.array(ys), np.array(zs)

def processDiffSimple(fn1, fn2):
    "Find diff between two evenly-spaced surfaces and fit"

    r = np.load(fn1)
    x1 = r['x']
    y1 = r['y']
    z1 = r['z']

    r = np.load(fn2)
    x2 = r['x']
    y2 = r['y']
    z2 = r['z']

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x1, y1, z1)
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x2, y2, z2)
    plt.show()

    # Here we are assuming the x, y dims are the same
    zDiff = z1 - z2

    print zDiff

    print "plotting surface diff ..."
    plt.imshow(zDiff, interpolation="nearest", origin="upper")
    plt.colorbar()
    plt.show()

    print "Zernikie's for surface diffs:"
    fits, c = zernikeFit(zDiff)

def testTransforms():

    x = y = z = [1.]
    identityTest(x, y, z)

    x = [100]
    y = [-200]
    z = [50]
    identityTest(x, y, z)

def identityTest(x, y, z):

    r, lat, lng = cartesian_to_spherical(x, y, z)

    print r, lat, lng

    xs, ys, zs = spherical_to_cartesian(r, lat, lng)

    print xs, ys, zs

def multi(i, j):
    return i*j

def assign(k):
    return k if k <= 2 else 0

def identity(k):
    return k

def testDask():

    n = 3
    ks = np.ndarray((n, n))
    kss = []
    for i in range(n):
        for j in range(n):
            k = delayed(multi)(i, j)
            v = delayed(assign)(k)
            # ks[i][j] = v
            # v = k if k <= 2 else 0
            # ks[i][j] = v
            kss.append((i, j, v))
    kss = delayed(identity)(kss)
    print kss

    kss = kss.compute(scheduler='processes')
            
    print kss

    for i, j, v in kss:
        ks[i][j] = v

    print ks

def plotXYZ(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z)
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.show()

def testInterp():

    fn = "smoothXYZ10.npz"
    r = np.load(fn)
    x = r['x']
    y = r['y']
    z = r['z']

    # plotXYZ(x, y, z)


    # look at inner part to avoid zeros and nans    
    n = z.shape[0]
    start = n / 3
    end = (2*n) / 3

    x = x[start:end, start:end]
    y = y[start:end, start:end]
    z = z[start:end, start:end]

    plotXYZ(x, y, z)

    # now take a rectangular slice of THAT
    xw = (x.max() - x.min())
    xcntr = x.min() + (xw/2)
    yw = (y.max() - y.min())
    ycntr = y.min() + (yw/2)


    xmask = np.logical_and(x < xcntr + (xw/4), x > xcntr - (xw/4))

    x = x[xmask]
    y = y[xmask]
    z = z[xmask]

    print x
    print y
    print z

    plotXYZ(x, y, z)

    f = interpolate.interp2d(x.flatten(),
                             y.flatten(),
                             z.flatten(),
                             kind='cubic')

    xnew = np.linspace(x.min(), x.max(), n)
    ynew = np.linspace(y.min(), y.max(), n)

    znew = f(xnew, ynew)       

    mx, my = np.meshgrid(xnew, ynew)

    plotXYZ(mx, my, znew)

def addCenterBump(x, y, z, rScale=10, zScale=0.25):

    # add a bump to the center bit
    xMin = np.nanmin(x)
    xMax = np.nanmax(x)
    yMin = np.nanmin(y)
    yMax = np.nanmax(y)

    # d = 10
    d = rScale

    xW = xMax - xMin
    xStep = xW / d
    xStart = xMin + (d/2 - 1)*xStep
    xEnd = xMax - (d/2 -1)*xStep
    assert xStart < xEnd

    yW = yMax - yMin
    yStep = yW / d
    yStart = yMin + (d/2 - 1)*yStep
    yEnd = yMax - (d/2 -1)*yStep
    assert yStart < yEnd

    cnt = 0
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        if xi > xStart and xi < xEnd and yi > yStart and yi < yEnd:
            cnt +=1 
            z[i] = zi + (zScale*zi)
    print "Added bump to %d pnts" % cnt

    return z

def smoothSpherical(fn, n, sigAz=None, sigEl=None, addBump=False):

    # 10: seconds
    # 50: 2.5 mins
    # 100: 8 mins
    # n = 20

    print "importing CSV data ..."
    x, y, z = importCsv(fn)

    if addBump:
        # add a bump to the center bit
        xMin = np.nanmin(x)
        xMax = np.nanmax(x)
        yMin = np.nanmin(y)
        yMax = np.nanmax(y)

        d = 10

        xW = xMax - xMin
        xStep = xW / d
        xStart = xMin + (d/2 - 1)*xStep
        xEnd = xMax - (d/2 -1)*xStep
        assert xStart < xEnd

        yW = yMax - yMin
        yStep = yW / d
        yStart = yMin + (d/2 - 1)*yStep
        yEnd = yMax - (d/2 -1)*yStep
        assert yStart < yEnd

        cnt = 0
        for i in range(len(x)):
            xi = x[i]
            yi = y[i]
            zi = z[i]
            if xi > xStart and xi < xEnd and yi > yStart and yi < yEnd:
                cnt +=1 
                z[i] = zi + (0.25*zi)
        print "Added bump to %d pnts" % cnt

    print "Converting to spherical coords ..."
    r, az, el = cart2sph(x, y, z)



    print "smoothing data ..."
    azLoc, elLoc, rSmooth = smoothDask(az, el, r, n, sigAz=sigAz, sigEl=sigEl)


    # back to x, y, z
    print "converting back to cartesian ..."
    xs, ys, zs = sph2cart(azLoc, elLoc, rSmooth)

    fns = fn + ".smoothed"
    np.savez(fns, x=xs, y=ys, z=zs)

    return xs, ys, zs

def processDiff(fn1, fn2):
    """
    Taking in two raw Leica data files, smooth both in 
    the same spherical space, diff the two cartesians
    """
    n = 30
    x1, y1, z1 = smoothSpherical(fn1, n)
    x2, y2, z2 = smoothSpherical(fn2, n)


    # fn2 = "data/randomSampleScan10.csv" has some bad data.
    z2org = copy(z2)
    z2[z2 < -65] = np.nan

    # interpolate both surfaces using the same xyz grid
    x1max = np.nanmax(x1)
    x1min = np.nanmin(x1)
    x2max = np.nanmax(x2)
    x2min = np.nanmin(x2)
    xrange = (min(x1min, x2min), max(x1max, x2max))
    y1max = np.nanmax(y1)
    y1min = np.nanmin(y1)
    y2max = np.nanmax(y2)
    y2min = np.nanmin(y2)
    yrange = (min(y1min, y2min), max(y1max, y2max))

    sx1, sy1, sz1 = interpXYZ(x1, y1, z1, n, xrange=xrange, yrange=yrange)
    sx2, sy2, sz2 = interpXYZ(x2, y2, z2, n, xrange=xrange, yrange=yrange)

    plotXYZ(sx1, sy1, sz1)
    plotXYZ(sx2, sy2, sz2)

    assert sx1 == sx2
    assert sy1 == sy2

    zdiff = sz1 - sz2

    plotXYZ(sx2, sy2, zdiff)

    zernikeFit(zdiff)

def findTheDamnBumps():

    fn1 = "data/randomSampleBumpScan14pnts2m.csv"
    #fn2 = "data/randomSampleBumpScan14pnts6m.csv"
    
    n = 30
    x1, y1, z1 = smoothSpherical(fn1, n, sigAz=1.0, sigEl=1.0)    
    x2, y2, z2 = smoothSpherical(fn1, n)    


def testSmoothXYZ():

    x = np.linspace(-20, 20)
    y = np.linspace(-20, 20)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros(xx.shape)
    xs, ys, zs = smoothXYZDask(xx, yy, zz, xx.shape[0], sigX=1, sigY=1)
    print zs

def smoothGPUs(gpuPath,
               inFile,
               outFile,
               n,
               verbose=False,
               noCos=False,
               spherical=False,
               sigAzEl=None):
    "Ssh's to RH6 machine to run gpu code"

    # catch mutually exclusive options
    if noCos and spherical:
        raise "noCos and spherical are mutally exclusive options"

    # the sigAz and sigEl will always be identical
    if sigAzEl is None:
        sigAzEl = 0.001

    cmd = "runGpuSmooth %s %s %s %d %1.5f" % (gpuPath, inFile, outFile, n, sigAzEl)
    # if sigAz is not None:
    #     cmd += " --sigAz %1.5f " % sigAz
    # if sigEl is not None:
    #     cmd += " --sigEl %1.5f " % sigEl
    if noCos:
        cmd += " --no-cos"

    # spherical option means whether GPUs will be
    # doing spherical coord transform or not
    if spherical:
        cmd += " --no-conv"

    print "system cmd: ", cmd
    os.system(cmd)

# def smoothProcessedFile(fpath, N=512, squared=False):
#     "Smooths processed file contents via GPUs, optionally squaring"

#     assert os.path.isfile(fpath)

#     # if squared we need to first square this data, then
#     # write it to a new file for processing
#     if squared:
#         # open the file: wich was written with np.savetxt
#         xyz = np.loadtxt(fpath, delimiter=",")
#         x, y, z = splitXYZ(xyz)

#         print "Converting to spherical coords ..."
#         r, az, el = cart2sph(x, y, z)

#         # now square the data!
#         r2 = r**2

#         # now write it back to file
#         sph = aggregateXYZ(r, az, el)
#         np.savetxt(sph, delimiter=",")
#     # we need to specify the aboslute path of our inputs
#     abspath = os.path.abspath(fpath)

#     # our output will be same file name, but with appendages
#     outfile = os.path.basename(fpath)

#     smoothGPUs(GPU_PATH, abspath, outfile, N)

#     # make sure the output is where it should be
#     outfiles = []
#     for dim in ['x', 'y', 'z']:
#         dimFile = "%s.%s.csv" % (outfile, dim)
#         dimPath = os.path.join(GPU_PATH, dimFile)
#         outfiles.append(dimPath)
#         assert os.path.isfile(dimPath)

#     return outfiles

# def getWeightsFromInitialSmoothing(smoothedFileBase, processedPath):
#     """
#     Calculated weights via: 
#        * w = 1/sigma**2 / sum(1/sigma**2)
#        * sigma**2 = <z**2> - <z>**2
#        * <> == smoothing
#     """

#     # we need to reload the processed data so we can
#     # square it then smooth it (<z**2>)
#     smoothSquaredProcessedFile(fpath)

def testSmoothGPUs():

    # gpuPath = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing"
    gpuPath = GPU_PATH
    abspath = os.path.abspath(os.path.curdir)
    inFile = "Test1_STA14_Bump1_High-02_METERS.ptx.csv"
    fpath1 = os.path.join(abspath, "data", inFile)
    n = 100
    assert os.path.isfile(fpath1)
    #outFile1 = os.path.basename(fpath1)
    outFile1 = inFile
    smoothGPUs(gpuPath, fpath1, outFile1, n)

    xOutfile = os.path.join(gpuPath, inFile + ".x.csv")
    assert os.path.isfile(xOutfile)

def main():
    testSmoothGPUs()
    return

    #testSmoothXYZ()

    #fn1 = "data/randomSampleSta10.csv"
    fn = "data/randomSampleScan10.csv"
    n = 30
    x1, y1, z1 = smoothSpherical(fn, n)    
    np.savetxt("x.smoothed.csv", x1, delimiter=",")
    np.savetxt("y.smoothed.csv", y1, delimiter=",")
    np.savetxt("z.smoothed.csv", z1, delimiter=",")
    # testInterp()
    # testDask()
    # n = 50
    # fn = "data/randomSampleSta10.csv"
    # fn2 = "data/randomSampleScan10.csv"
    # processDiff(fn, fn2)
    # print "processing img1"
    # process(fn, n, "img1")
    # print ""
    # print "processing img2"
    # process(fn, n, "img2", noise=True)
    # processDiff("img1.npz", "img2.npz")

if __name__ == "__main__":
    main()
    # testTransforms()
