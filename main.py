
import csv
from copy import copy

import numpy as np
from astropy.coordinates import cartesian_to_spherical
from astropy.coordinates import spherical_to_cartesian
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d, Axes3D
import opticspy
from scipy import interpolate

from dask import delayed
from dask import config
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler
from dask.diagnostics import visualize

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

    f = interpolate.interp2d(x3, y3, z3, kind='linear')
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
        lgtMax = znew > np.nanmax(z)
        if lgtMax.any():
            print "Replacing values greater then original with NaNs"
            znew[lgtMax] = np.nan
        lstMin = znew < np.nanmin(z)
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
            w=2*np.pi*np.exp( (- (az - azLoc[j,k])**2 /( 2.*sigAz**2 )-(el-elLoc[j,k])**2 /(2.*sigEl**2 )))
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
    z2 = y[[not b for b in np.isnan(z)]]

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

    for j in range(n):
        # print "J:", j
        for k in range(n):
            w=2*np.pi*np.exp( (- (x - xLoc[j,k])**2 /( 2.*sigX**2 )-(y-yLoc[j,k])**2 /(2.*sigY**2 )))
            norm=sum(w)
            if norm==0:
                norm=1
                zSm[j,k]=np.nan #0 #min( r )
            else:
                w = w / norm
                zSm[j,k] = sum(z * w)

    print "xLoc", xLoc
    print "yLoc", yLoc
    print "zSm", zSm
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


def main():
    fn1 = "data/randomSampleSta10.csv"
    # fn2 = "data/randomSampleScan10.csv"
    n = 30
    x1, y1, z1 = smoothSpherical(fn1, n)    
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
