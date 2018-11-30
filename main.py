
import csv

import numpy as np
from astropy.coordinates import cartesian_to_spherical
from astropy.coordinates import spherical_to_cartesian
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import opticspy

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
    xss, yss, zss = smoothXYZ(xs, ys, zs, n, sigX=0.5, sigY=0.5)

    print "plotting final smoothed x y z"
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xss, yss, zss)
    plt.show()

    print "plotting final smoothed x y z as scatter"
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xss, yss, zss)
    plt.show()

    # find the zernike fit to this!
    # first convert NaN's to zeros
    zss[np.isnan(zs)] = 0.
    # zs[np.isnan(zs)] = 0.

    zernikeFit(zss)

    # save processing results
    # np.savez(outputName, x=xss, y=yss, z=zss)
    # np.savez(outputName, x=xs, y=ys, z=zs)

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
    return 2*np.pi*np.exp((-(az-azLoc[j,k])**2/(2.*sigAz**2)-(el-elLoc[j,k])**2/(2.*sigEl**2 )))

def assignWeight(w, r):

    norm=sum(w)
    if norm==0:
        norm=1
        v=np.nan #0 #min( r )
    else:
        w = w / norm
        v = sum(r * w)   
    return v

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
        print "J:", j
        for k in range(n):
            w=delayed(getWeight)(az, el, azLoc, elLoc, sigAz, sigEl, j, k)
            # w=2*np.pi*np.exp( (- (az - azLoc[j,k])**2 /( 2.*sigAz**2 )-(el-elLoc[j,k])**2 /(2.*sigEl**2 )))
            # norm=sum(w)
            # if norm==0:
            #     norm=1
            #     rSm[j,k]=np.nan #0 #min( r )
            # else:
            #     w = w / norm
            #     rSm[j,k] = sum(r * w)
            v = delayed(assignWeight)(w, r)
            rSms.append((j, k, v))

    rSms = delayed(identity)(rSms)

    with ProgressBar(), Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler() as cprof:    
        rSms = rSms.compute(scheduler="processes")

    for j, k, v in rSms:
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
        sigX=0.001
    if sigY is None:    
        sigY=0.001;

    # init our smoothing result
    zSm = np.ndarray(shape=(n,n))

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

def processDiff(fn1, fn2):
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

def main():
    # testDask()
    n = 30
    fn = "data/randomSampleSta10.csv"
    print "processing img1"
    process(fn, n, "img1")
    # print ""
    # print "processing img2"
    # process(fn, n, "img2", noise=True)
    # processDiff("img1.npz", "img2.npz")

if __name__ == "__main__":
    main()
    # testTransforms()
