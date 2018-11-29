
import csv

import numpy as np
from astropy.coordinates import cartesian_to_spherical
from astropy.coordinates import spherical_to_cartesian
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import opticspy

def process(fn, n, outputName):

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

    # print "unwrapping az values ..."
    # az = unwrap(az)

    print "smoothing data ..."
    azLoc, elLoc, rSmooth = smooth(az, el, r, n)

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

    print "plotting ..."
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xs, ys, zs)
    plt.show()

    xss, yss, zss = smooth(x, y, z, n, sigEl=0.1, sigAz=0.1)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xss, yss, zss)
    plt.show()

    
    # find the zernike fit to this!
    # first convert NaN's to zeros
    zss[np.isnan(zss)] = 0.

    zernikeFit(zss)

    # save processing results
    np.savez(outputName, x=xss, y=yss, z=zss)

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

    for j in range(n):
        print "J:", j
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

def main():
    n = 10
    fn = "data/randomSampleSta10.csv"
    process(fn, n, "img1")
    # process(fn, n, "img2")
    # processDiff("img1", "img2")

if __name__ == "__main__":
    main()
    # testTransforms()
