from copy import copy
import random
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import leastsq
# Do this if you run into the dreaded Tkinter import error
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from utils.utils import sph2cart, cart2sph

from rotate import *

from SmoothedFITS import SmoothedFITS

def parabola(xdata, ydata, focus, v1x, v1y, v2, heavy=False):
    if heavy:
        return parabolaHeavy(xdata, ydata, focus, v1x, v1y, v2)
    else:
        return parabolaSimple(xdata, ydata, focus, v1x, v1y, v2)

def parabolaSimple(xdata, ydata, focus, v1x, v1y, v2):
    "simply the equation for a 2-D parabola"
    focus = 60.
    return (1 / (4.*focus))*(xdata - v1x)**2 + (1 / (4.*focus))*(ydata - v1y)**2 + v2

def parabolaSimple_(xdata, ydata, focus):
    "simply the equation for a 2-D parabola"
    return (1./(4.*focus))*(xdata)**2 + (1./(4.*focus))*(ydata)**2

def parabolaHeavy(xdata, ydata, focus, v1x, v1y, v2): #, hr=104.):
    "Reject everything outside the circle our data shows up in"
    #return (1 / (4.*focus))*(xdata - v1x)**2 + (1 / (4.*focus))*(ydata - v1y)**2 + v2
    #r2 = (1 / (4.*focus))*(xdata - v1x)**2 + (1 / (4.*focus))*(ydata - v1y)**2 
    #return (r2 * (hr**2 > r2)) + v2

    # caluclate the parabola normally
    z = parabolaSimple(xdata, ydata, focus, v1x, v1y, v2)

    # now return only those values within our cynlinder
    xoffset = 0. #54.
    yoffset = 54. #0.
    radius = 50.
    mask = heavySide(xdata, ydata, v1x, v1y, xoffset, yoffset, radius)
    print("Of %d elements, %d masked by heavyside" % (len(z), np.sum(mask)))
    #assert np.all(mask)
    #return z[mask]
    # z[mask] = 0.0
    return z
    
def heavySide(x, y, v1x, v1y, xoffset, yoffset, radius):
    return ((x - v1x - xoffset)**2 + (y - v1y - yoffset)**2) > radius**2    

def rotate(x, y, z, aroundXrads, aroundYrads):
    # first, around x
    # x = x
    yRot = y*np.cos(aroundXrads) - z*np.sin(aroundXrads)
    zRot = y*np.sin(aroundXrads) + z*np.cos(aroundXrads)
    
    # then around y
    xRot = x*np.cos(aroundYrads) + zRot*np.sin(aroundYrads);
    # y = y;
    zRot = zRot*np.cos(aroundYrads) - x*np.sin(aroundYrads);
    return xRot, yRot, zRot

def parabolaRot(xdata, ydata, focus, v1x, v1y, v2, rotX, rotY):
    zdata = parabola(xdata, ydata, focus, v1x, v1y, v2)
    #zdata = np.zeros(xdata.shape)
    #_, _, zdataRot = rotate(xdata, ydata, zdata, rotX, rotY)

    L = np.array([xdata.flatten(), ydata.flatten(), zdata.flatten()])
    pry = (rotX, rotY, 0.)
    xr = PrimeX(pry, L)
    yr = PrimeY(pry, L)
    zr = PrimeZ(pry, L)
    return xr, yr, zr

def rotateData(xdata, ydata, zdata, rotX, rotY):
    L = np.array([xdata.flatten(), ydata.flatten(), zdata.flatten()])
    pry = (rotX, rotY, 0.)
    xr = PrimeX(pry, L)
    yr = PrimeY(pry, L)
    zr = PrimeZ(pry, L)
    return xr, yr, zr

def fitParabolaNew(L, f, v1x, v1y, v2, xTheta, yTheta):
    coeffs = [f, v1x, v1y, v2, xTheta, yTheta]
    return fitParabola(coeffs, L[0], L[1], L[2])


def fitParabolaWithWeights(coeffs, x, y, z, weights):
    
    z = fitParabola(coeffs, x, y, z)
    return z*weights

def fitParabola(coeffs, x, y, z):
    
    
    L = np.array([x.flatten(), y.flatten(), z.flatten()])
    pry = (coeffs[4], coeffs[5], 0.)
    xr = PrimeX(pry, L)
    yr = PrimeY(pry, L)
    zr = PrimeZ(pry, L)
    
    zdata = parabola(xr, yr, coeffs[0], coeffs[1], coeffs[2], coeffs[3])

    return zr - zdata

def newParabola(xdata, ydata, zdata, focus, v1x, v1y, v2, rotX, rotY):

    L = np.array([xdata.flatten(), ydata.flatten(), zdata.flatten()])
    pry = (rotX, rotY, 0.)
    xr = PrimeX(pry, L)
    yr = PrimeY(pry, L)
    #zr = PrimeZ(pry, L)
    
    new_zdata = parabola(xr, yr, focus, v1x, v1y, v2)

    return xr, yr, new_zdata

def fun(coeffs, xdata, ydata):
    focus = coeffs[0]
    v1x = coeffs[1]
    v1y = coeffs[2]
    v2 = coeffs[3]
    return parabola(xdata, ydata, focus, v1x, v1y, v2)

def errfun(coeffs, xdata, ydata, zdata):
    return fun(coeffs, xdata, ydata) - zdata

def findTheBumps():

    fn = "data/BumpScan.csv.smoothed.sig.001.all.npz"
    bumpScan = fitLeicaScan(fn)
    fn = "data/Baseline_STA10_HIGH_METERS.csv.smoothed.sig.001.all.npz"
    refScan = fitLeicaScan(fn)

    bumps = bumpScan - refScan

    bumpDiff = np.log(np.abs(bumps))
    imagePlot(bumpDiff, "Bumps!")

    return bumps

def fitLeicaScan(fn,
                 numpy=True,
                 N=None,
                 rFilter=False,
                 xyz=None,
                 inSpherical=False,
                 weights=None,
                 plot=True):

    # TBF: ignore weights for now!

    if N is None:
        N = 512

    # load the data: it might be two different formats, depending
    # on whether it was done in python (numpy) or with GPUs
    if xyz is None:
        orgData, cleanData = loadLeicaData(fn, n=N, numpy=numpy)
        x0, y0, z0 = orgData
        x, y, z = cleanData
        if inSpherical:
            # we need to convert this data to cartesian:
            # x0, y0, z0 == r, az, el
            print("converting inputs from spherical to cartesian")
            x0, y0, z0 = sph2cart(y0, z0, x0)
            x, y, z = sph2cart(y, z, x)

    else:
        x0, y0, z0 = xyz
        print("passed in data of shape", x0.shape, y0.shape, z0.shape)
        x = x0[np.logical_not(np.isnan(x0))];
        y = y0[np.logical_not(np.isnan(y0))];
        z = z0[np.logical_not(np.isnan(z0))];


    if plot:
        scatter3dPlot(x, y, z, "Sample of Leica Data", sample=10.0)

    print("x range:", np.min(x), np.max(x))
    print("y range:", np.min(y), np.max(y))


    if np.all(np.isnan(x)) or np.all(np.isnan(y)) or np.all(np.isnan(z)):
        print("fitLeicaScan cannot work on data with all NANs")
        return None, None, None

    if rFilter:
        xOffset = -8.0
        yOffset = 50.0
        radius = 47.
        x, y, z = radialFilter(x, y, z, xOffset, yOffset, radius)
        if plot:
            scatter3dPlot(x, y, z, "Leica Data Radial Filtered")

    # assemble initial guess
    f = 60.
    v1x = v1y = v2 = 0
    xTheta = 0. #-np.pi / 2.
    yTheta = 0.
    guess = [f, v1x, v1y, v2, 0., 0.]

    if weights is not None:
        print("Weights: ", weights.shape, weights)
        print("z0: ", z0.shape, z0)
        # asure correct dimensions
        z0.shape = (N, N)
        weights.shape = (N, N)
        print("filtering out nans in xyz data from weights")
        # this will flatten the weights
        weights = weights[np.logical_not(np.isnan(z0))];
        print("replacing nans in weights by zeros")
        weights[np.isnan(weights)] = 0.

    print("fitLeicaData xyz.shape: ", x.shape, y.shape, z.shape)

    r = fitLeicaData(x, y, z, guess, weights=weights, verbose=True)

    # plot fitted data
    c = r.x
    print("cleaned data fitted with coefficients: ", c)
    newX, newY, newZ = newParabola(x0, y0, z0, c[0], c[1], c[2], c[3], c[4], c[5])
    newX.shape = newY.shape = newZ.shape = (N, N)
    if plot:
        surface3dPlot(newX, newY, newZ, "Fitted Data")

    # rotate original data using fitted coefficients
    xThetaFit = r.x[4]
    yThetaFit = r.x[5]
    xrr, yrr, zrr = rotateData(x0, y0, z0, xThetaFit, yThetaFit)
    # also apply the translations
    #xrr -= c[1]
    #yrr -= c[2]
    #zrr -= c[3]
    xrr.shape = yrr.shape = zrr.shape = (N, N)
    if plot:
        surface3dPlot(xrr, yrr, zrr, "Original Data (Rotated)")

    # return difference between fitted data and rotated original data
    diff = zrr - newZ
    diff2 = np.log(np.abs(np.diff(diff)))
    if plot:
        imagePlot(diff2, "Fit - Org (Rotated)")

    return diff, newX, newY

def imagePlot(z, title):
    fig = plt.figure()
    ax = fig.gca()
    #cax = ax.imshow(np.log(np.abs(np.diff(zrr - newZ))))
    cax = ax.imshow(z)
    fig.colorbar(cax)
    plt.title(title)

def fitLeicaData(x, y, z, guess, bounds=None, weights=None, method=None, max_nfev=10000, ftol=1e-12, xtol=1e-12, verbose=False):

    if verbose:
        print("fit leica boss!")
    
    # Set boundaries for the fit parameters.
    if not bounds:
        inf = np.inf
        pi2 = 2*np.pi
        b1 = [0., -inf, -inf, -inf, -pi2, -pi2]
        b2 = [inf, inf,  inf,  inf,  pi2,  pi2]
        bounds = (b1, b2)

    # robust fit: weights outliers outside of f_scale less
    loss = "soft_l1"
    # 0.05 is a good educated guess from Andrew
    # f_scale = .05
    # f_scale = .01
    f_scale = 1.0
    if verbose:
        print("fitLeicaData with robust, soft_l1, f_scale %f" % f_scale)

    if method is None:
        if weights is None:
            if verbose:
                print ("Using method fitParabola")
            method = fitParabola
            args = (x .flatten(), y.flatten(), z.flatten())
        else:
            if verbose:
                print("Using weights for fit!")
            method = fitParabolaWithWeights
            args = (x .flatten(), y.flatten(), z.flatten(), weights.flatten())
    else:
        if verbose:
            print("Using user supplied method.")
        args = (x .flatten(), y.flatten(), z.flatten())

    r = least_squares(method,
                      guess,
                      args=args,
                      bounds=bounds,
                      max_nfev=max_nfev,
                      loss=loss,
                      f_scale=f_scale,
                      ftol=ftol,
                      xtol=xtol)
    return r

def loadLeicaDataFromNumpy(fn):

    data = np.load(fn)

    x = data['x']
    y = data['y']
    z = data['z']

    xx = x.flatten()
    yy = y.flatten()
    zz = z.flatten()
    
    return (x, y, z), (xx, yy, zz)

def loadSmoothedFits(fn):
    f = SmoothedFITS()
    f.read(fn)
    return f.x, f.y, f.z

    #return xyzs['x'], xyzs['y'], xyzs['z']

def loadLeicaDataFromGpus(fn):
    "Crudely loads x, y, z csv files into numpy arrays"

    # kluge:
    # if this is a fits file, then 
    # load from that
    _, ext = os.path.splitext(fn)
    if ext == ".fits":
        print("processing fits file")
        return loadSmoothedFits(fn)

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

def loadLeicaData(fn, n=None, numpy=True):

    if numpy:
        orgData, flatData = loadLeicaDataFromNumpy(fn)
        x, y, z = orgData
        xx, yy, zz = flatData
    else:
        xx, yy, zz = loadLeicaDataFromGpus(fn)
        x = copy(xx)
        y = copy(yy)
        z = copy(zz)
        if n is not None:
            x.shape = y.shape = z.shape = (n, n)

    xxn = xx[np.logical_not(np.isnan(xx))];
    yyn = yy[np.logical_not(np.isnan(yy))];
    zzn = zz[np.logical_not(np.isnan(zz))];

    return (x, y, z), (xxn, yyn, zzn)

def sampleXYZData(x, y, z, samplePercentage):
    "Return a random percentage of the data"

    assert len(x) == len(y)
    assert len(y) == len(z)

    lenx = len(x)

    sampleSize = int((lenx * samplePercentage) / 100.)

    idx = random.sample(range(lenx), sampleSize)

    return copy(x[idx]), copy(y[idx]), copy(z[idx])

def radialReplace(x, y, z, xOffset, yOffset, radius, replacement):
    "Replace z values outside of given radius with a new value"

    cnt = 0
    for i in range(len(x)):
        r = np.sqrt((x[i]-xOffset)**2 + (y[i]-yOffset)**2)
        if r > radius:
            cnt += 1
            z[i] = replacement

    print("radialReplace replaced %d points with %s" % (cnt, replacement))
    # only z gets changed
    return z

def surface3dPlot(x, y, z, title, xlim=None, ylim=None, sample=None):
    fig = plt.figure()
    ax = Axes3D(fig)

    # plot all the data, or just some?
    if sample is not None:
        print("Plotting %5.2f percent of data" % sample)
        x, y, z = sampleXYZData(x, y, z, sample)

    ax.plot_surface(x, y, z)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

def scatter3dPlot(x, y, z, title, xlim=None, ylim=None, sample=None):

    # plot all the data, or just some?
    if sample is not None:
        print("Plotting %5.2f percent of data" % sample)
        x, y, z = sampleXYZData(x, y, z, sample)
        print("Now length of data is %d" % len(x))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

def fitNoRot():

    n = 20
    n2 = 50
    x = np.linspace(-n, n)
    y = np.linspace(-n, n)

    x2d, y2d = np.meshgrid(x, y)

    focus = 6.
    v1x = 5.
    v1y = 4.
    v2 = 3.
    fakedata = parabola(x2d, y2d, focus, v1x, v1y, v2) + np.random.rand(n2, n2)

    #coeffs = [focus, v1x, v1y, v2]
    coeffs = [1., 0., 0., 0.]
    r = least_squares(errfun, coeffs, args=(x2d.flatten(), y2d.flatten(), fakedata.flatten()))
    print(r)

def fun2(coeffs, xdata, ydata):
    focus = coeffs[0]
    v1x = coeffs[1]
    v1y = coeffs[2]
    v2 = coeffs[3]
    aroundX = coeffs[4]
    aroundY = coeffs[5]
    # rotate the data
    # then call the parabola 
    #xrot, yrot, z = parabolaRot(xdata, ydata, focus, v1x, v1y, v2, aroundX, aroundY)
    return z

def errfun2(coeffs, xdata, ydata, zdata):
    # rotate zdatar using given coeffs
    zdatar = zdata
    return fun2(coeffs, xdata, ydata) - zdata

def fun3(coeffs, xdata, ydata, zdata):
    focus = coeffs[0]
    v1x = coeffs[1]
    v1y = coeffs[2]
    v2 = coeffs[3]
    aroundX = coeffs[4]
    aroundY = coeffs[5]

    # rotate the data
    L = np.array([xdata.flatten(), ydata.flatten(), zdata.flatten()])
    pry = (aroundX, aroundY, 0.)
    xr = PrimeX(pry, L)
    yr = PrimeY(pry, L)
    zr = PrimeZ(pry, L)

    # now find parabola
    z3 = parabola(xr, yr, focus, v1x, v1y, v2)
    z4 = zr - z3

    return z4

def fitRot():

    
    n = 20
    n2 = 50
    x = np.linspace(-n, n)
    y = np.linspace(-n, n)

    x2d, y2d = np.meshgrid(x, y)

    focus = 1.
    v1x = 0.
    v1y = 0.
    v2 = 0.
    #rotX = np.pi/2. 
    rotX = np.pi
    rotY = 0. #np.pi/2.
    #rotZ = -np.pi/4.
    actuals = [focus, v1x, v1y, v2, rotX, rotY]
    xrot, yrot, fakedata = parabolaRot(x2d, y2d, focus, v1x, v1y, v2, rotX, rotY)
    fakedata.shape = xrot.shape = yrot.shape = (n2, n2)
    fakedata = fakedata + np.random.rand(n2, n2)

    #coeffs = [focus, v1x, v1y, v2, rotX, rotY]
    #coeffs = [5., 4., 3., 2., 0., 0.]
    coeffs = [1.0, 0., 0., 0., np.pi, 0.0]
    #r = least_squares(errfun2,
    pi2 = 2*np.pi
    inf = np.inf
    infb = (-inf, inf)
    #bounds = [infb, infb, infb, infb, (-pi2, pi2), (-pi2, pi2)]
    b1 = [-inf, -inf, -inf, -inf, -pi2, -pi2]
    b2 = [inf, inf, inf, inf, pi2, pi2]
    bounds = (b1, b2)
    r = least_squares(fun3,
                      coeffs,
                      args=(x2d.flatten(), y2d.flatten(), fakedata.flatten()),
                      #args=(xrot.flatten(), yrot.flatten(), fakedata.flatten()),
                      bounds=bounds,
                      max_nfev=1e4,
                      ftol=1e-10,
                      xtol=1e-10)
    print(r)
    print("real:", focus, v1x, v1y, v2, rotX, rotY)
    print("guessed:", coeffs)
    print("vs:", r.x)
    #print "vs:", r[0]
    c = r.x

    z4 = fun3(c, x2d.flatten(), y2d.flatten(), fakedata.flatten())

    xrot, yrot, fittedData = parabolaRot(x2d.flatten(), y2d.flatten(), c[0], c[1], c[2], c[3], c[4], c[5])
    #xrot, yrot, fittedData = parabolaRot(xrot.flatten(), yrot.flatten(), c[0], c[1], c[2], c[3], c[4], c[5])
    fittedData.shape = xrot.shape = yrot.shape = (n2, n2)


    residuals = fittedData - fakedata
    print("max residual: ", np.max(residuals))
    print("mean residual: ", np.mean(residuals))

    return x2d, y2d, xrot, yrot, fakedata, fittedData, actuals, coeffs, c, z4


def funFinal(coeffs, xdata, ydata):
    "function for forming a parabola - no rotations"
    f = coeffs[0]
    v1x = coeffs[1]
    v1y = coeffs[2]
    v2 = coeffs[3]
    # optional Heavy Side
    hr = 30.
    return parabola(xdata, ydata, f, v1x, v1y, v2) 
    
def errfunFinal(coeffs, xdata, ydata, zdata):
    "Error function handles rotation properly"
    
    # rotate the original data first!
    xr, yr, zr = rotateXY(xdata, ydata, zdata, coeffs[4], coeffs[5])
    z = funFinal(coeffs, xr, yr)
    return zr - z
    
def simData(xs2d, ys2d, f, v1x, v1y, v2, xRot, yRot):
    "Returns a rotated parabola based off inputs"

    zs2d = parabola(xs2d, ys2d, f, v1x, v1y, v2)
    xdata, ydata, zdata = rotateXY(xs2d, ys2d, zs2d, xRot, yRot)
    return xdata, ydata, zdata

def lsqFit(xdata, ydata, zdata, coeffs):
    "least squares fit"

    # bound the fit
    inf = np.inf
    pi2 = 2*np.pi
    b1 = [-inf, -inf, -inf, -inf, -pi2, -pi2]
    b2 = [inf, inf, inf, inf, pi2, pi2]
    bounds = (b1, b2)

    r = least_squares(errfunFinal, coeffs, args=(xdata.flatten(), ydata.flatten(), zdata.flatten()),
                          bounds=bounds,
                          method='trf',
                          max_nfev=100000,
                          gtol=1e-15,
                          ftol=1e-15,
                          xtol=1e-15)
    return r 



def tryFit(answer, guess=None):
    "simulate a parabola, and see if we can try and fit it."

    f, v1x, v1y, v2, xRot, yRot = answer

    # angles!
    expAnswer = [f, v1x, v1y, v2, -xRot, -yRot]

    ##########################
    #     Create  Data       #
    ##########################
    xs = np.linspace(-20, 20)
    ys = np.linspace(-20, 20)
    
    xs2d, ys2d = np.meshgrid(xs, ys)
    
    #answer = [f, v1x, v1y, v2, -xRot, -yRot]

    xdata, ydata, zdata = simData(xs2d, ys2d, f, v1x, v1y, v2, xRot, yRot)

    # did we supply a guess?
    if guess is None:
        # then cheat!
        coeffs = [f, v1x, v1y, v2, -xRot, -yRot]
    else:
        coeffs = guess

    ##########################
    #     Create  Fit        #
    ##########################
    # bound the fitting
    #inf = np.inf
    #pi2 = 2*np.pi
    #b1 = [-inf, -inf, -inf, -inf, -pi2, -pi2]
    #b2 = [inf, inf, inf, inf, pi2, pi2]
    #bounds = (b1, b2)

    #r = least_squares(errfunFinal, coeffs, args=(xdata.flatten(), ydata.flatten(), zdata.flatten()),
    #                      bounds=bounds,
    #                      method='trf',
    #                      max_nfev=100000,
    #                      gtol=1e-15,
    #                      ftol=1e-15,
    #                      xtol=1e-15)
    r = lsqFit(xdata, ydata, zdata, coeffs)
    return expAnswer, coeffs, r.x, np.abs(expAnswer - r.x)

def tryFits():
    "A sequence of manual tests"

    f = 5.0
    v1x = 0.0
    v1y = 0.0
    v2 = 10.0
    xRot = np.pi/2
    yRot = 0. 
    data = [f, v1x, v1y, v2, xRot, yRot]

    # pass
    answer, guess, fit, diff = tryFit(data)
    checkFit(answer, guess, fit, diff)

    # pass
    data[5] = 0.1
    #answer, guess, fit, diff = testFit(f, v1x, v1y, v2, xRot, 0.1)
    answer, guess, fit, diff = tryFit(data)
    checkFit(answer, guess, fit, diff)

    # pass
    data[5] = np.pi/10 
    #answer, guess, fit, diff = testFit(f, v1x, v1y, v2, xRot, np.pi/10)
    answer, guess, fit, diff = tryFit(data)
    checkFit(answer, guess, fit, diff)

    # fail!
    data[5] = np.pi/2 
    #answer, guess, fit, diff = testFit(f, v1x, v1y, v2, xRot, np.pi/2)
    answer, guess, fit, diff = tryFit(data)
    checkFit(answer, guess, fit, diff)

    # pass!
    data[4] = np.pi/10 
    data[5] = np.pi/10 
    #answer, guess, fit, diff = testFit(f, v1x, v1y, v2, np.pi/10, np.pi/10)
    answer, guess, fit, diff = tryFit(data)
    checkFit(answer, guess, fit, diff)

    # now see how far off the guesses can be
    # pass!
    guess = data
    answer, guess, fit, diff = tryFit(data, guess)
    checkFit(answer, guess, fit, diff)

    guess = [1., 0., 0., 0., 0., 0.]
    answer, guess, fit, diff = tryFit(data, guess)
    checkFit(answer, guess, fit, diff)

def checkFit(answer, guess, fit, diff):
    tol = 1e-1
    for i, d in enumerate(diff):
        #print i, answer[i], fit[i]
        #assert d < tol
        if d > tol:
            print("fit failed with diff: ", answer, diff)
            return False
    print("fit passed with diff: ", answer, diff)
    return True

def radialFilter(x, y, z, xOffset, yOffset, radius):
    "returns only those points within the radius"
    xr = []
    yr = []
    zr = []
    for i in range(len(x)):
        r = np.sqrt((x[i]-xOffset)**2 + (y[i]-yOffset)**2)
        if r < radius:
            xr.append(x[i])
            yr.append(y[i])
            zr.append(z[i])
    xr = np.array(xr)        
    yr = np.array(yr)        
    zr = np.array(zr)        
    return xr, yr, zr

def radialMask(x, y, z, xOffset, yOffset, radius):
    r = np.sqrt((x-xOffset)**2 + (y-yOffset)**2)
    x[r > radius] = np.nan
    y[r > radius] = np.nan
    z[r > radius] = np.nan
    return x, y, z
    
def main():
    #tryFits()
    fn = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing/randomSampleScan10.csv"
    x, y, z = loadLeicaDataFromGpus(fn)
    print(x, x[np.logical_not(np.isnan(x))])
    print(y)
    print(z)

if __name__=='__main__':
    main()
