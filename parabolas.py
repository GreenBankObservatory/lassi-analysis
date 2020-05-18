
import os
import random
import numpy as np

from copy import copy
from scipy.optimize import least_squares

from rotate import *
from SmoothedFITS import SmoothedFITS
from utils.utils import sph2cart, cart2sph, sampleXYZData
from plotting import scatter3dPlot, surface3dPlot, imagePlot

"""
TODO: 
- move plotting methods to the plotting module.
- clean up unused methods.
- devise a method that can fit for the parabola focus
  while producing accurate rotation parameters (robust).
"""


def parabola(xdata, ydata, focus, v1x=0, v1y=0, v2=0, heavy=False):
    if heavy:
        return parabolaHeavy(xdata, ydata, focus, v1x, v1y, v2)
    else:
        return parabolaSimple(xdata, ydata, focus, v1x, v1y, v2)


def parabolaSimple(xdata, ydata, focus, v1x, v1y, v2):
    """
    Simply the equation for a 2-D parabola.
    """

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


def fitParabolaNew(L, f, v1x, v1y, v2, xTheta, yTheta):
    coeffs = [f, v1x, v1y, v2, xTheta, yTheta]
    return fitParabola(coeffs, L[0], L[1], L[2])


def fitParabolaWithWeights(coeffs, x, y, z, weights):
    
    z = fitParabola(coeffs, x, y, z)
    return z*weights


def fitParabola(coeffs, x, y, z):
    """
    Computes the residuals between a paraboloid and a 3D vector.
    coeffs[0] is the focus, 
    coeffs[1] the offset in the x coordinate, 
    coeffs[2] the offset in the y coordinate,
    coeffs[3] the offset in the z coordinate,
    coeffs[4] a rotation along the x axis,
    coeffs[5] a rotation along the y axis.
    """
    
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
                 plot=True,
                 guess=None):

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
    if guess is None:
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

    # Rotate original data using fitted coefficients.
    xThetaFit = r.x[4]
    yThetaFit = r.x[5]
    xrr, yrr, zrr = shiftRotateXYZ(x0, y0, z0, [0, 0, 0, xThetaFit, yThetaFit, 0]) # Slightly faster.
    #xrr, yrr, zrr = rotateData(x0, y0, z0, xThetaFit, yThetaFit)
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

    # Robust fit: weights outliers outside of f_scale less
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
            args = (x.flatten(), y.flatten(), z.flatten())
        else:
            if verbose:
                print("Using weights for fit!")
            method = fitParabolaWithWeights
            args = (x.flatten(), y.flatten(), z.flatten(), weights.flatten())
    else:
        if verbose:
            print("Using user supplied method for fit.")
        args = (x.flatten(), y.flatten(), z.flatten())

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
    """
    Loads Leica data stored by numpy's save at 
    the end of lassiAnalysis.processLeicaData.
    """
    
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


def loadLeicaDataFromGpus(fn):
    """
    Crudely loads x, y, z csv files into numpy arrays
    """

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
    """
    Loads data processed by lassiAnalysis.processLeicaScan.
    """

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


def costParabola(coef, x, y, z):
    """
    Cost function for a parabola.
    """
     
    fl = coef[0]
    x0 = coef[1]
    y0 = coef[2]
    z0 = coef[3]
    rx = coef[4]
    ry = coef[5]
    
    crx = np.cos(rx)
    srx = np.sin(rx)
    cry = np.cos(ry)
    sry = np.sin(ry)
    
    xp = x*cry + y*srx*sry + z*crx*sry - x0
    yp = y*crx - z*srx - y0
    zp = -x*sry + y*srx*cry + z*crx*cry - z0
    
    xp2 = np.power(xp, 2.)
    yp2 = np.power(yp, 2.)
    
    fp = 1./(4.*fl)*(xp2 + yp2)
    
    return fp - zp


def jacParabola(coef, x, y, z):
    """
    Jacobian for a parabola.
    """
    
    fl = coef[0]
    x0 = coef[1]
    y0 = coef[2]
    z0 = coef[3]
    rx = coef[4]
    ry = coef[5]
    
    crx = np.cos(rx)
    srx = np.sin(rx)
    cry = np.cos(ry)
    sry = np.sin(ry)
    
    xp = x*cry + y*srx*sry + z*crx*sry - x0
    yp = y*crx - z*srx - y0
    zp = -x*sry + y*srx*cry + z*crx*cry - z0
    
    xp2 = np.power(xp, 2.)
    yp2 = np.power(yp, 2.)
    
    dxdrx = y*crx*sry - z*srx*sry
    
    dxdry = -x*sry + y*srx*cry + z*crx*cry
    
    dydrx = -y*srx - z*crx
    
    dzdrx = y*crx*cry - z*srx*cry
    
    dzdry = -x*cry - y*srx*sry - z*crx*sry
    
    # Jacobian columns.
    dGdfl = -1/(4*fl**2)*(xp2 + yp2)
    
    dGdx0 = -1/(4*fl)*2*xp
    
    dGdy0 = -1/(4*fl)*2*yp
    
    dGdz0 = np.ones(len(z))
    
    dGdrx = 1/(4*fl)*(2*xp*dxdrx + 2*yp*dydrx) - dzdrx
    
    dGdry = 1/(4*fl)*(2*x*dxdry) - dzdry
    
    return np.array([dGdfl, dGdx0, dGdy0, dGdz0, dGdrx, dGdry]).T


def costParabolaZ(coef, x, y, z):
    """
    Cost function for a parabola with Z4, Z5 and Z6 deformations.
    """
    
    xm = 50
    ym = 100
    
    fl = coef[0]
    x0 = coef[1]
    y0 = coef[2]
    z0 = coef[3]
    rx = coef[4]
    ry = coef[5]
    c4 = coef[6]
    c5 = coef[7]
    c6 = coef[8]
    #c9 = coef[9]
    xc = coef[9]
    yc = coef[10]
    
    crx = np.cos(rx)
    srx = np.sin(rx)
    cry = np.cos(ry)
    sry = np.sin(ry)
    
    xp = x*cry + y*srx*sry + z*crx*sry - x0
    yp = y*crx - z*srx - y0
    zp = -x*sry + y*srx*cry + z*crx*cry - z0
    
    xh = (xp - xc)/xm
    yh = (yp - yc)/ym
    
    xp2 = np.power(xp, 2.)
    yp2 = np.power(yp, 2.)
    xh2 = np.power(xh, 2.)
    yh2 = np.power(yh, 2.)
    
    fp = 1/(4*fl)*(xp2 + yp2)
    z4 = c4*(xh2 - yh2)
    z5 = c5*(2*xh2 + 2*yh2 - 1.)
    z6 = c6*xh*yh
    #z9 = c9*(3*xh2*yh + 3*yh2*yh - 2*yh)
    
    return fp + z4 + z5 + z6 - zp


def jacParabolaZ(coef, x, y, z):
    """
    Jacobian for a parabola with Z4, Z5 and Z6 deformations.
    """
    
    xm = 50.
    ym = 100.
    
    fl = coef[0]
    x0 = coef[1]
    y0 = coef[2]
    z0 = coef[3]
    rx = coef[4]
    ry = coef[5]
    c4 = coef[6]
    c5 = coef[7]
    c6 = coef[8]
    #c9 = coef[9]
    xc = coef[9]
    yc = coef[10]
    
    crx = np.cos(rx)
    srx = np.sin(rx)
    cry = np.cos(ry)
    sry = np.sin(ry)
    
    xp = x*cry + y*srx*sry + z*crx*sry - x0
    yp = y*crx - z*srx - y0
    zp = -x*sry + y*srx*cry + z*crx*cry - z0
    
    dxdrx = y*crx*sry - z*srx*sry
    
    dxdry = -x*sry + y*srx*cry + z*crx*cry
    
    dydrx = -y*srx - z*crx
    
    dzdrx = y*crx*cry - z*srx*cry
    
    dzdry = -x*cry - y*srx*sry - z*crx*sry
    
    xh = (xp - xc)/xm
    yh = (yp - yc)/ym
    
    xp2 = np.power(xp, 2.)
    yp2 = np.power(yp, 2.)
    xh2 = np.power(xh, 2.)
    yh2 = np.power(yh, 2.)
    
    dxhdx = -1./xm # Same for dxh/dxc or dxh/dx0
    dyhdy = -1./ym # Same for dyh/dyc or dyh/dy0
    
    dxhdrx = dxhdx*dxdrx
    
    dxhdry = dxhdx*dxdry
    
    dyhdrx = dyhdy*dydrx
    
    dyhdry = 0
    
    dGdfl = -1./(4.*fl**2.)*(xp2 + yp2)
    
    dGdx0 = -1./(4.*fl)*2.*xp + c4*2.*xh*dxhdx + c5*4.*xh*dxhdx + c6*yh*dxhdx #+ c9*6.*xh*yh*dxhdx
    
    dGdy0 = -1./(4.*fl)*2.*yp - c4*2.*yh*dyhdy + c5*4.*yh*dyhdy + c6*xh*dyhdy #+ c9*(3.*xh2 + 9.*yh2 - 2.)*dyhdy
    
    dGdz0 = np.ones(len(z))
    
    dGdrx = 1./(4.*fl)*(2.*xp*dxdrx + 2.*yp*dydrx) + c4*(2.*xh*dxhdrx - 2.*yh*dyhdrx) + \
            c5*(4.*xh*dxhdrx + 4.*yh*dyhdrx) \
            + c6*(yh*dxhdrx + xh*dyhdrx) - dzdrx #+ c9*(6.*xh*yh*dxhdrx + 3.*xh*dyhdrx + 9.*yh2*dyhdrx - 2.*dyhdrx) - dzdrx
    
    dGdry = 1./(4.*fl)*(2.*x*dxdry) + c4*2.*xh*dxhdry + c5*4.*xh*dxhdry + c6*yh*dxhdry - dzdry #+ c9*6.*xh*yh*dxhdry - dzdry
    
    dGdc4 = (xh2 - yh2)
    
    dGdc5 = 2*xh2 + 2*yh2 - 1
    
    dGdc6 = xh*yh
    
    #dGdc9 = 3*xh2*yh + 3*yh2*yh - 2*yh
    
    dGdxc = c4*2.*xh*dxhdx + c5*4.*xh*dxhdx + c6*yh*dxhdx #+ c9*6.*xh*yh*dxhdx
    
    dGdyc = -c4*2.*yh*dyhdy + c5*4.*yh*dyhdy + c6*xh*dyhdy #+ c9*(3.*xh2*dyhdy + 9.*yh2*dyhdy - 2.*dyhdy)
    
    return np.array([dGdfl, dGdx0, dGdy0, dGdz0, dGdrx, dGdry, dGdc4, dGdc5, dGdc6, dGdxc, dGdyc]).T


def parabolaFit(x, y, z, guess, method=costParabola, jac=jacParabola, 
                bounds=None,  max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12, 
                loss="soft_l1", f_scale=1e-5, tr_solver=None, x_scale=None, verbose=False):
    
    # Set boundaries for the fit parameters.
    if bounds is None:
        inf = np.inf
        pi2 = 2*np.pi
        omin = -10
        omax = 10
        cmin = -1e0
        cmax = 1e0
        if len(guess) == 11:
            b1 = [59, omin, omin, -55, -pi2, -pi2, cmin, cmin, cmin, -2, 45]
            b2 = [61, omax, omax, -45,  pi2,  pi2, cmax, cmax, cmax,  2, 55]
        else:
            b1 = [59, -inf, -inf, -inf, -pi2, -pi2]
            b2 = [61,  inf,  inf,  inf,  pi2,  pi2]
            
        bounds = (b1, b2)
    
    if verbose:
        print(bounds)
    
    if x_scale is None:
        xsr = np.deg2rad(0.001)
        xsz = 5e-5
        xso = 0.2
        if len(guess) == 11:
            x_scale = [1e-3, xso, xso, xso, xsr, xsr, xsz, xsz, xsz, xso/50., xso/100.]
        else:
            x_scale = [1e-3, xso, xso, xso, xsr, xsr]
    
    if jac == None:
        jac = "3-point"
    
    args = (x.flatten(), y.flatten(), z.flatten())
    
    r = least_squares(method, guess, jac,
                      args=args,
                      bounds=bounds,
                      max_nfev=max_nfev,
                      loss=loss,
                      f_scale=f_scale,
                      ftol=ftol,
                      xtol=xtol,
                      gtol=gtol,
                      x_scale=x_scale,
                      tr_solver=tr_solver)
    return r


def main():
    #tryFits()
    fn = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing/randomSampleScan10.csv"
    x, y, z = loadLeicaDataFromGpus(fn)
    print(x, x[np.logical_not(np.isnan(x))])
    print(y)
    print(z)


if __name__=='__main__':
    main()
