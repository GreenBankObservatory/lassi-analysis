import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import leastsq
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

#from findBumps import PrimeX, PrimeY, PrimeZ

from rotate import *

def parabola(xdata, ydata, focus, v1x, v1y, v2):
    return parabolaOld(xdata, ydata, focus, v1x, v1y, v2)

def parabolaOld(xdata, ydata, focus, v1x, v1y, v2):
    return (1 / (4.*focus))*(xdata - v1x)**2 + (1 / (4.*focus))*(ydata - v1y)**2 + v2

def parabolaHeavy(xdata, ydata, focus, v1x, v1y, v2):
    #return (1 / (4.*focus))*(xdata - v1x)**2 + (1 / (4.*focus))*(ydata - v1y)**2 + v2
    r2 = (1 / (4.*focus))*(xdata - v1x)**2 + (1 / (4.*focus))*(ydata - v1y)**2 
    return (r2 * (104**2 > r2)) + v2

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
    
    zdata = parabola(xr, yr, focus, v1x, v1y, v2)

    return xr, yr, zdata


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

def fitLeicaScan(fn):

    N = 512

    orgData, cleanData = loadLeicaData(fn)
    x0, y0, z0 = orgData
    x, y, z = cleanData
    scatter3dPlot(x, y, z, "Leica Data")

    f = 60.
    v1x = v1y = v2 = 0
    xTheta = 0. #-np.pi / 2.
    yTheta = 0.
    guess = [f, v1x, v1y, v2, 0., 0.]
    r = fitLeicaData(x, y, z, guess)

    # plot fitted data
    c = r.x
    print "cleaned data fitted with coefficients: ", c
    newX, newY, newZ = newParabola(x0, y0, z0, c[0], c[1], c[2], c[3], c[4], c[5])
    newX.shape = newY.shape = newZ.shape = (N, N)
    surface3dPlot(newX, newY, newZ, "Fitted Data")

    # rotate original rotations for fitted coefficients
    xThetaFit = r.x[4]
    yThetaFit = r.x[5]
    xrr, yrr, zrr = rotateData(x0, y0, z0, xThetaFit, yThetaFit)
    xrr.shape = yrr.shape = zrr.shape = (N, N)
    surface3dPlot(xrr, yrr, zrr, "Original Data (Rotated)")

    # return difference between fitted data and rotated original data
    diff = zrr - newZ
    diff2 = np.log(np.abs(np.diff(diff)))
    imagePlot(diff2, "Fit - Org (Rotated)")

    return diff

def imagePlot(z, title):
    fig = plt.figure()
    ax = fig.gca()
    #cax = ax.imshow(np.log(np.abs(np.diff(zrr - newZ))))
    cax = ax.imshow(z)
    fig.colorbar(cax)
    plt.title(title)



def fitLeicaData(x, y, z, guess):

    #guess = [f, v1x, v1y, v2, 0., 0.]
    inf = np.inf
    pi2 = 2*np.pi
    b1 = [-inf, -inf, -inf, -inf, -pi2, -pi2]
    b2 = [inf, inf, inf, inf, pi2, pi2]
    bounds = (b1, b2)
    r = least_squares(fitParabola, guess, args=(x .flatten(), y.flatten(), z.flatten()),
                      #bounds=bounds,
                      method='lm',
                      max_nfev=1000000,
                  
                      ftol=1e-15,
                      xtol=1e-15)
    return r

def loadLeicaData(fn):

    data = np.load(fn)

    x = data['x']
    y = data['y']
    z = data['z']

    xx = x.flatten()
    yy = y.flatten()
    zz = z.flatten()
    
    xxn = xx[np.logical_not(np.isnan(xx))];
    yyn = yy[np.logical_not(np.isnan(yy))];
    zzn = zz[np.logical_not(np.isnan(zz))];

    return (x, y, z), (xxn, yyn, zzn)

def surface3dPlot(x, y, z, title):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)

def scatter3dPlot(x, y, z, title):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)

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
    print r

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
    print r
    print "real:", focus, v1x, v1y, v2, rotX, rotY
    print "guessed:", coeffs
    print "vs:", r.x
    #print "vs:", r[0]
    c = r.x

    z4 = fun3(c, x2d.flatten(), y2d.flatten(), fakedata.flatten())

    xrot, yrot, fittedData = parabolaRot(x2d.flatten(), y2d.flatten(), c[0], c[1], c[2], c[3], c[4], c[5])
    #xrot, yrot, fittedData = parabolaRot(xrot.flatten(), yrot.flatten(), c[0], c[1], c[2], c[3], c[4], c[5])
    fittedData.shape = xrot.shape = yrot.shape = (n2, n2)


    residuals = fittedData - fakedata
    print "max residual: ", np.max(residuals)
    print "mean residual: ", np.mean(residuals)

    return x2d, y2d, xrot, yrot, fakedata, fittedData, actuals, coeffs, c, z4


def funFinal(coeffs, xdata, ydata):
    f = coeffs[0]
    v1x = coeffs[1]
    v1y = coeffs[2]
    v2 = coeffs[3]
    #aroundX = coeffs[4]
    #aroundY = coeffs[5]
    #return  (1 / (4.*f))*(xdata - v1x)**2 + (1 / (4.*f))*(ydata - v1y)**2 + v2
    hr = 30.
    #r2 = (xdata - v1x)**2 + (ydata - v1y)**2
    #return ( 1 / (4.*f) ) * r2 * ( hr**2 > r2 )  + v2
    return parabolaOld(xdata, ydata, f, v1x, v1y, v2) 
    
def errfunFinal(coeffs, xdata, ydata, zdata):
    
    # rotate the original data first!
    xr, yr, zr = rotateXY(xdata, ydata, zdata, coeffs[4], coeffs[5])
    #xr = xdata
    #yr = ydata
    #zr = zdata
    z = funFinal(coeffs, xr, yr)
    # now what we are trying to minimize is the difference between the rotated 
    # input data and a parabola in the same frame
    return zr - z

    
def simData(xs2d, ys2d, f, v1x, v1y, v2, xRot, yRot):

    #zs2d = (1 / (4.*f))*(xs2d  - v1x)**2 +  (1 / (4.*f))*(ys2d - v1y)**2 + v2
    zs2d = parabolaOld(xs2d, ys2d, f, v1x, v1y, v2)
    xdata, ydata, zdata = rotateXY(xs2d, ys2d, zs2d, xRot, yRot)
    return xdata, ydata, zdata

def fitRot2():

    ##########################
    #     Create  Data       #
    ##########################
    xs = np.linspace(-20, 20)
    ys = np.linspace(-20, 20)
    
    xs2d, ys2d = np.meshgrid(xs, ys)
    
    f = 5.0
    v1x = 0.0
    v1y = 0.0
    v2 = 10.0
    xRot = np.pi/2
    #xRot = 0.
    #yRot = 0.1
    yRot = 0 #np.pi/10
    
    answer = [f, v1x, v1y, v2, -xRot, -yRot]

    #zs2d = (1 / (4.*f))*(xs2d  - v1x)**2 +  (1 / (4.*f))*(ys2d - v1y)**2 + v2
    #xdata, ydata, zdata = rotateXY(xs2d, ys2d, zs2d, xRot, yRot)
    xdata, ydata, zdata = simData(xs2d, ys2d, f, v1x, v1y, v2, xRot, yRot)

    ##########################
    #     Create  Fit        #
    ##########################
    coeffs = [f, v1x, v1y, v2, -xRot, -yRot]
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
    # Look into robust method (loss='soft_l1', f_scale=0.1)
    print "fit:", r.x
    print r.success, r.message, r.nfev

    print "compare: "
    print answer - r.x

    return r

def testFit(f, v1x, v1y, v2, xRot, yRot):
    ##########################
    #     Create  Data       #
    ##########################
    xs = np.linspace(-20, 20)
    ys = np.linspace(-20, 20)
    
    xs2d, ys2d = np.meshgrid(xs, ys)
    
    answer = [f, v1x, v1y, v2, -xRot, -yRot]

    xdata, ydata, zdata = simData(xs2d, ys2d, f, v1x, v1y, v2, xRot, yRot)

    coeffs = [f, v1x, v1y, v2, -xRot, -yRot]
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
    return answer, coeffs, r.x, answer - r.x

def testFits():

    f = 5.0
    v1x = 0.0
    v1y = 0.0
    v2 = 10.0
    xRot = np.pi/2
    yRot = 0. 
   
    # pass
    answer, guess, fit, diff = testFit(f, v1x, v1y, v2, xRot, yRot)
    checkFit(answer, guess, fit, diff)

    # pass
    answer, guess, fit, diff = testFit(f, v1x, v1y, v2, xRot, 0.1)
    checkFit(answer, guess, fit, diff)

    # pass
    answer, guess, fit, diff = testFit(f, v1x, v1y, v2, xRot, np.pi/10)
    checkFit(answer, guess, fit, diff)

    # fail!
    answer, guess, fit, diff = testFit(f, v1x, v1y, v2, xRot, np.pi/2)
    checkFit(answer, guess, fit, diff)

    # pass!
    answer, guess, fit, diff = testFit(f, v1x, v1y, v2, np.pi/10, np.pi/10)
    checkFit(answer, guess, fit, diff)

def checkFit(answer, guess, fit, diff):
    tol = 1e-1
    for i, d in enumerate(diff):
        #print i, answer[i], fit[i]
        #assert d < tol
        if d > tol:
            print "fit failed with diff: ", answer, diff
            return False
    print "fit passed with diff: ", answer, diff
    return True


def main():
    testFits()

if __name__=='__main__':
    main()
