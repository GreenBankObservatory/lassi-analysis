import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.optimize import least_squares

def Rx(theta):

    x1 = [1., 0., 0.]
    x2 = [0., np.cos(theta), -np.sin(theta)]
    x3 = [0., np.sin(theta), np.cos(theta)]

    return np.array([x1, x2, x3])

def Ry(theta):

    x1 = [np.cos(theta), 0., np.sin(theta)]
    x2 = [0., 1., 0.]
    x3 = [-np.sin(theta), 0., np.cos(theta)]

    return np.array([x1, x2, x3])

def Rz(theta):
    return np.array([[1., 0., 0.],[0., 1., 0.], [0., 0., 1.]])

def Prime(pry):
    #return Rx(pry[0])*Ry(pry[1])*Rz(pry[2])
    return np.dot(Rx(pry[0]), np.dot(Ry(pry[1]), Rz(pry[2])))

def PrimeX(pry, v):
    #return np.array([1., 0., 0.]) * Prime(pry) * v
    return np.dot(np.array([1., 0., 0.]), np.dot(Prime(pry), v))

def PrimeY(pry, v):
    #return np.array([0., 1., 0.]) * Prime(pry) * v
    return np.dot(np.array([0., 1., 0.]), np.dot(Prime(pry), v))

def PrimeZ(pry, v):
    cnst = np.array([0., 0., 1.])
    #print "cnst: ", cnst, cnst.shape
    ppry = Prime(pry)
    #print "pry", pry, pry.shape
    #print "Prime(pry)", ppry, ppry.shape
    #print "v:", v, v.shape
    #return np.array([0., 0., 1.]) * Prime(pry) * v
    return np.dot(cnst, np.dot(ppry, v))

def Func(b, x):
    #return 4 * b[3] * (b[3]-(PrimeZ(b[:3], x) - b[6]) - () + ()
    #x1 = b[3] - (PrimeZ(b[:3], x) - b[6])
    x1 = b[3] - (PrimeZ(b[:3], x) - b[6])
    x2 = (PrimeX(b[:3], x) - b[4]) ** 2 
    x3 = (PrimeY(b[:3], x) - b[5]) ** 2 
    #print "Func1: ", 4 * b[3] * x1 
    #print "Func2: ", (x2 + x3)
    #print "Func2 x2: ", x2
    #print "Func2 x3: ", x3
    return 4 * b[3] * x1 - (x2 + x3)

def FuncZ(b, x):
    #fz = @(b,x) b(4) - 0.25/b(4)*( (Prime_x(b(1:3),x)- b(5)).^2 ...
    #+(Prime_y(b(1:3),x)-b(6)).^2 );
    return b[3] - 0.25/b[3] * ((PrimeX(b[:3], x) - b[4])**2 + (PrimeY(b[:3], x) - b[5])**2)


def parabola(xdata, ydata, focus, v1x, v1y, v2):
    "returns a parabola for x, y data, conforming to given displacement coefficients"
    return (1 / (4.*focus))*(xdata - v1x)**2 + (1 / (4.*focus))*(ydata - v1y)**2 + (2*v2)


def fitParabola(coeffs, x, y, z):
    "Rotates data first, then finds new z from parabolar, then resturns the difference" 
    
    L = np.array([x.flatten(), y.flatten(), z.flatten()])
    pry = (coeffs[4], coeffs[5], 0.)
    xr = PrimeX(pry, L)
    yr = PrimeY(pry, L)
    zr = PrimeZ(pry, L)
    
    zdata = parabola(xr, yr, coeffs[0], coeffs[1], coeffs[2], coeffs[3])

    return zr - zdata

def newParabola(xdata, ydata, zdata, focus, v1x, v1y, v2, rotX, rotY):
    "rotates data, then returns parabola from rotated data and coefficients"
    L = np.array([xdata.flatten(), ydata.flatten(), zdata.flatten()])
    pry = (rotX, rotY, 0.)
    xr = PrimeX(pry, L)
    yr = PrimeY(pry, L)
    #zr = PrimeZ(pry, L)

    zdata = parabola(xr, yr, focus, v1x, v1y, v2)

    return xr, yr, zdata

def rotateData(xdata, ydata, zdata, rotX, rotY):
    "Returns rotated x, y, z data, by only rotations in x and y"
    L = np.array([xdata.flatten(), ydata.flatten(), zdata.flatten()])
    pry = (rotX, rotY, 0.)
    xr = PrimeX(pry, L)
    yr = PrimeY(pry, L)
    zr = PrimeZ(pry, L)
    return xr, yr, zr

def testRot():

    pry = [np.pi/4, -np.pi/2., np.pi/2.]

    n = 100
    x = np.linspace(-n, n)
    y = np.linspace(-n, n)
    x2, y2 = np.meshgrid(x, y)
    z2 = parabola(x2, y2, 1., 10., 80., 100.)

    L = np.array([x2.flatten(), y2.flatten(), z2.flatten()])
    xr = PrimeX(pry, L)
    yr = PrimeY(pry, L)
    zr = PrimeZ(pry, L)

    #N = 50
    #xr.shape = (N,N)
    #yr.shape = (N,N)
    #zr.shape = (N,N)
    xr.shape = x2.shape
    yr.shape = y2.shape
    zr.shape = z2.shape

    return xr, yr, zr


def fitScanOrg(fn):

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

    L = np.array([xxn, yyn, zzn])
    #tt = [1., 2., 3. , 4., 5., 6., 7., 8., 9.]
    #L = np.array([tt, tt, tt])
    L = np.transpose(L)
    print "shape L: ", L.shape
    xy = L
    xyPrime = np.transpose(xy)

    #popt, pcov = curve_fit(func, xdata, ydata)

    b = (0., 0., 0., 60., min(xxn), min(yyn), min(zzn))
    #args = (b
    #popt, pcov = curve_fit(Func, b, xyPrime)
    #fit, success = leastsq(Func, b, args=(xyPrime))
    maxfev = L.size * 200

    lsq = leastsq(Func, b, args=(xyPrime), maxfev=maxfev, full_output=True)
    #b = np.array([0., 0., 0., 60., min(xxn), min(yyn), min(zzn)])
    #lsq = least_squares(Func, b, args=(xyPrime))

    #print "fit: ", fit, success
    print lsq

    #fittedData = Func(lsq.x, xyPrime)
    fittedData = None
    return xx, yy, zz, xyPrime, lsq, fittedData

def test1():

    x = range(10)
    L = np.transpose(np.array([x, x, x]))
    xyPrime = np.transpose(L)

    b = (0., 0., 0., 60., min(x), min(x), min(x))
    maxfev = L.size * 200

    print "Func:"
    r = Func(b, xyPrime)
    print r

    #lsq = leastsq(Func, b, args=(xyPrime), maxfev=maxfev, full_output=True)
    #lsq = leastsq(Func, b, args=(xyPrime), maxfev=maxfev)

    #print "fit: ", fit, success
    #print lsq

    exp = np.array([14400., 14158., 13912., 13662., 13408., 13150., 12888., 12622., 12352., 12078.])
    assert (r == exp).all()
  
    print "Primes:"
    print "PrimeX:"
    print PrimeX(b[:3], xyPrime)

def test2():

    print "test2"
    #x = np.array(range(100)) * .5
    #y = np.array(range(100, 200)) * .1
    #z = np.array(range(200, 300)) * .01
    x = np.array(range(10)) * 0.5
    y = np.array(range(10, 20)) * .1
    z = np.array(range(20, 30)) * 0.01
    L = np.transpose(np.array([x, y, z]))
    xyPrime = np.transpose(L)

    b = (0., 0., 0., 60., min(x), min(y), min(z))
    #maxfev = L.size * 200
    maxfev = 512*512*3 * 200

    print "testing with", b
    print xyPrime
    r = Func(b, xyPrime)
    print r

    exp = np.array([14400., 14397.34, 14394.16, 14390.46, 14386.24, 14381.5, 14376.24, 14370.46,
 14364.16, 14357.34])
    assert np.min(np.abs(r - exp)) < 1e-10

    print "Primes:"
    print PrimeX(b[:3], xyPrime)
    print PrimeY(b[:3], xyPrime)
    print PrimeZ(b[:3], xyPrime)

    print "Least Squares: "
    lsq = leastsq(Func, b, args=(xyPrime), maxfev=maxfev)
    #lsq = least_squares(Func, b, args=(xyPrime))
    print lsq
    fit = lsq[0]
    matlabFit = np.array([0, -7.6577, 1.4819, .0001, -1.0566, -0.1675, 15.5614])

    for i in range(len(fit)):
        print i, fit[i], matlabFit[i], fit[i] - matlabFit[i]

def fitScan(fn):
    "Load Leica scanner data, and fit it."

    # the NxN size of the Leica scanner data
    N = 512

    # load data
    data = np.load(fn)
    x = data['x']
    y = data['y']
    z = data['z']

    # prepare data - remove NaNs
    xx = x.flatten()
    yy = y.flatten()
    zz = z.flatten()
    
    xxn = xx[np.logical_not(np.isnan(xx))];
    yyn = yy[np.logical_not(np.isnan(yy))];
    zzn = zz[np.logical_not(np.isnan(zz))];

    L = np.array([xxn, yyn, zzn])

    # fit data:
    # what's our initial guess?
    f = 60. # focus
    v1x = v1y = v2 = 0. # displacement
    xTheta = 0.
    yTheta = 0.
    
    guess = [f, v1x, v1y, v2, xTheta, yTheta]

    # lets bound the fit (for methods other then 'lm')
    inf = np.inf
    pi2 = 2*np.pi
    b1 = [-inf, -inf, -inf, -inf, -pi2, -pi2]
    b2 = [inf, inf, inf, inf, pi2, pi2]
    bounds = (b1, b2)

    # OK, actually find the fit
    r = least_squares(fitParabola, guess, args=(xxn.flatten(), yyn.flatten(), zzn.flatten()),
                      #bounds=bounds,
                      method='lm',
                      max_nfev=1000000,
                  
                      ftol=1e-15,
                      xtol=1e-15)

    print "answer: ", r.x
    print "success? ", r.success

    # get the parabola from the original data, but the new fitted coefficients
    c = r.x
    xThetaFit = c[4]
    yThetaFit = c[5]
    newX, newY, newZ = newParabola(xx, yy, zz, c[0], c[1], c[2], c[3], xThetaFit, yThetaFit)

    # rotate the original data via our fitted rotations
    xrr, yrr, zrr = rotateData(xx, yy, zz, xThetaFit, yThetaFit)

    # reintroduce the known shape
    newX.shape = newY.shape = newZ.shape = (N, N)
    xrr.shape = yrr.shape = zrr.shape = (N, N)

    # our 'fit' is actually the difference between these two parabolas (why?)
    return zrr - newZ, (newX, newY, newZ), (xrr, yrr, zrr)


def findTheBumps():
    "Fit ref and bump scan, and the difference shows us the bumps!"
   
    fn = "data/Baseline_STA10_HIGH_METERS.csv.smoothed.sig.001.all.npz"
    fit1 = fitScan(fn)
    fn = "data/BumpScan.csv.smoothed.sig.001.all.npz"
    fit2 = fitScan(fn)
    diff = fit2[0] - fit1[0]

    return diff, fit1, fit2

def exportBumps(diff, fit1, fit2):

    d = ','
    np.savetxt("lassiDiffZ.csv", diff, delimiter=d)
    
    # also save the x and y
    diff1, newXYZ, rotXYZ = fit1

    x, y, z = rotXYZ
    
    np.savetxt("lassiX.csv", x, delimiter=d)
    np.savetxt("lassiY.csv", y, delimiter=d)

def main():

    d, f1, f2 = findTheBumps()
    exportBumps(d, f1, f2)

    #fn = "data/Baseline_STA10_HIGH_METERS.csv.smoothed.sig.001.all.npz"
    #fitScan(fn)
    #test2()

if __name__ == '__main__':
    main()
