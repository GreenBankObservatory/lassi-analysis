import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import leastsq
from findBumps import PrimeX, PrimeY, PrimeZ

def parabola(xdata, ydata, focus, v1x, v1y, v2):
    return (1 / (4.*focus))*(xdata - v1x)**2 + (1 / (4.*focus))*(ydata - v1y)**2 + v2

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

def main():
    fitRot()

if __name__=='__main__':
    main()
