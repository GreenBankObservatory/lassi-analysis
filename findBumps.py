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
    return (1 / (4.*focus))*(xdata - v1x)**2 + (1 / (4.*focus))*(ydata - v1y)**2 + (2*v2)

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


def fitScan(fn):

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

def main():

    fn = "data/Baseline_STA10_HIGH_METERS.csv.smoothed.sig.001.all.npz"
    #fitScan(fn)
    test2()

if __name__ == '__main__':
    main()
