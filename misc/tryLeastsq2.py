import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import least_squares


def fun(x, xdata):
    return x[0]*np.exp(x[1]*xdata)

def errfun(x, xdata, ydata):
    return fun(x, xdata) - ydata

def tryLeastsq():

    xdata = np.array([0.9, 1.5, 13.8, 19.8, 24.1, 28.2, 35.2, 60.3, 74.6, 81.3])
    ydata =  np.array([455.2, 428.6, 124.1, 67.3, 43.2, 28.1, 13.1, -0.4, -1.3, -1.5])

    #fun = @(x,xdata)x(1)*exp(x(2)*xdata);
    
    x0 = np.array([100,-1])
    #[x,resnorm,residual,exitflag,output] = lsqcurvefit(fun,x0,xdata,ydata);    
    #x, success = leastsq(errfun, x0[:], args=(xdata, ydata))
    r = least_squares(errfun, x0, args=(xdata, ydata))
    #print x
    #print success
    print(r.success)
    print(r.x)

    return xdata, ydata, r.x, r.success

def main():
    tryLeastsq()

if __name__ == '__main__':
    main()
