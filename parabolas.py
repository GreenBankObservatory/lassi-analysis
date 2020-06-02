
import os
import random
import numpy as np

from copy import copy
from scipy.optimize import least_squares

from rotate import PrimeX, PrimeY, PrimeZ, align2Paraboloid
from SmoothedFITS import SmoothedFITS
from utils.utils import sampleXYZData
from plotting import scatter3dPlot, surface3dPlot, imagePlot



def paraboloid(xdata, ydata, focus, v1x=0, v1y=0, v2=0):
    """
    General function for computing a paraboloid.
    """

    return _paraboloidSimple(xdata, ydata, focus, v1x, v1y, v2)


def _paraboloidSimple(xdata, ydata, focus, vx, vy, vz):
    """
    Simply the equation for a 2-D parabola, a paraboloid.

    Parameters
    ----------
    xdata: array
        x coordinates where to evaluate the paraboloid.
    ydata : array
        y coordinates where to evaluate the paraboloid.
    focus : float
        Focal length of the paraboloid.
    vx : float
        Location of the paraboloid's vertex in the x coordinate.
    vy : float
        Location of the paraboloid's vertex in the y coordinate.
    vz : float
        Location of the paraboloid's vertex in the z coordinate.
    """

    return (1 / (4.*focus))*(xdata - vx)**2 + (1 / (4.*focus))*(ydata - vy)**2 + vz


def fitParaboloid(coeffs, x, y, z, w):
    """
    Computes the residuals between a paraboloid and a vector in 3D.
    coeffs[0] is the focal length, 
    coeffs[1] the position of the vertex in the x coordinate, 
    coeffs[2] the position of the vertex in the y coordinate,
    coeffs[3] the position of the vertex in the z coordinate,
    coeffs[4] a rotation around the x axis in radians,
    coeffs[5] a rotation around the y axis in radians.
    """
    
    L = np.array([x.flatten(), y.flatten(), z.flatten()])
    pry = (coeffs[4], coeffs[5], 0.)
    xr = PrimeX(pry, L)
    yr = PrimeY(pry, L)
    zr = PrimeZ(pry, L)
    
    zdata = paraboloid(xr, yr, coeffs[0], coeffs[1], coeffs[2], coeffs[3])

    return (zr - zdata)*w


def fitLeicaData(x, y, z, guess, bounds=None, weights=None, method=fitParaboloid, max_nfev=10000, ftol=1e-12, xtol=1e-12, verbose=False):
    """
    Fits a paraboloid to a scan of the primary reflector.

    Parameters
    ----------
    x : array
        Array with the x coordinates.
    y : array
        Array with the y coordinates.
    z : array
        Array with the z coordinates.
    guess : list
        List with the initial guess for the least squares fit.
        [focal length, x coordinate of the vertex, 
        y coordinate of the vertex, z coordinate of the vertex, 
        rotation around x, rotation around y]
        The rotations are in radians.
    """

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
    f_scale = 1.0
    if verbose:
        print("fitLeicaData with robust, soft_l1, f_scale %f" % f_scale)

    if weights is None:
        weights = np.ones(len(x.flatten()), dtype=np.float)

    args = (x.flatten(), y.flatten(), z.flatten(), weights.flatten())

    if verbose:
        print("Using this method for the fit:")
        print(method)

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
    Loads Leica data stored by `numpy.save` at 
    the end of `lassiAnalysis.processLeicaData`.
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
    Crudely loads (x,y,z) csv files into numpy arrays.
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
    Loads data processed by `lassiAnalysis.processLeicaScan`.
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


def costParaboloid(coef, x, y, z):
    """
    Cost function for a paraboloid.


    Parameters
    ----------
    coef : list
        List with the coefficients describing a shifted rotated paraboloid.
        [focal length, shift in x, shift in y, shift in z, rotation around x, rotation around y]
    x : array
        1-d array with the x coordinates where to evaluate the parabolid.
    y : array
        1-d array with the y coordinates where to evaluate the parabolid.
    z : array
        1-d array with the z coordinates of the data to be compared with the paraboloid.

    Returns
    -------
    result : array
        1-d array with the difference between the paraboloid 
        described by coefs and the z coordinates.
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


def jacParaboloid(coef, x, y, z):
    """
    Jacobian for a shifted rotated paraboloid.

    Parameters
    ----------
    coef : list
        List with the coefficients describing a shifted rotated paraboloid.
        [focal length, shift in x, shift in y, shift in z, rotation around x, rotation around y]
    x : array
        1-d array with the x coordinates where to evaluate the parabolid.
    y : array
        1-d array with the y coordinates where to evaluate the parabolid.
    z : array
        1-d array with the z coordinates of the data to be compared with the paraboloid.

    Returns
    -------
    result : array
        6xN array with the Jacobian.
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


def costParaboloidZ(coef, x, y, z):
    """
    Cost function for a shifted rotated paraboloid with Z4, Z5 and Z6 deformations.
    It assumes that the projection of the paraboloid into the XY plane is a circle
    of radius 50 m.

    Parameters
    ----------
    coef : list
        List with the coefficients describing a shifted rotated paraboloid.
        [focal length, shift in x, shift in y, shift in z, 
         rotation around x, rotation around y, C4, C5, C6, 
         center of the deformation in x, center of the deformation in y]
    x : array
        1-d array with the x coordinates where to evaluate the parabolid.
    y : array
        1-d array with the y coordinates where to evaluate the parabolid.
    z : array
        1-d array with the z coordinates of the data to be compared with the paraboloid.

    Returns
    -------
    result : array
        1-d array with the difference between the paraboloid 
        described by coefs and the z coordinates.
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


def jacParaboloidZ(coef, x, y, z):
    """
    Jacobian for a shifted rotated paraboloid with Z4, Z5 and Z6 deformations.
    It assumes that the projection of the paraboloid into the XY plane is a circle
    of radius 50 m.

    Parameters
    ----------
    coef : list
        List with the coefficients describing a shifted rotated paraboloid.
        [focal length, shift in x, shift in y, shift in z, 
         rotation around x, rotation around y, C4, C5, C6, 
         center of the deformation in x, center of the deformation in y]
    x : array
        1-d array with the x coordinates where to evaluate the parabolid.
    y : array
        1-d array with the y coordinates where to evaluate the parabolid.
    z : array
        1-d array with the z coordinates of the data to be compared with the paraboloid.

    Returns
    -------
    result : array
        11xN array with the Jacobian.

    See Also
    --------
    costParaboloidZ : Compute the cost function of the deformed paraboloid.
    costParaboloid : Compute the cost function of an ideal paraboloid.
    jacParaboloid : Compute the Jacobian of an ideal paraboloid.

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


def paraboloidFitter(x, y, z, guess, method=costParaboloid, jac=jacParaboloid, 
                     bounds=None,  max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12, 
                     loss="soft_l1", f_scale=1e-5, tr_solver=None, x_scale=None, verbose=False):
    """
    Fits a paraboloid to the (x,y,z) data. This is a wrapper around `scipy.optimize.least_squares`.

    Parameters
    ----------
    x : array
        1-d array with the x coordinates where to evaluate the parabolid.
    y : array
        1-d array with the y coordinates where to evaluate the parabolid.
    z : array
        1-d array with the z coordinates of the data to be compared with the paraboloid.
    guess : list
        Starting values for the least-squares minimization.
        If using `method` = `costParaboloid` then `guess` = [focal length, 
        shift in x, shift in y, shift in z, rotation around x, rotation around y].
    method : callable
        Method used to compute the vector of residuals.
    jac : {'2-point', '3-point', callable}, optional
        Method used to compute the Jacobian matrix. See `scipy.optimize.least_squares`
        for details.
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on independent variables. See `scipy.optimize.least_squares`
        for details.
    max_nfev : None or int, optional
        Maximum number of function evaluations before the termination. See 
        `scipy.optimize.least_squares` for details.
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function. See 
        `scipy.optimize.least_squares` for details.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
        See `scipy.optimize.least_squares` for details.
    gtol : float or None, optional
        Tolerance for termination by the norm of the gradient.
        See `scipy.optimize.least_squares` for details.
    loss : str or callable, optional
        Determines the loss function. See `scipy.optimize.least_squares` for details.
    f_scale : float, optional
        Value of soft margin between inlier and outlier residuals.
        See `scipy.optimize.least_squares` for details.
    tr_solver : {None, 'exact', 'lsmr'}, optional
        Method for solving trust-region subproblems.
        See `scipy.optimize.least_squares` for details.
    x_scale : array_like or 'jac', optional
        Characteristic scale of each variable.
        See `scipy.optimize.least_squares` for details.
    verbose : bool, optional
        Print additional information during method call. Defaults to False.

    Returns
    -------
    `OptimizeResult` with the field defined in `scipy.optimize.least_squares`.
    
    """

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
        print("Using bounds:")
        print(bounds)
    
    if x_scale is None:
        xsr = np.deg2rad(0.001)
        xsz = 5e-5
        xso = 0.2
        if len(guess) == 11:
            x_scale = [1e-3, xso, xso, xso, xsr, xsr, xsz, xsz, xsz, xso/50., xso/100.]
        else:
            x_scale = [1e-3, xso, xso, xso, xsr, xsr]
    
    if verbose:
        print("Using x_scale:")
        print(x_scale)

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


def subtractParaboloid(x, y, z, guess=[60, 0, 0, -50, 0, 0, 0, 0, 0, 0, 50],
                       method=costParaboloidZ, jac=jacParaboloidZ, tr_solver=None,
                       bounds=None, max_nfev=100000, ftol=1e-12, xtol=1e-15, 
                       f_scale=1e-2, verbose=False):
    """
    Fits a paraboloid to the data and subtracts it.
    
    Parameters
    ----------
    x : array
        1-d array with the x coordinates.
    y : array
        1-d array with the y coordinates.
    z : array
        1-d array with the z coordinates.
    guess : list, optional
        Initial guess for the least-squares problem.
    method : callable, optional
        Cost funtion for the least-squares problem.
    jac : callable, optional
        Function to compute the Jacobian.
    tr_solver : {'',None}, optional
    bounds : tuple, optional
        Bounds for the parameters.
    max_nfev : int, optional
    ftol : float, optional
    xtol : float, optional
    f_scale : float, optional
    verbose : bool, optional
        Verbose output?

    Returns
    -------
    tuple
        Residuals of the paraboloid fit and the 
        results for the best-fit paraboloid.
    """

    fit = paraboloidFit(x, y, z, guess, method=method, jac=jac, 
                        bounds=bounds, tr_solver=tr_solver, 
                        max_nfev=max_nfev, ftol=ftol, xtol=xtol,
                        f_scale=f_scale, verbose=verbose)

    if verbose:
        print("Results from the fit:")
        print(fit)

    # Align the scan with the paraboloid.    
    xr, yr, zr = align2Paraboloid(x, y, z, fit.x)
    # Subtract the paraboloid.
    diff = zr - paraboloid(xr, yr, fit.x[0])

    return diff, fit


def paraboloidFit(x, y, z, guess=[60, 0, 0, -50, 0, 0, 0, 0, 0, 0, 50],
                  method=costParaboloidZ, jac=jacParaboloidZ, tr_solver=None,
                  bounds=None, max_nfev=100000, ftol=1e-12, xtol=1e-15,
                  f_scale=1e-2, verbose=False):
    """
    Parameters
    ----------
    x : array
        1-d array with the x coordinates.
    y : array
        1-d array with the y coordinates.
    z : array
        1-d array with the z coordinates.
    guess : list, optional
        Initial guess for the least-squares problem.
    method : callable, optional
        Cost funtion for the least-squares problem.
    jac : callable, optional
        Function to compute the Jacobian.
    tr_solver : {'',None}, optional
    bounds : tuple, optional
        Bounds for the parameters.
    max_nfev : int, optional
    ftol : float, optional
    xtol : float, optional
    f_scale : float, optional
    verbose : bool, optional
        Verbose output?

    Returns
    -------
    tuple
        Results for the best-fit paraboloid.
    """

    zm = np.ma.masked_invalid(z)
    xm = np.ma.masked_where(zm.mask, x)
    ym = np.ma.masked_where(zm.mask, y)

    fit = paraboloidFitter(xm.compressed(), ym.compressed(), zm.compressed(),
                           guess, method=method, jac=jac, tr_solver=tr_solver,
                           bounds=bounds, max_nfev=max_nfev, ftol=ftol, xtol=xtol,
                           f_scale=f_scale, verbose=verbose)

    return fit


