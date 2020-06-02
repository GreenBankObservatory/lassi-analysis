import csv
import random
import warnings
import numpy as np

from copy import copy
from numpy.lib.stride_tricks import as_strided

from skimage import morphology
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

from astropy.time import Time
#from astropy.coordinates import cartesian_to_spherical
#from astropy.coordinates import spherical_to_cartesian


def polyFitRANSAC(x, y, order):
    """
    Uses the RANSAC algorithm to fit a polynomial.

    Parameters
    ----------
    x : array
        x coordinates.
    y : array
        y coordinates.
    order : int
        Polynomial order.
    """

    y = np.ma.masked_invalid(y)

    model = make_pipeline(PolynomialFeatures(order), RANSACRegressor(min_samples=0.5))
    model.fit(x[~y.mask,np.newaxis], y[~y.mask,np.newaxis])
    yp = model.predict(x[:,np.newaxis])

    return yp[:,0]


def mjd2utc(mjd):
    """
    Converts MJD values to UTC datetime objects
    """

    t = Time(mjd, format='mjd')
    t.format = 'datetime'
    return t.value


def utc2mjd(dt):
    """
    Converts UTC values to MJD datetime objects
    """

    t = Time(dt, format='datetime')
    t.format = 'mjd'
    return t.value


#def cart2sph(x, y, z, verbose=True):
#    "Wrapper around astropy's cartesian_to_spherical"
#    rs, lats, lngs = cartesian_to_spherical(x, y, z)
#
#    # convert from astropy Quantities to simple numpy
#    rs = np.array([r.value for r in rs])
#    lats = np.array([l.value for l in lats])
#    lngs = np.array([l.value for l in lngs])
#
#    if verbose:
#        print("min/max lats (radians)", lats.min(), lats.max())
#        print("lats supposed range (radians)", -np.pi/2, np.pi/2)
#        print("min/max lngs (radians)", lngs.min(), lngs.max())
#        print("lngs supposed range (radians)", 0, 2*np.pi)
#
#    return rs, lats, lngs
#
#
#def sph2cart(az, el, r, verbose=True):
#    "Wrapper around astropy's spherical_to_cartesian"
#
#    if verbose:
#        print("min/max az (radians)", np.nanmin(az), np.nanmax(az))
#        print("az supposed range (radians)", -np.pi/2, np.pi/2)
#        print("min/max el (radians)", np.nanmin(el), np.nanmax(el))
#        print("el supposed range (radians)", 0, 2*np.pi)
#        print("radial ranges", np.nanmin(r), np.nanmax(r))
#
#    xs, ys, zs = spherical_to_cartesian(r, az, el)
#
#    # convert from astropy Quantities to simple numpy
#    xs = np.array([x.value for x in xs])
#    ys = np.array([y.value for y in ys])
#    zs = np.array([z.value for z in zs])
#
#    return xs, ys, zs 


def aggregateXYZ(x, y, z):
    """
    x, y, z -> [(x0, y0, z0), (x1, y1, z1), ...]
    """
    
    xyz = []
    for i in range(len(x)):
        xyz.append((x[i], y[i], z[i]))
    return np.array(xyz)


def splitXYZ(xyz):
    """
    [(x0, y0, z0), (x1, y1, z1), ...] -> x, y, z
    """

    x = []
    y = []
    z = []

    for xi, yi, zi in xyz:
        x.append(xi); y.append(yi); z.append(zi)
        
    x = np.array(x)
    z = np.array(z) 
    y = np.array(y)    
    return x, y, z


def maskDiff(diff, window=(20,20), threshold=2):
    """
    Generates mask for `diff` based on the Z scores computed using a rolling window.
    """

    diff = np.ma.masked_invalid(diff)

    # Pad the map to avoid problems at the map edges.
    diff_pad = np.pad(diff.filled(np.nan), (window,window), constant_values=np.nan)
    diff_pad = np.ma.masked_invalid(diff_pad)

    # Compute rms and mean maps.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diff_rms = getRollingStat(diff_pad, func=np.nanstd, window=window)
        diff_avg = getRollingStat(diff_pad, func=np.nanmean, window=window)
    
    # Remove the extra padding from the rms and mean maps.
    diff_rms = diff_rms[window[0]:-window[1],window[0]:-window[1]]
    diff_avg = diff_avg[window[0]:-window[1],window[0]:-window[1]]

    # Mask zero values.
    diff_rms = np.ma.masked_where(diff_rms == 0, diff_rms)
    diff_rms = np.ma.masked_invalid(diff_rms)
    diff_avg = np.ma.masked_invalid(diff_avg)

    mask = (np.ma.abs(diff - diff_avg)/diff_rms > threshold) | (diff_rms == 0)
    
    return mask


def midPoint(x):
    """
    Returns the midpoint of an array:
    (max(x) - min(x))/2 + min(x)
    """

    return (np.nanmax(x) - np.nanmin(x))/2. + np.nanmin(x)


def radialMask(x, y, r, xc=None, yc=None):
    """
    Generates a circular mask.

    Parameters
    ----------
    x : ndarray
        Array with x coordinates to define a circle.
    y : ndarray
        Array with y coordinates to define a circle.
    r : float
        Radius of the circular mask. Values <=`r` will be masked.
    xc : float, optional
        Center of the circle in the x coordinate.
    yc : float, optional
        Center of the circle in the y coordinate.

    Returns
    -------
    array of bools
        Array of the same shape as x and y with True for values inside the circle.
    """

    if xc is None:
        xc = midPoint(x)
    if yc is None:
        yc = midPoint(y)

    return np.sqrt( (x - xc)**2. + (y - yc)**2. ) <= r


def ellipticalMask(x, y, xc, yc, bMaj, bMin, angle):
    """
    Generates an elliptical mask.

    Parameters
    ----------
    x : ndarray
        Array with x coordinates to define an ellipse.
    y : ndarray
        Array with y coordinates to define an ellipse.
    xc : float
        Center of the ellipse in the x coordinate.
    yc : float
        Center of the ellipse in the y coordinate.
    bMaj : float
        Major axis of the ellipse.
    bMin : float
        Minor axis of the ellipse.
    angle : float in degrees
        Rotation angle of the ellipse.
    
     Returns
    -------
    array of bools
        Array of the same shape as x and y with True for values inside the ellipse.
    """

    cos_angle = np.cos(np.radians(180. - angle))
    sin_angle = np.sin(np.radians(180. - angle))

    # Shift the points.
    xs = x - xc
    ys = y - yc

    # Rotate the points.
    xsr = xs * cos_angle - ys * sin_angle
    ysr = xs * sin_angle + ys * cos_angle

    mask = (xsr**2./(bMaj)**2.) + (ysr**2./(bMin)**2.) <= 1.

    return mask


def gridLimits(arr0, arr1):
    """
    Finds the minimum and maximum values common to two arrays.
    """

    vmin = np.max([np.nanmin(arr0), np.nanmin(arr1)])
    vmax = np.min([np.nanmax(arr0), np.nanmax(arr1)])

    return vmin, vmax


def dishLimits(maskedDish):
    """
    Returns the indices of the dish that are not masked.

    return: [xmin, xmax, ymin, ymax]
    """

    noz = np.ma.nonzero(maskedDish)
    ynm = noz[0]
    xnm = noz[1]

    return [np.min(xnm), np.max(xnm), np.min(ynm), np.max(ynm)]


def importCsv(filename):
    """
    Import x,y,z values from CSV file
    """


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


def sampleXYZData(x, y, z, samplePercentage):
    """
    Return a random percentage of the data.
    """

    assert len(x) == len(y)
    assert len(y) == len(z)

    lenx = len(x)

    sampleSize = int((lenx * samplePercentage) / 100.)

    idx = random.sample(range(lenx), sampleSize)

    return copy(x[idx]), copy(y[idx]), copy(z[idx])


def makeGrid(xmin, xmax, dx, ymin, ymax, dy):
    """
    Creates a regularly sampled cartesian grid.
    """

    # Make grid.
    xx,yy = np.mgrid[xmin:xmax:dx,
                     ymin:ymax:dy]
    zz = np.zeros((xx.shape[0],xx.shape[1]))

    return xx,yy,zz


def _check(a, r_c, subok=False):
    """
    Performs the array checks necessary for stride and block.

    Parameters
    ----------
    a : array or list.
    r_c : tuple/list/array of rows x cols.
    subok : from numpy 1.12 added, keep for now

    Returns
    -------
    tuple
        Attempts will be made to produce a shape at least (1*c).  
        For a scalar, the minimum shape will be (1*r) for 1D array 
        or (1*c) for 2D array if r<c.
    """

    if isinstance(r_c, (int, float)):
        r_c = (1, int(r_c))
    r, c = r_c
    if a.ndim == 1:
        a = np.atleast_2d(a)
    r, c = r_c = (min(r, a.shape[0]), min(c, a.shape[1]))
    a = np.array(a, copy=False, subok=subok)
    return a, r, c, tuple(r_c)


def stride(a, r_c=(3, 3)):
    """
    Provide a 2D sliding/moving view of an array.
    There is no edge correction for outputs.
    
    """
    
    a, r, c, r_c = _check(a, r_c)
    shape = (a.shape[0] - r + 1, a.shape[1] - c + 1) + r_c
    strides = a.strides * 2
    a_s = (as_strided(a, shape=shape, strides=strides)).squeeze()
    return a_s


def rolling_stat(a, func=np.nanstd, **kwargs):
    """
    Statistics on the last two dimensions of an array.

    Parameters
    ----------
    a : 2-D array 
        Array where to compute the statistics.
    func : callable, optinal
        Function used to compute the statistics, e.g., np.nanmean.
    kwargs : optinal
        Keyword arguments passed to `func`.
    """

    a = np.asarray(a)
    a = np.atleast_2d(a)
    ax = None
    if a.ndim > 1:
        ax = tuple(np.arange(len(a.shape))[-2:])
   
    a_stat = func(a, axis=ax, **kwargs)
   
    return a_stat


def getRollingStat(z, func=np.nanstd, window=(4,4), **kwargs):
    """
    Computes statistics over a 2D array using a rolling window.
    It will pad the array during the computation to fit the rolling window
    on the array's edges.

    Parameters
    ----------
    z : 2-D array
        Array where to compute the statistics.
    func : callable
        Function used to compute the statistics, e.g., np.nanmean.
    window : tuple of ints, optional
        Rolling window size.
    """
    
    zs = z.shape
    
    z_s = stride(z.filled(np.nan), r_c=window)
    z_stat = rolling_stat(z_s, func=func, **kwargs)
    
    # Pad the resulting array to get the same shape as the input.
    z_stat_pad = padArray(z_stat, zs, fill_value=np.nan)
    
    return z_stat_pad


def padArray(a, target_shape, fill_value=np.nan):
    """
    Pads a 2D array with a fill value to match the target shape.

    Parameters
    ----------
    a : 2-D array 
        Array to pad.
    target_shape : tuple
        Shape of the padded array.
    fill_value : float, optional
        Value to use on the new rows and columns.
    """

    a_shape = a.shape
    pad = np.subtract(target_shape, a_shape)
    pad_ = np.divide(pad, 2)
    pad0 = list(map(int, np.ceil(pad_)))
    padf = list(map(int, np.floor(pad_)))
    a_pad = np.pad(a, ((pad0[0],padf[0]),(pad0[1],padf[1])),
                        mode='constant', constant_values=fill_value)

    return a_pad


def maskEdges(a, window=(4,4)):
    """
    Mask the edges of an area with valid values.
    Invalid values should be NaNs.
    """

    shape = a.shape

    a_s = stride(a, r_c=window)
    nan_s = np.isnan(a_s)

    # Count the number of NaN values within the window.
    num_nan_s = np.sum(nan_s, axis=(2,3))
    num_nan_s_ = padArray(num_nan_s.astype(np.float), shape)
    # Mask any pixels which are adjacent to a NaN pixel.
    mask = num_nan_s_ > 0
    # Remove islands from the mask.
    mask = morphology.remove_small_holes(mask, area_threshold=64)

    return np.ma.masked_where(mask, a)
