import csv

import numpy as np
from astropy.coordinates import cartesian_to_spherical
from astropy.coordinates import spherical_to_cartesian
from astropy.time import Time

def mjd2utc(mjd):
    "Converts MJD values to UTC datetime objects"
    t = Time(mjd, format='mjd')
    t.format = 'datetime'
    return t.value

def utc2mjd(dt):
    "Converts MJD values to UTC datetime objects"
    t = Time(dt, format='datetime')
    t.format = 'mjd'
    return t.value

def cart2sph(x, y, z, verbose=True):
    "Wrapper around astropy's cartesian_to_spherical"
    rs, lats, lngs = cartesian_to_spherical(x, y, z)

    # convert from astropy Quantities to simple numpy
    rs = np.array([r.value for r in rs])
    lats = np.array([l.value for l in lats])
    lngs = np.array([l.value for l in lngs])

    if verbose:
        print("min/max lats (radians)", lats.min(), lats.max())
        print("lats supposed range (radians)", -np.pi/2, np.pi/2)
        print("min/max lngs (radians)", lngs.min(), lngs.max())
        print("lngs supposed range (radians)", 0, 2*np.pi)

    return rs, lats, lngs

def sph2cart(az, el, r, verbose=True):
    "Wrapper around astropy's spherical_to_cartesian"

    if verbose:
        print("min/max az (radians)", np.nanmin(az), np.nanmax(az))
        print("az supposed range (radians)", -np.pi/2, np.pi/2)
        print("min/max el (radians)", np.nanmin(el), np.nanmax(el))
        print("el supposed range (radians)", 0, 2*np.pi)
        print("radial ranges", np.nanmin(r), np.nanmax(r))

    xs, ys, zs = spherical_to_cartesian(r, az, el)

    # convert from astropy Quantities to simple numpy
    xs = np.array([x.value for x in xs])
    ys = np.array([y.value for y in ys])
    zs = np.array([z.value for z in zs])

    return xs, ys, zs 

def aggregateXYZ(x, y, z):
    "x, y, z -> [(x0, y0, z0), (x1, y1, z1), ...]"
    xyz = []
    for i in range(len(x)):
        xyz.append((x[i], y[i], z[i]))
    return np.array(xyz)

def splitXYZ(xyz):
    "[(x0, y0, z0), (x1, y1, z1), ...] -> x, y, z"

    x = []
    y = []
    z = []

    for xi, yi, zi in xyz:
        x.append(xi); y.append(yi); z.append(zi)
        
    x = np.array(x)
    z = np.array(z) 
    y = np.array(y)    
    return x, y, z

def log(x):
    "short cut to numpy log"
    return np.log(np.abs(x))

def difflog(x):
    "short cut to tnumpy log and diff"
    return np.log(np.abs(np.diff(x)))

def log10(x):
    "short cup to numpy log10"
    return np.log10(np.abs(x))

def circular_mask(shape, centre, radius, angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order and in degrees.
    Centre and radius are in pixels.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx, y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

def midPoint(x):
    """
    Returns the midpoint of an array:
    (max(x) - min(x))/2 + min(x)
    """

    return (np.nanmax(x) - np.nanmin(x))/2. + np.nanmin(x)

def gridLimits(arr0, arr1):
    """
    Finds the minimum and maximum values present in two arrays.
    """

    vmin = np.min([np.nanmin(arr0), np.nanmin(arr1)])
    vmax = np.max([np.nanmax(arr0), np.nanmax(arr1)])

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
