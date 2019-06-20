
import numpy as np
from astropy.coordinates import cartesian_to_spherical
from astropy.coordinates import spherical_to_cartesian

def cart2sph(x, y, z, verbose=True):
    "Wrapper around astropy's cartesian_to_spherical"
    rs, lats, lngs = cartesian_to_spherical(x, y, z)

    # convert from astropy Quantities to simple numpy
    rs = np.array([r.value for r in rs])
    lats = np.array([l.value for l in lats])
    lngs = np.array([l.value for l in lngs])

    if verbose:
        print "min/max lats (radians)", lats.min(), lats.max()
        print "lats supposed range (radians)", -np.pi/2, np.pi/2
        print "min/max lngs (radians)", lngs.min(), lngs.max()
        print "lngs supposed range (radians)", 0, 2*np.pi

    return rs, lats, lngs

def sph2cart(az, el, r, verbose=True):
    "Wrapper around astropy's spherical_to_cartesian"

    if verbose:
        print "min/max az (radians)", np.nanmin(az), np.nanmax(az)
        print "az supposed range (radians)", -np.pi/2, np.pi/2
        print "min/max el (radians)", np.nanmin(el), np.nanmax(el)
        print "el supposed range (radians)", 0, 2*np.pi
        print "radial ranges", np.nanmin(r), np.nanmax(r)

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