import numpy as np

#import opticspy

import zernikies

def addCenterBump(x, y, z, rScale=10., zScale=0.05):

    # add a bump to the center bit
    xMin = np.nanmin(x)
    xMax = np.nanmax(x)
    yMin = np.nanmin(y)
    yMax = np.nanmax(y)

    # d = 10
    d = rScale

    xW = xMax - xMin
    xStep = xW / d
    xStart = xMin + (d/2 - 1)*xStep
    xEnd = xMax - (d/2 -1)*xStep
    assert xStart < xEnd

    yW = yMax - yMin
    yStep = yW / d
    yStart = yMin + (d/2 - 1)*yStep
    yEnd = yMax - (d/2 -1)*yStep
    assert yStart < yEnd

    print(xStart, xEnd, yStart, yEnd)
    
    zadds = []
    cnt = 0
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        if xi > xStart and xi < xEnd and yi > yStart and yi < yEnd:
            cnt +=1 
            # don't just add to z, but to radial distance
            zadd = zScale*zi
            zadds.append(zadd)
            z[i] = zi + zadd
            #r = np.sqrt(xi**2 + yi**2 + zi**2)
            #r = r + (zScale*r)
            # how does that change z?
            #z[i] = np.sqrt(r**2 - xi**2 - yi**2)

    zadds = np.array(zadds)        
    print("Added bump to %d pnts of %d pnts using rscale %5.2f and zscale %5.2f" % (cnt, len(z), rScale, zScale))
    print("Mean additions: ", np.mean(zadds))
    return z


def zernikePoly(x, y, xOffset, yOffset, coefficients, xMax=-1e22, yMax=-1e22, verbose=False):
    """

    Wrapper around opticspy.interferometer_zenike.__zernikepolar__
    """

    if len(coefficients) > zernikies.nMax + 1:
        raise ValueError('coefficients must have less than {} items.'.format(zernikies.nMax+1))

    if xMax == -1e22:
        xMax = np.nanmax(x - xOffset)
    if yMax == -1e22:
        yMax = np.nanmax(y - yOffset)

    xcn = (x - xOffset)/xMax
    ycn = (y - yOffset)/yMax

    rcn = np.sqrt(xcn**2. + ycn**2.)
    ucn = np.arctan2(xcn, ycn)

    z = zernikies.zernikePolar(coefficients, rcn, ucn)

    if verbose:
        print("Zernike polynomials with coefficients", coefficients)
        print("Their linear combination has mean: {0:.2e}, min: {1:.2e}, max: {2:.2e}".format(np.mean(z), np.nanmin(z), np.nanmax(z)))

    return z

def zernikePolyOpticspy(x, y, xOffset, yOffset, coefficients, verbose=False):
    """

    Wrapper around opticspy.interferometer_zenike.__zernikepolar__
    """

    if len(coefficients) > 38:
        raise ValueError('coefficients must have less than {} items.'.format(zernikies.nMax+1))

    xcn = (x - xOffset)/np.nanmax(x - xOffset)
    ycn = (y - yOffset)/np.nanmax(y - yOffset)

    # Flip x and y when evaluating the radius and angle.
    rcn = np.sqrt(xcn**2. + ycn**2.)
    ucn = np.arctan2(ycn, xcn)

    
    import opticspy
    #z = opticspy.interferometer_zenike.__zernikepolar__(amplitude, rcn, ucn)
    z = opticspy.interferometer_zenike.__zernikepolar__(coefficients, rcn, ucn)

    if verbose:
        print("Zernike polynomials with coefficients", coefficients)
        print("Their linear combination has mean: {0:.2e}, min: {1:.2e}, max: {2:.2e}".format(np.mean(z), np.nanmin(z), np.nanmax(z)))

    return z

def gaussian(x, y, amplitude, xOffset, yOffset, width):
    # https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    print("Applying 2D guassian of amplitude: %f and width: %f" % (amplitude, width))
    xx = (x - xOffset)**2 / 2*(width**2)
    yy = (y - yOffset)**2 / 2*(width**2)
    z = amplitude * np.exp(-(xx + yy))
    print("min: %f, max: %f" % (np.nanmin(z), np.nanmax(z)))
    return z
