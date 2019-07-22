import numpy as np

import opticspy

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

def zernikeFour(x, y, xOffset, yOffset, amplitude=None):
    xc = x + xOffset
    yc = y + yOffset
    if amplitude is None:
        amplitude = 0.0017 / (np.sqrt(6) * (np.nanmax(x)**2))
    z = amplitude * np.sqrt(6) * (xc**2 - yc**2)
    print("z4 with amplitude", amplitude)
    print("mean: %f, min: %f, max: %f" % (np.mean(z), np.nanmin(z), np.nanmax(z)))
    return z

def zernikeFive(x, y, xOffset, yOffset, amplitude=1.0, scale=1.0):
    rho = np.sqrt((x+xOffset)**2 + (y+yOffset)**2)
    z5 = amplitude * (np.sqrt(3) * (2*rho**2 - (scale*1)))
    print("z5 with amplitude %f and scale %f produces" % (amplitude, scale))
    print("mean: %f, min: %f, max: %f" % (np.mean(z5), np.nanmin(z5), np.nanmax(z5)))
    return z5

def zernikeSix(x, y, xOffset, yOffset, amplitude=1.0):
    """
    Following opticspy convention:
    https://github.com/Sterncat/opticspy/blob/master/opticspy/interferometer_zenike.py
    """

    xc = x + xOffset
    yc = y + yOffset
    z = amplitude * 2*np.sqrt(6) * xc*yc
    print("z6 with amplitude", amplitude)
    print("mean: %f, min: %f, max: %f" % (np.mean(z), np.nanmin(z), np.nanmax(z)))
    return z

def zernikeSeven(x, y, xOffset, yOffset, amplitude=1.0):
    """
    Following opticspy convention:
    https://github.com/Sterncat/opticspy/blob/master/opticspy/interferometer_zenike.py
    """

    xc = x + xOffset
    yc = y + yOffset
    rho = np.sqrt((xc)**2 + (yc)**2)
    z = amplitude * np.sqrt(8)*yc*(3*rho**2 - 2)
    print("z7 with amplitude", amplitude)
    print("mean: %f, min: %f, max: %f" % (np.mean(z), np.nanmin(z), np.nanmax(z)))
    return z

def zernikeEight(x, y, xOffset, yOffset, amplitude=1.0):
    """
    Following opticspy convention:
    https://github.com/Sterncat/opticspy/blob/master/opticspy/interferometer_zenike.py
    """

    xc = x + xOffset
    yc = y + yOffset
    rho = np.sqrt((xc)**2 + (yc)**2)
    z = amplitude * np.sqrt(8)*xc*(3*rho**2 - 2)
    print("z8 with amplitude", amplitude)
    print("mean: %f, min: %f, max: %f" % (np.mean(z), np.nanmin(z), np.nanmax(z)))
    return z

def zernikeNine(x, y, xOffset, yOffset, amplitude=1.0):
    """
    Following opticspy convention:
    https://github.com/Sterncat/opticspy/blob/master/opticspy/interferometer_zenike.py
    """

    xc = x + xOffset
    yc = y + yOffset
    rho = np.sqrt((xc)**2 + (yc)**2)
    z = amplitude * np.sqrt(8)*yc*(3*xc**2 - yc**2)
    print("z9 with amplitude", amplitude)
    print("mean: %f, min: %f, max: %f" % (np.mean(z), np.nanmin(z), np.nanmax(z)))
    return z

def zernikeTen(x, y, xOffset, yOffset, amplitude=1.0):
    """
    Following opticspy convention:
    https://github.com/Sterncat/opticspy/blob/master/opticspy/interferometer_zenike.py
    """

    xc = x + xOffset
    yc = y + yOffset
    rho = np.sqrt((xc)**2 + (yc)**2)
    z = amplitude * np.sqrt(8)*xc*(xc**2 - 3*yc**2)
    print("z10 with amplitude", amplitude)
    print("mean: %f, min: %f, max: %f" % (np.mean(z), np.nanmin(z), np.nanmax(z)))
    return z

def zernikePoly(x, y, xOffset, yOffset, amplitude=np.zeros(38, dtype=np.float)):
    """

    Wrapper around opticspy.interferometer_zenike.__zernikepolar__
    """

    if len(amplitude) != 38:
        raise ValueError('amplitude must have 38 items.')

    xcn = (x - xOffset)/np.nanmax(x - xOffset)
    ycn = (y - yOffset)/np.nanmax(y - yOffset)

    # Flip x and y when evaluating the radius and angle.
    rcn = np.sqrt(xcn**2 + ycn**2)
    ucn = np.arctan2(ycn, xcn)

    z = opticspy.interferometer_zenike.__zernikepolar__(amplitude, rcn, ucn)

    print("Zernike polynomials with amplitudes", amplitude)
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
