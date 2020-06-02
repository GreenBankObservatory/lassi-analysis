"""
Functions to process the PTX files written by the Leica ScanStation P40.
"""

import sys
import os
import random
import warnings
import numpy as np
from copy import copy

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from astropy.stats import sigma_clip

from rotate import shiftRotateXYZ, align2Paraboloid
from plotting import scatter3dPlot
from utils.utils import mjd2utc, splitXYZ, radialMask
from parabolas import paraboloid, fitLeicaData, paraboloidFit, jacParaboloidZ, costParaboloidZ


def previewData(ptxFile, sample=None):
    """
    Loads a PTX file and displays its contents as a 3D scatter plot.

    :param ptxFile: PTX file to display.
    :param sample: Percentage of the data to display.
    """

    print("opening file", ptxFile)
    with open(ptxFile, 'r') as f:
        ls = f.readlines()

    print("reading data ...")
    x, y, z, i = getRawXYZ(ls)
    
    print("plotting preview ...")
    if sample is None:
        sample = 1.0
    scatter3dPlot(x, y, z, "preview", sample=sample)


def previewEllipticalFilter(PTXFile, ellipse):
    """
    Loads the PTX data and shows it along with the data 
    that would be filtered by the elliptical filter.
    """

    print("Opening PTX file", PTXFile)
    with open(PTXFile, 'r') as f:
        ls = f.readlines()

    tryEllipticalOffsets(ls, ellipse)
 

def getRawXYZsample(fn, sampleRate):
    with open(fn, 'r') as f:
        ls = f.readlines()

    # skip the header
    ls = ls[11:]

    xs = []
    ys = []
    zs = []

    for i, l in enumerate(ls):
        # print i, l
        if "Line" in l:
            # print l
            continue
        # TBF: just take a sample
        if i % sampleRate != 0:
            continue    
        ll = l.split(' ')
        x = float(ll[0])
        y = float(ll[1])
        z = float(ll[2])
        xs.append(x)
        ys.append(y)
        zs.append(z)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    return xs, ys, zs


def getRawXYZ(ls, sampleSize=None):
    """
    Given the lines of a PTX file it extracts x, y, z and intensity values.
    
    Parameters
    ----------
    ls :  
        Lines of the PTX file.
    sampleSize : int, optional
        Use a random sample of the lines of this length.
    
    Returns
    -------
    tuple
        x, y, z and intensity values read from the lines of the PTX file.
    """

    # Skip the header.
    ls = ls[10:]

    numLines = len(ls)

    if sampleSize is not None:
        print("Picking %d random data points from a total of %d" % (sampleSize, numLines))
        lsIdx = random.sample(range(numLines), sampleSize)
    else:
        lsIdx = range(numLines)

    xs = []
    ys = []
    zs = []
    it = []

    numLines = 0

    for i in lsIdx:
        l = ls[i]

        if "Line" in l:
            numLines += 1
            continue

        ll = l.split(' ')
        try:
            x = float(ll[0])
            y = float(ll[1])
            z = float(ll[2])
            i = float(ll[3])
            xs.append(x)
            ys.append(y)
            zs.append(z)
            it.append(i)
        except IndexError:
            print("Line {0} is missing columns.".format(i))
            continue

    print("Skipped %d non-data lines" % numLines)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    it = np.array(it)

    return xs, ys, zs, it


def tryOffsets(lines, xOffset, yOffset, radius):
    """
    Creates plots of the scan and the location of the circular filter.
    This is used to determine if the circular filter captures the primary
    reflector.
    """

    sampleSize = 10000
    x, y, z, _ = getRawXYZ(lines, sampleSize=sampleSize)

    # plot the random sample of xyz data
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Random sample of %d points" % sampleSize)

    # plot the raw data including where the origin is
    f = plt.figure()
    ax = f.gca()
    ax.plot(x, y, 'bo', [0], [0], '*')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("X Y orientation of data")

    # translate data
    xo = x - xOffset
    yo = y - yOffset

    xr = []
    yr = []
    xro = []
    yro = []
    for i in range(len(xo)):
        thisX = xo[i]
        thisY = yo[i]
        r = np.sqrt(thisX**2 + thisY**2)
        if r < radius:
            xr.append(thisX)
            yr.append(thisY)
        else:
            xro.append(thisX)
            yro.append(thisY)
    xr = np.array(xr)
    yr = np.array(yr)
    xro = np.array(xro)
    yro = np.array(yro)

    f = plt.figure()
    ax = f.gca()
    ax.plot(xr, yr, 'o', xro, yro, 'ro', [0], [0], 'r*')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("XY centered and below radius: %5.2f" % radius)


def tryEllipticalOffsets(lines, ellipse):
    """
    Creates plots of the scan and the location of the circular filter.
    This is used to determine if the circular filter captures the primary
    reflector.
    """

    sampleSize = 10000
    x, y, z, _ = getRawXYZ(lines, sampleSize=sampleSize)

    # plot the random sample of xyz data
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Random sample of %d points" % sampleSize)

    # plot the raw data including where the origin is
    f = plt.figure()
    ax = f.gca()
    ax.plot(x, y, 'bo', [0], [0], '*')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data in the XY plane")

    xin, yin, zin, _, _ = ellipticalFilter(x, y, z, ellipse[0], ellipse[1], ellipse[2], ellipse[3], ellipse[4])

    f = plt.figure()
    ax = f.gca()
    ax.plot(x, y, 'ro')
    ax.plot(xin, yin, 'o')
    ax.plot([0], [0], 'y*')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data inside the ellipse")


def radialFilter(x, y, z, xOffset, yOffset, radius, dts=None):
    """
    Removes points outside a circle centered at (xOffset, yOffset).

    :param x: x coordinates.
    :param y: y coordinates.
    :param z: z coordinates.
    :param xOffset: Center of the circle in the x coordinate.
    :param yOffset: Center of the circle in the y coordinate.
    :param radius: Radius of the circle.
    :param dts: Date time stamps.
    :returns: Filtered x, y, z and dts.
    """
    
    mask = np.power(x - xOffset, 2.) + np.power(y - yOffset, 2.) < radius**2.

    if dts is not None:
        dts = dts[mask]

    return x[mask], y[mask], z[mask], dts


def ellipticalFilter(x, y, z, xOffset, yOffset, bMaj, bMin, angle, dts=None, intensity=None):
    """
    Filter points that lie outside an ellipse with 
    a semi-major axis of bMaj, semi-minor axis bMin and rotated by angle.

    :param x: x coordinates.
    :param y: y coordinates.
    :param z: z coordinates.
    :param xOffset: Center of the ellipse in the x coordinate.
    :param yOffset: Center of the ellipse in the y coordinate.
    :param bMaj: Ellipse major axis.
    :param bMin: Ellipse minor axis.
    :param angle: Rotation of the ellipse with respect to the y axis.
    :param dts: Date time stamps.
    :param intensity: Intensity values.
    :returns: Filtered x, y, z, dts and intensity values.
    """

    cos_angle = np.cos(np.radians(180. - angle))
    sin_angle = np.sin(np.radians(180. - angle))

    # Shift the points.
    xc = x - xOffset
    yc = y - yOffset

    # Rotate the points.
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle

    mask = (xct**2./(bMaj)**2.) + (yct**2./(bMin)**2.) <= 1.

    if dts is not None:
        dts = dts[mask]

    if intensity is not None:
        intensity = intensity[mask]

    return x[mask], y[mask], z[mask], dts, intensity


def nearFilter(x, y, z, tol=10., dts=None, intensity=None):
    """
    Filter points that are closer than tol from the TLS.

    Parameters
    ----------
    x : array
        x coordinates.
    y : array
        y coordinates.
    z : array
        z coordinates.
    tol : float, optional
        Distance threshold for filtering.
    dts : array
        Date time stamps.
    intensity : array, optional
        Intensity values.

    Returns
    -------
    tuple
        Filtered x, y, z, dts and intensity.
    """

    r = np.sqrt(np.power(x, 2.) + np.power(y, 2.) + np.power(z, 2.))
    mask = r > tol

    printMaskStats(mask)

    if dts is not None:
        dts = dts[mask]

    if intensity is not None:
        intensity = intensity[mask]

    return x[mask], y[mask], z[mask], dts, intensity


def paraboloidMask(x, y, z, paraFit, threshold=3):
    """
    Given a best fit paraboloid it creates a mask where the elements 
    that are within threshold sigmas from the best fit paraboloid are True.
    Sigma is the standard deviation of the residuals.
    
    Parameters
    ----------
    
    """

    # Rotate and shift the range measurements.
    #cor = np.hstack((-1*paraFit[1:4],paraFit[4:6],0))
    #xsr, ysr, zsr = shiftRotateXYZ(x, y, z, cor)
    xr, yr, zr = align2Paraboloid(x, y, z, paraFit)
    # Build a paraboloid using the best fit parameters.
    zp = paraboloid(xr, yr, paraFit[0])
    # Compute the residuals between the paraboloid and the rotated range measurements.
    diff = zr - zp
    # Mask residuals larger than 1 cm.
    #mask = abs(diff) < 0.01
    # Mask residuals outside threshold sigmas.
    mdiff = sigma_clip(diff, threshold)
    mask = ~mdiff.mask

    return mask


def paraboloidFilter(x, y, z, threshold=3, guess=[60, 0, 0, -49, 0, 0, 0, 0, 0, 0, 50]):
    """
    Fits a deformed paraboloid to the (x,y,z) point cloud and returns an array of booleans 
    with True for elements that are within threshold sigmas from the best fit paraboloid.
    Sigma is the standard deviation of the residuals.
    """
    
    # Fit a paraboloid to the data.
    fitresult = paraboloidFit(x, y, z, guess,
                              method=costParaboloidZ, jac=jacParaboloidZ,
                              ftol=1e-12, xtol=1e-15, f_scale=1e-2)
    paraFit = fitresult.x

    # Create the mask.
    mask = paraboloidMask(x, y, z, paraFit, threshold=threshold)  
    # Tell the user how much data is kept.
    printMaskStats(mask)
 
    return mask, paraFit


def zLimitFilter(x, y, z, zLimit=-80, dts=None, intensity=None):
    """
    Filter points whose z coordinate is larger than zLimit.

    Parameters
    ----------
    x : array
        x coordinates.
    y : array
        y coordinates.
    z : array
        z coordinates.
    zLimit : float, optional
        Filter values larger than this.
    dts : array, optional
        Date time stamps.
    intensity : array, optional
        Intensity values.

    Returns
    -------
    tuple
        The filtered x, y, z, dts and intensity.
    """

    mask = np.logical_and(z > zLimit, z < -10)

    printMaskStats(mask)

    x = x[mask]
    y = y[mask]
    z = z[mask]

    if dts is not None:
        dts = dts[mask]

    if intensity is not None:
        intensity = intensity[mask]

    return x, y, z, dts, intensity


def printMaskStats(mask):
    """
    Prints what percentage of the data will be discarded 
    and how many elements are kep for a given mask.

    Parameters
    ----------
    mask: array of bools
        Array with the mask to be applied, where the 
        elements to be kept are True.
    """

    percent = mask.sum()/len(mask) * 100.
    print("Filter will remove {0:.2f}% of the data.".format(100. - percent))
    print("Keeping {} out of {} points.".format(mask.sum(), len(mask)))


def processNewPTXData(lines,
                      xyzi=None,
                      dts=None,
                      plotTest=True,
                      rot=None,
                      sampleSize=None,
                      iFilter=False,
                      nFilter=True,
                      rFilter=True,
                      filterClose=True,
                      filterParaboloid=True,
                      ellipse=[-8., 50., 49., 49., 0.],
                      residual_threshold=0.008):
    """
    
    """

    "this is the processing we see works with 2019 data"

    if rot is None:
        rot = -90.0
    if ellipse is None:
        warnings.warn("No ellipse given. Will use default values.")
        ellipse=[-8., 50., 49., 49., 0.]

    print("ProcessNewPTXData with: ", ellipse)

    if plotTest and lines is not None:
        # make some plots that ensure how we are doing
        # our radial filtering
        tryEllipticalOffsets(lines, ellipse)

    if lines is not None:
        # Get the actual float values from the file contents.
        x, y, z, i = getRawXYZ(lines, sampleSize=sampleSize)
    else:
        x, y, z, i = xyzi

    print("Starting with %d lines of data" % len(x))

    # First remove all the zero data.
    print("Filtering measurements with zero intensity.")
    mask = i != 0.0
    intensity = i[mask]
    x = x[mask]
    y = y[mask]
    z = z[mask]
    if dts is not None:
        dts = dts[mask]

    printMaskStats(mask)


    # Remove aggregious jumps in data?
    if nFilter:
        print("Neighbor filter.")
        # TBF: document where our tolerance comes from
        x, y, z, mask = neighborFilter(x, y, z, 0.122)
        intensity = intensity[mask]
        print("Now we have %d lines of data" % len(x))
        if dts is not None:
            dts = dts[mask]

    
    # We only want data that has a decent intesity.
    meanI = np.mean(intensity)
    stdI = np.std(intensity)
    print("Intensity: max=%5.2f, min=%5.2f, mean=%5.2f, std=%5.2f" % (np.max(intensity),
                                                                      np.min(intensity),
                                                                      meanI,
                                                                      stdI))

    if False: # iFilter    
        mask = np.logical_and(intensity > 0.75, intensity < 0.85)
        intensity = intensity[mask]
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if dts is not None:
            dts = dts[mask]
        numFilteredOut = len(x) - len(intensity)
        percent = (float(numFilteredOut) / float(len(x))) * 100.
        print("Filtered out %d points of %d (%5.2f%%) via intensity" % (numFilteredOut, len(x), percent))
        print(("Now we have %d lines of data" % len(x)))

    assert len(x) == len(y)
    assert len(y) == len(z)
    assert len(z) == len(intensity)

    # Save the point cloud thus far for the paraboloid filter.
    x_p = copy(x)
    y_p = copy(y)
    z_p = copy(z)
    i_p = copy(intensity)

    # We want as much as possible of the dish.
    # Since the dish will be rotated, it will look like an ellipse to the TLS.
    # This filter is later superseded by the paraboloid filter, but it is
    # necessary to find a best fit paraboloid.
    if rFilter:
        orgNum = len(x)
        print('Elliptical filter.')
        print("The ellipse has semi-major axis {0:.2f} m, semi-minor axis {1:.2f} m and angle {2:.2f} degrees".format(ellipse[2], ellipse[3], ellipse[4]))
        x, y, z, dts, intensity = ellipticalFilter(x, y, z, 
                                                   ellipse[0], ellipse[1], ellipse[2], 
                                                   ellipse[3], ellipse[4], 
                                                   dts=dts, intensity=intensity)
        newNum = len(x)
        print("Elliptical filter removed {0} points outside the ellipse".format(orgNum - newNum))
        print("Now we have %d lines of data" % len(x))

    # The scanner is upside down.
    z = -z

    # Filter points below the dish.
    print("z filter.")
    zLimit = -80    
    x, y, z, dts, intensity = zLimitFilter(x, y, z, zLimit=zLimit, dts=dts, intensity=intensity)

    
    if filterClose:
        tooClose = 10.
        print("Removing points closer than {0:.2f} m from the scanner.".format(tooClose))
        # Removes points that are closer than 10 m from the scanner.
        x, y, z, dts, intensity = nearFilter(x, y, z, tol=tooClose, dts=dts, intensity=intensity)

    
    # Rotate the data for the paraboloid filter.
    print("Rotating about z by {0:5.2f} degrees.".format(rot))
    xr, yr, zr = shiftRotateXYZ(x, y, z, [0, 0, 0, 0, 0, np.deg2rad(rot)])


    if filterParaboloid:

        print("Paraboloid filter.")

        perc = 0.01
        print("Fitting a paraboloid to {}% of the data.".format(perc*100.))
        sampleSize = int(len(xr)*perc)
        lsIdx = random.sample(range(len(xr)), sampleSize)
        xs = xr[lsIdx]
        ys = yr[lsIdx]
        zs = zr[lsIdx]

        # Fit a parabola to the data.
        pMask, _ = paraboloidFilter(xs, ys, zs, threshold=3)

        # Fit the paraboloid again, discarding the already filtered values.
        print("Fitting paraboloid again to the un-filtered values.")
        xs = xs[pMask]
        ys = ys[pMask]
        zs = zs[pMask]
        pMask, paraFit = paraboloidFilter(xs, ys, zs, threshold=3)

        print("Applying paraboloid filter to the data before the elliptical filter.")
        x = x_p
        y = y_p
        z = z_p
        i = i_p
        z = -z
        x, y, z = shiftRotateXYZ(x, y, z, [0, 0, 0, 0, 0, np.deg2rad(rot)])
        print("Number of data points: {}".format(len(z)))

        pMask = paraboloidMask(x, y, z, paraFit, threshold=2)

        print("Applying paraboloid filter to the data.")
        printMaskStats(pMask)

        x = x[pMask]
        y = y[pMask]
        z = z[pMask]
        i = i[pMask]

        # Rotate and shift the range measurements.
        #cor = np.hstack((-1*paraFit[1:4],paraFit[4:6],0))
        #xrp, yrp, zrp = shiftRotateXYZ(x, y, z, cor)
        xrp, yrp, zrp = align2Paraboloid(x, y, z, paraFit)

        # Use a circle to mask data outside of the primary reflector.
        r = 51
        xc = np.mean((np.nanmax(xrp) - r, np.nanmin(xrp) + r))
        yc = np.nanmax(yrp) - r
        rMask = radialMask(xrp, yrp, r, xc=xc, yc=yc)
        
        x = x[rMask]
        y = y[rMask]
        z = z[rMask]
        i = i[rMask]

        print("Applying circular mask to the paraboloid filtered data.")
        printMaskStats(rMask)

        #cor = np.hstack((-1*paraFit[1:4],paraFit[4:6],0))
        #xr, yr, zr = shiftRotateXYZ(x, y, z, cor)
        
        ## Rotate and shift the range measurements.
        #cor = np.hstack((-1*c[1:4],c[4:6],0))
        #xr, yr, zr = shiftRotateXYZ(x, y, z, cor)
        ## Build a parabola using the best fit parameters.
        #zp = paraboloid(xr, yr, c[0])

        ## Compute the residuals between the parabola and the rotated range measurements.
        #diff = zr - zp
        ## Only keep range measurements whose residuals are less than residual_threshold.
        ##print("Removing points with residuals larger than {}".format(residual_threshold))
        ##mask = abs(diff) < residual_threshold
        #mdiff = sigma_clip(diff, 3)
        #mask = ~mdiff.mask
        #percent = mask.sum()/len(mask) * 100.
        #print("Parabola filter will remove {0:.2f}% of the data.".format(100. - percent))
        #print("Keeping {} out of {} points.".format(mask.sum(), len(mask)))

        ## Use the rotated range measurements for the last filter.
        #xr = xr[mask]
        #yr = yr[mask]
        #zr = zr[mask]

        #x = x[mask]
        #y = y[mask]
        #z = z[mask]
        #intensity = intensity[mask]

        ## Only keep range measurements within a circle of radius r.
        #r = 51 # m
        #xc = np.nanmin(xr) + r
        #yc = np.nanmin(yr) + r
        #circular_mask = (xr - xc)**2 + (yr - yc)**2 <= r**2.

        #x = x[circular_mask]
        #y = y[circular_mask]
        #z = z[circular_mask]
        #intensity = intensity[circular_mask]

        #print("Keeping {} points inside a circle of radius {} m and center ({},{}) m.".format(circular_mask.sum(), r, xc, yc))

    
    # Rotate around the z axis since the scanner coordinate system does not match the telescope's.
    #if rot is not None or rot != 0.0:
    #    print("Rotating about z by %5.2f degrees" % rot)
    #    xr, yr, zr = shiftRotateXYZ(x, y, z, [0, 0, 0, 0, 0, np.deg2rad(rot)])


    # Create a Nx3 matrix for the coordinates.
    xyz = np.c_[x, y, z]


    if plotTest:
        print("Plotting.")

        xlim = (-100, 100)
        ylim = (-100, 100)
        fig, ax = scatter3dPlot(xr, yr, zr, "Sampling of processed data", xlim=xlim, ylim=ylim, sample=1)
        
        # take a look at the x y orientation
        f = plt.figure()
        ax = f.gca()
        ax.plot(xr, yr, 'bo', [0], [0], '*')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("X Y orientation of data")

    print("XYZ output of ProcessNewPTXData has {0} lines of data.".format(len(xyz)))
    print("Intensity output of ProcessNewPTXData has {0} lines of data.".format(len(intensity)))

    return xyz, dts, i


def getTimeStamps(fpath):
    "Read in file of form *_times.csv and return MJDs"

    # assuming file of form <path>/<name>.ptx, then our timestamp
    # data should be <path>/<name>_times.csv

    if fpath[-4:] != ".ptx":
        return None

    dtPath = fpath[:-4] + "_times.csv"

    if not os.path.isfile(dtPath):
        return None

    print("loading timestamps from: ", dtPath)
    mjds = np.loadtxt(dtPath)

    # don't do this here because some of this data
    # will later get filtered out most likely
    # print "converting MJDs to datetimes ..."
    # convert to datetimes; do it efficiently though
    # mjd2utcArray = np.frompyfunc(mjd2utc, 1, 1)

    # return mjd2utcArray(mjds)
    return mjds

def processNewPTX(fpath,
                  useTimestamps=False,
                  convertToDatetimes=False,
                  rot=None,
                  sampleSize=None,
                  iFilter=False,
                  rFilter=True,
                  filterClose=True,
                  ellipse=[-8., 50., 49., 49., 0.],
                  plotTest=True):

    # Is there associated time data?
    if useTimestamps:
        dts = getTimeStamps(fpath)
    else:
        dts = None    

    with open(fpath, 'r') as f:
        ls = f.readlines()
    
    xyz, dts, intensity = processNewPTXData(ls,
                                            dts=dts,
                                            rot=rot,
                                            sampleSize=sampleSize,
                                            iFilter=iFilter,
                                            rFilter=rFilter,
                                            filterClose=filterClose,
                                            ellipse=ellipse,
                                            plotTest=plotTest)

    # Write to CSV file.
    outf = fpath + ".csv"
    print("Writing out dish xyz coordinates to:", outf)
    np.savetxt(outf, xyz, delimiter=",")

    if dts is not None:
        outf = fpath + "_times.processed.csv"
        print("Writing out associated MJDs to:", outf)
        np.savetxt(outf, dts, delimiter=",")

    # Converting to datetimes takes a LONG time.
    if dts is not None and convertToDatetimes:
        print("Converting timestamps (MJDs) to datetime ...")
        mjd2utcArray = np.frompyfunc(mjd2utc, 1, 1)
        dts = mjd2utcArray(dts)

    if intensity is not None:
        outf = fpath + "_int.processed.csv"
        print("Writing out associated intensities to:", outf)
        np.savetxt(outf, intensity, delimiter=",")

    return xyz, dts, intensity

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

    cnt = 0
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        if xi > xStart and xi < xEnd and yi > yStart and yi < yEnd:
            cnt +=1 
            # don't just add to z, but to radial distance
            z[i] = zi + (zScale*zi)
            #r = np.sqrt(xi**2 + yi**2 + zi**2)
            #r = r + (zScale*r)
            # how does that change z?
            #z[i] = np.sqrt(r**2 - xi**2 - yi**2)

    print("Added bump to %d pnts of %d pnts using rscale %5.2f and zscale %5.2f" % (cnt, len(z), rScale, zScale))

    return z

def neighborFilter(x, y, z, tol):
    assert len(x) == len(y)
    assert len(y) == len(z)

    orgLen = len(x)

    r = np.sqrt(x**2 + y**2 + z**2)
    rdiff = np.diff(r)
    # TBF: why does this return a tuple?
    rdiffTolIdx = np.where(np.abs(rdiff) > tol)[0]
    # convert these diff indicies to the indicies of the pairs
    rdiffTolIdx2 = copy(rdiffTolIdx)
    rdiffTolIdx2 = rdiffTolIdx2 + 1
    # these are the indicies of the points we want to filter out
    rTolIdx = np.concatenate((rdiffTolIdx, rdiffTolIdx2))

    # convert this to the indicies of the points we want to keep
    rIdx = np.array(range(orgLen))
    badMask = np.isin(rIdx, rTolIdx)
    mask = np.logical_not(badMask)

    xnew = x[mask]
    newLen = len(xnew)
    fLen = orgLen - newLen
    fPcnt = (float(fLen) / float(orgLen)) * 100.0
    print("neighborFilter reduces %d points to %d points (%d filtered, %f %%) using tol: %f" % (orgLen, newLen, fLen, fPcnt, tol))

    # return the mask as well so we can filter out other things as well
    return xnew, y[mask], z[mask], mask

if __name__ == "__main__":
    import os
    dataPath = "/home/sandboxes/jbrandt/Telescope27Mar2019"
    refFn = "Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx"
    fn = os.path.join(dataPath, refFn)

    #fpath = "/home/sandboxes/pmargani/LASSI/data/LeicaDropbox/PTX/Test1_STA14_Bump1_High-02_METERS.ptx"
    #fpath = sys.argv[1]
    #testRotation(90.)
    #fpath = "data/Test1_STA14_Bump1_Hig-02_METERS_SAMPLE.ptx"
    processPTX(fn)

