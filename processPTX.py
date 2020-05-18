import sys
import os
import random
import warnings
from copy import copy

# import matplotlib
# matplotlib.use("agg")

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianRepresentation
from astropy.coordinates.matrix_utilities import rotation_matrix

from rotate import shiftRotateXYZ
from parabolas import parabola, fitLeicaData
from plotting import scatter3dPlot
from utils.utils import mjd2utc, splitXYZ, aggregateXYZ


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


def rotateXYaboutZ(xyz, rotDegrees):

    # define it as cartesian
    rep = CartesianRepresentation(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    # define our rotation about z
    # TBF: why must we negate this to match Mathematica results?
    rot = rotation_matrix(-rotDegrees * u.deg, axis='z')

    # apply the rotation
    new_rep = rep.transform(rot)

    new_xyz = new_rep.get_xyz().value

    xyzNew = np.c_[new_xyz[0], new_xyz[1], new_xyz[2]]

    return xyzNew 
        
    
def processPTX(fpath, rotationAboutZdegrees=None, searchRadius=None, rFilter=True):


    # import ipdb; ipdb.set_trace() 
    if rotationAboutZdegrees is None:
        rotationAboutZdegrees = 150.
    if searchRadius is None:    
        searchRadius = 47
   
    with open(fpath, 'r') as f:
        ls = f.readlines()
    
    xyz = processPTXdata(ls, rotationAboutZdegrees, searchRadius)

    # write to CSV file
    outf = fpath + ".csv"
    np.savetxt(outf, xyz, delimiter=",")


def processPTXdata(lines, rotationAboutZdegrees, searchRadius, quiet=True, sampleSize=None, preview=True):

    ls = lines

    if not quiet:
        print("Original file num lines: ", len(ls))

    #badLine = '0 0 0 0.500000\r\n'
    badLine = '0 0 0 0.500000\n'
    # reject the '0 0 0 0.500000' lines - TBF: what are those?
    ls = [l for l in ls if l != badLine]
    #for l in ls:
    #    print "'%s' vs '%s'" % (l, badLine)

    if not quiet:
        print("File minus '0 0 0 0.50000' lines: ", len(ls))

    #print "beginning of file: "
    #print ls[:12]

    # remove the header
    # ls = ls[10:]
    ls = ls[11:]
    # import ipdb; ipdb.set_trace()
    #print "now beginning of file w/ out header: "
    #print ls[:12]

    # parse strings so we can ignore the fourth value
    sample = 1000
    xyz = []
    for i, l in enumerate(ls):
        # print i, l
        if "Line" in l:
            # print l
            continue
        ll = l.split(' ')
        x = float(ll[0])
        y = float(ll[1])
        z = float(ll[2])
        xyz.append((x, y, z))

    xyz = np.array(xyz)    

    # print "Skipping rotation and filtering!"
    # return xyz

    # rotation!  This takes a long time.  TBF: why? *)
    #rot=AffineTransform[{RotationMatrix[150Degree,{0,0,1}],{0,0,0}}];
    #lall=Map[rot,lall];
    xyz = rotateXYaboutZ(xyz, rotationAboutZdegrees)

    # and we only use those parts that are within this radius TBF? *)
    # ls=Select[ls,Norm[Take[#,2]-{-54,0}]<47&]
    # Here's what's going on:
    # Select will go through and apply the predicat # < 47 to each element of 'ls'
    # but, in this case each element, #, is passed to Norm[Take[#,2]-{-54, 0}]
    # What's happening there is that the x, y from each element in xyz above is taken,
    # and 54 is added to each x.  Norm == Sqrt(x**2 + y**2); that looks like a radius to me.
    # so, if the radius is less the 47, this data element is kept.


    if preview:
        # a preview of what our processed data looks like
        sampleSize = 10000
        if len(xyz) < sampleSize:
            sampleSize = len(xyz)
        lsIdx = random.sample(range(len(xyz)), sampleSize)
        xyzr = xyz[lsIdx]
        xr, yr, zr = splitXYZ(xyzr)
        xlim = (-100, 100)
        ylim = (-100, 100)
        scatter3dPlot(xr, yr, zr, "Sampling of processed data", xlim=xlim, ylim=ylim)
        f = plt.figure()
        ax = f.gca()
        # take a look at the x y orientation
        ax.plot(xr, yr, 'bo', [0], [0], '*')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("X Y orientation of data")

    return filterOutRadius(xyz, searchRadius=searchRadius)

    
def filterOutRadius(xyz, searchRadius=None, mysteryX=None):
    "return only those xyz points where sqrt(x**2 + y**2) is within a limit"

    if mysteryX is None:
        mysteryX = 54
    if searchRadius is None:    
        searchRadius = 47

    xyzInside = []
    for x, y, z in xyz:
        x2 = x + mysteryX
        r = np.sqrt(x2**2 + y**2)
        # if r < searchRadius and r > 20:
        if r < searchRadius:
        #if z > 0. and z < 60.:
            xyzInside.append((x, y, z))
    xyzInside = np.array(xyzInside)

    return xyzInside


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
    
    :param ls: Lines of the PTX file.
    :param sampleSize: Use a random sample of the lines of this length.
    :returns: x, y, z and intensity values read from the lines of the PTX file.
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
    "creates plots to make sure we are centered"

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
    plt.title("XY centered and below radus: %5.2f" % radius)

def tryEllipticalOffsets(lines, ellipse):
    """
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

    xin, yin, zin, _, _ = ellipticalFilter(x, y, z, ellipse[0], ellipse[1], ellipse[2], ellipse[3], ellipse[4])

    f = plt.figure()
    ax = f.gca()
    ax.plot(x, y, 'ro')
    ax.plot(xin, yin, 'o')
    ax.plot([0], [0], 'y*')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("XY inside ellipse")


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

    :param x: x coordinates.
    :param y: y coordinates.
    :param z: z coordinates.
    :param tol: Distance threshold for filtering.
    :param dts: Date time stamps.
    :param intensity: Intensity values.
    :returns: Filtered x, y, z, dts and intensity.
    """

    r = np.sqrt(np.power(x, 2.) + np.power(y, 2.) + np.power(z, 2.))
    mask = r > tol

    if dts is not None:
        dts = dts[mask]

    if intensity is not None:
        intensity = intensity[mask]

    return x[mask], y[mask], z[mask], dts, intensity


def zLimitFilter(x, y, z, zLimit=-80, dts=None, intensity=None):
    """
    Filter points whose z coordinate is larger than zLimit.

    :param x: x coordinates.
    :param y: y coordinates.
    :param z: z coordinates.
    :param zLimit: Filter values larger than this.
    :param dts: Date time stamps.
    :param intensity: Intensity values.
    :returns: The filtered x, y, z, dts and intensity.
    """

    # z - filter: at this point we should have the
    # dish, but with some things the radial filter didn't
    # get rid of above or below the dish
    mask = np.logical_and(z > zLimit, z < -10)
    x = x[mask]
    y = y[mask]
    z = z[mask]

    if dts is not None:
        dts = dts[mask]

    if intensity is not None:
        intensity = intensity[mask]

    return x, y, z, dts, intensity


def processNewPTXData(lines,
                      xyzi=None,
                      dts=None,
                      plotTest=True,
                      rot=None,
                      sampleSize=None,
                      simSignal=None,
                      iFilter=False,
                      nFilter=True,
                      rFilter=True,
                      addOffset=False,
                      filterClose=True,
                      parabolaFilter=True,
                      ellipse=[-8., 50., 49., 49., 0.],
                      residual_threshold=0.008):
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
        xo, yo, zo, io = getRawXYZ(lines, sampleSize=sampleSize)
    else:
        xo, yo, zo, io = xyzi

    print("Starting with %d lines of data" % len(xo))

    # First remove all the zero data.
    mask = io != 0.0
    intensity = io[mask]
    x = xo[mask]
    y = yo[mask]
    z = zo[mask]
    if dts is not None:
        dts = dts[mask]

    numFilteredOut = len(x) - len(intensity)
    percent = (float(numFilteredOut) / float(len(x))) * 100.
    print("Filtered out %d points of %d (%5.2f%%) intensity equal to zero" % (numFilteredOut, len(x), percent))
    print("Now we have %d lines of data" % len(x))

    # Remove aggregious jumps in data?
    if nFilter:
        # TBF: document where our tolerance comes from
        x, y, z, mask = neighborFilter(x, y, z, 0.122)
        intensity = intensity[mask]
        print("Now we have %d lines of data" % len(x))
        if dts is not None:
            dts = dts[mask]

    # We only want the data that has a decent intesity.
    meanI = np.mean(intensity)
    stdI = np.std(intensity)
    print("Intensity: max=%5.2f, min=%5.2f, mean=%5.2f, std=%5.2f" % (np.max(intensity),
                                                                      np.min(intensity),
                                                                      meanI,
                                                                      stdI))

    if iFilter:    
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

    # We want as much as possible of the dish.
    # Since the dish will be rotated, it will look like an ellipse to the TLS.
    if rFilter:
        orgNum = len(x)
        print('Elliptical fitler parameters:')
        print(ellipse)
        x, y, z, dts, intensity = ellipticalFilter(x, y, z, 
                                                   ellipse[0], ellipse[1], ellipse[2], 
                                                   ellipse[3], ellipse[4], 
                                                   dts=dts, intensity=intensity)
        newNum = len(x)
        print("Elliptical filter removed {0} points outside the ellipse".format(orgNum - newNum))
        print("The ellipse has semi-major axis {0:.2f} m, semi-minor axis {1:.2f} m and angle {2:.2f} degrees".format(ellipse[2], ellipse[3], ellipse[4]))
        print("Now we have %d lines of data" % len(x))

    # The scanner is upside down.
    z = -z

    # Filter points below the dish.
    orgNum = len(z)
    zLimit = -80    
    x, y, z, dts, intensity = zLimitFilter(x, y, z, zLimit=zLimit, dts=dts, intensity=intensity)
    newNum = len(z)
    print("z - limit filtered out {0} points below {1:5.2f} and above -10".format((orgNum - newNum), zLimit))
    print("Now we have {} lines of data".format(len(z)))

    
    if filterClose:
        # Removes points that are closer than 10 m from the scanner.
        tooClose = 10.
        orgNum = len(z)
        x, y, z, dts, intensity = nearFilter(x, y, z, tol=tooClose, dts=dts, intensity=intensity)
        newNum = len(z)
        print("Removed {0:.0f} points closer than {1:.2f} m from the scanner.".format((orgNum - newNum), tooClose))


    if parabolaFilter:

        sampleSize = int(len(x)*0.05)
        lsIdx = random.sample(range(len(x)), sampleSize)
        xs = x[lsIdx]
        ys = y[lsIdx]
        zs = z[lsIdx]

        # Fit a parabola to the data.
        guess = [60, 0, 0, -49, 0, 0]
        fitresult = fitLeicaData(xs, ys, zs, guess, bounds=None, weights=None)
        c = fitresult.x

        # Start from the original data.
        print("Reloaded {} lines of data.".format(len(xo)))
        mask = io != 0.0
        intensity = io[mask]
        x = xo[mask]
        y = yo[mask]
        z = zo[mask]

        x, y, z, mask = neighborFilter(x, y, z, 0.122)
        intensity = intensity[mask]

        z = -z

        # Rotate and shift the range measurements.
        cor = np.hstack((-1*c[1:4],c[4:6],0))
        xr, yr, zr = shiftRotateXYZ(x, y, z, cor)
        # Build a parabola using the best fit parameters.
        zp = parabola(xr, yr, c[0])

        # Compute the residuals between the parabola and the rotated range measurements.
        diff = zr - zp
        # Only keep range measurements whose residuals are less than residual_threshold.
        print("Removing points with residuals larger than {}".format(residual_threshold))
        mask = abs(diff) < residual_threshold
        percent = mask.sum()/len(mask) * 100.
        print("Parabola filter will remove {0:.2f}% of the data.".format(100. - percent))
        print("Keeping {} out of {} points.".format(mask.sum(), len(mask)))

        # Use the rotated range measurements for the last filter.
        xr = xr[mask]
        yr = yr[mask]
        zr = zr[mask]

        x = x[mask]
        y = y[mask]
        z = z[mask]
        intensity = intensity[mask]

        # Only keep range measurements within a circle of radius r.
        r = 51 # m
        xc = np.nanmin(xr) + r
        yc = np.nanmin(yr) + r
        circular_mask = (xr - xc)**2 + (yr - yc)**2 <= r**2.

        x = x[circular_mask]
        y = y[circular_mask]
        z = z[circular_mask]
        intensity = intensity[circular_mask]

        print("Keeping {} points inside a circle of radius {} m and center ({},{}) m.".format(circular_mask.sum(), r, xc, yc))

    
    # Rotate around the z axis since the scanner coordinate system does not match the telescope's.
    if rot is not None or rot != 0.0:
        print("Rotating about Z by %5.2f degrees" % rot)
        xr, yr, zr = shiftRotateXYZ(x, y, z, [0, 0, 0, 0, 0, np.deg2rad(rot)])


    # Create a Nx3 matrix for the coordinates.
    xyz = np.c_[xr, yr, zr]


    if plotTest:
        # we plotted stuff earlier, so let's get
        # a preview of what our processed data looks like
        if sampleSize is None:
            sampleSize = 10000
            lsIdx = random.sample(range(len(xyz)), sampleSize)
            xyzr = xyz[lsIdx]
            xr, yr, zr = splitXYZ(xyzr)
        else:
            xr, yr, zr = splitXYZ(xyz)    
        xlim = (-100, 100)
        ylim = (-100, 100)
        scatter3dPlot(xr, yr, zr, "Sampling of processed data", xlim=xlim, ylim=ylim)
        f = plt.figure()
        ax = f.gca()
        # take a look at the x y orientation
        ax.plot(xr, yr, 'bo', [0], [0], '*')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("X Y orientation of data")

    print("XYZ output of ProcessNewPTXData has {0} lines of data.".format(len(xyz)))
    print("Intensity output of ProcessNewPTXData has {0} lines of data.".format(len(intensity)))

    return xyz, dts, intensity


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
                  addOffset=False,
                  filterClose=True,
                  simSignal=None,
                  ellipse=[-8., 50., 49., 49., 0.]):

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
                                            simSignal=simSignal,
                                            iFilter=iFilter,
                                            rFilter=rFilter,
                                            addOffset=addOffset,
                                            filterClose=filterClose,
                                            ellipse=ellipse)

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

