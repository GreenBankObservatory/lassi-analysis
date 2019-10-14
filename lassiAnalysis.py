"""
This is a very high level module for integrating other modules and 
analyzing a single scan of Leica data once it's available.
"""

import os
import time
import numpy as np
from copy import copy
from shutil import copyfile
from scipy.interpolate import griddata
# import matplotlib
# matplotlib.use('agg')

import opticspy

from astropy.stats import sigma_clip

from processPTX import processPTX, processNewPTX, aggregateXYZ
from main import smoothGPUs, smoothXYZGpu, splitXYZ
from main import loadLeicaDataFromGpus
from parabolas import fitLeicaScan, imagePlot, surface3dPlot, radialReplace, loadLeicaData, fitLeicaData, \
                      newParabola, rotateData, parabola
from zernikeIndexing import noll2asAnsi, printZs
from simulateSignal import addCenterBump, zernikeFour
from simulateSignal import zernikeFive, gaussian
from plotting import sampleXYZData, scatter3dPlot
from utils import sph2cart, cart2sph, log, difflog
from weightSmooth import weightSmooth
import settings

# where is the code we'll be running?
# GPU_PATH = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing"
GPU_PATH = settings.GPU_PATH

def smooth(fpath, N=512, spherical=False):

    assert os.path.isfile(fpath)

    # we need to specify the aboslute path of our inputs
    abspath = os.path.abspath(fpath)

    # our output will be same file name, but with appendages
    outfile = os.path.basename(fpath)

    smoothGPUs(GPU_PATH,
               abspath,
               outfile,
               N,
               spherical=spherical)

    # make sure the output is where it should be
    outfiles = []
    for dim in ['x', 'y', 'z']:
        dimFile = "%s.%s.csv" % (outfile, dim)
        dimPath = os.path.join(GPU_PATH, dimFile)
        outfiles.append(dimPath)
        assert os.path.isfile(dimPath)
        print("GPUs created file: ", dimPath)

    return outfiles

def writeGPUoutput(gpuPath, fn, dim, data, cp2ext=None):

    fn = "%s.%s.csv" % (fn, dim)
    fpath = os.path.join(gpuPath, fn)
    # here we make sure each element gets its own line
    # with as much precision as what I think the GPU is doing
    print("writeGPUoutput to ", fpath)
    if cp2ext is not None and os.path.isfile(fpath):
        # avoid overwriting this file by copying to a location with 
        # this extension
        copyfile(fpath, fpath + '.' + cp2ext)

    np.savetxt(fpath, data, fmt="%.18f")
    return fpath

def smoothWithWeights(fpath, xyz, N=512):
    """
    Here we need to smooth twice so that we can determine
    the variance needed to compute our weights.  We also
    need to handle the coordinate transformations ourselves,
    rathern then in the GPUs, 
    since the variance calculations are done in spherical.
    """

    # first just do the normal smoothing, but don't let
    # GPUs do the coordinate transforms

    # convert to shperical
    # s = 1.0
    # print "working with %f percent of data" % s
    x, y, z = splitXYZ(xyz)
    # x, y, z = sampleXYZData(x, y, z, s)

    scatter3dPlot(x, y, z, "org sample of data")

    xOrg = copy(x)
    yOrg = copy(y)
    zOrg = copy(z)

    print("Converting to spherical ...")
    r, el, az = cart2sph(x, y, z)

    scatter3dPlot(el, az, r, "el az r of sample of data")

    print("input data ranges: ")
    print("r: ", np.nanmin(r), np.nanmax(r), len(r))
    print("el: ", np.nanmin(el), np.nanmax(el), len(el))
    print("az: ", np.nanmin(az), np.nanmax(az), len(az))

    sph = aggregateXYZ(r, az, el)

    # write this to the file format expected by the GPUs,
    # using 'sph' to denote the coord system
    assert fpath[-4:] == '.csv'
    outf = fpath[:-4] + ".sph.csv"
    print("Saving spherical to file: ", outf)
    np.savetxt(outf, sph, delimiter=",")    

    # TBF: GPU code is still labeling the output files
    # as [x,y,z] even though we are in spherical
    print("Smoothing R ...")
    smoothSphFiles = smooth(outf, spherical=True)

    # print "Identity Test!"
    # basename = os.path.basename(outf)
    # for data, dim in [(r, 'x'), (az, 'y'), (el, 'z')]:
    #     # write them to files as if the GPUs created them:
    #     fn = writeGPUoutput(GPU_PATH, basename, dim, data)

    # smoothSphFiles = [
    #     "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing/Clean9.ptx.sph.csv.x.csv",
    #     "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing/Clean9.ptx.sph.csv.y.csv",
    #     "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing/Clean9.ptx.sph.csv.z.csv"    
    # ]
 
    basename = os.path.basename(outf)
    gpuSphPath = os.path.join(GPU_PATH, basename)
    print("loading GPU data from", gpuSphPath)

    # retrieve the radial values for the variance calculation
    rs, azs, els = loadLeicaDataFromGpus(gpuSphPath)

    scatter3dPlot(els, azs, rs, "az el r of smoothed data")

    # print "rs vs r: ", rs.shape, r.shape, rs, r
    # assert np.all(rs == r)
    # assert np.all(azs == az)
    # assert np.all(els == el)

    # print "smoothed data ranges: "
    # print "r: ", np.nanmin(r), np.nanmax(r)
    # print "el: ", np.nanmin(el), np.nanmax(el)
    # print "az: ", np.nanmin(az), np.nanmax(az)

    # convert these back to cartesian, since this is what
    # we'll need for the subsequent fittings
    print("converting this spherical data to xyz ...")
    xs, ys, zs = sph2cart(els, azs, rs)

    scatter3dPlot(xs, ys, zs, "smoothed xyz data")

    smoothedFiles = []
    for data, dim, ext in [(xs, 'x', 'r'), (ys, 'y', 'az'), (zs, 'z', 'el')]:
        # write them to files as if the GPUs created them:
        fn = writeGPUoutput(GPU_PATH, basename, dim, data, cp2ext=ext)
        smoothedFiles.append(fn)
    print("cartesian smoothed files: ", smoothedFiles)    

    # print "Identity Test Again!"
    # print "xs vs xOrg", xs.shape, xOrg.shape, xs, xOrg

    # tol = 1e-3
    # assert np.all(np.abs((xs - xOrg)) < tol)
    # assert np.all(np.abs((ys - yOrg)) < tol)
    # assert np.all(np.abs((zs - zOrg)) < tol)
    # assert np.all(xs == xOrg)
    # assert np.all(ys == yOrg)
    # assert np.all(zs == zOrg)

    # fake the weights
    # weights = np.zeros((N, N))
    # weights = weights + 1
    # return smoothedFiles, weights

    # now we need to smooth again, but this time using
    # our radial values SQUARED.  So write this to file
    sph2 = aggregateXYZ(r**2, az, el)
    outf2 = fpath + ".sph2.csv"
    np.savetxt(outf2, sph2, delimiter=",")    

    scatter3dPlot(el, az, r**2, "az el r^2")

    print("Smoothing R^2 ...")
    smoothSphFiles2 = smooth(outf2, spherical=True)
    # smoothSphFiles = [
    #     "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing/Clean9.ptx.sph2.csv.x.csv",
    #     "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing/Clean9.ptx.sph2.csv.y.csv",
    #     "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing/Clean9.ptx.sph2.csv.z.csv"    
    # ]

    # retrieve the radial values for the variance calculation
    basename2 = os.path.basename(outf2)
    gpuSphPath2 = os.path.join(GPU_PATH, basename2)
    print("loading GPU data from", gpuSphPath2)

    r2s, azs, els = loadLeicaDataFromGpus(gpuSphPath2)

    scatter3dPlot(els, azs, r2s, "az el r2 of smoothed data")

    # now we can calculate the variance!
    # make sure small negative numerical errors are dealt with
    sigma2 = np.abs(r2s - (rs**2))

    print("sigma squared: ", sigma2.shape, np.nanmin(sigma2), np.nanmax(sigma2), np.nanmean(sigma2), np.nanstd(sigma2))

    # not normalized weights
    Ws_Not_Norm= 1/sigma2

    # make sure NaNs induced by zeros in sigma2 are dealt with
    Ws_Not_Norm[sigma2 == 0.0] = 0.0

    # normalize weights, making sure Nans in sum are dealt with
    ws = (Ws_Not_Norm) / np.sum(Ws_Not_Norm[np.logical_not(np.isnan(sigma2))])

    print("Computed weights: ", ws.shape, ws)

    sigma2.shape = (N, N)
    imagePlot(sigma2, "sigma^2")
    wsc = copy(ws)
    wsc.shape = (N, N)
    imagePlot(wsc, "weights")

    return smoothedFiles, ws

def OLDsmoothWithWeights(fpath, N=512):
    """
    Here we need to smooth twice so that we can determine
    the variance needed to compute our weights.
    """

    # our first smoothing is what we usually do,
    # but we need to leave result in spherical coordinates
    # so we can calculate the variance
    smoothedFiles = smooth(fpath, N=N, leaveInSpherical=True)

    # these files are what we'll be returning, but we
    # also need to read them in so we can compute the variance
    fn = smoothedFiles[0]
    assert fn[-5:] == 'x.csv'
    fn = fn[:-6]
    rs, azs, els = loadLeicaDataFromGPUs(fn)

    # OK, now we need to smooth the r^2 data, so we'll be doing

    return smoothedFiles

def processLeicaScan2(fpath,
                     N=512,
                     rot=None,
                     xOffset=None,
                     yOffset=None,
                     sampleSize=None,
                     parabolaFit=None,
                     useFittingWeights=False,
                     simSignal=None):
    """
    High level function for processing leica data:
       * processes PTX file
       * smoothes it by calling gpu code
       * fits parabolas to data
       * regrids final data
    Final processed scan is ready for difference between
    this and a ref or signal scan.   
    """

    assert os.path.isfile(fpath)

    fileBasename = os.path.basename(fpath)

    # we'll provide primitive timing reports
    s = time.time()

    # removes headers, does basic rotations, etc.
    if xOffset is None:
        xOffset = -8.
        #xOffset = -6.
    if yOffset is None:
        yOffset = 50.0
    if rot is None:
        rot = 0.
    
    processedPath = "{}/{}.csv".format(GPU_PATH, fileBasename)
    
    if False:
        print("Processing PTX file ...")
        xyz = processNewPTX(fpath,
                    rot=rot,
                    xOffset=xOffset,
                    yOffset=yOffset,
                    rFilter=True,
                    iFilter=False,
                    parabolaFit=parabolaFit,
                    simSignal=simSignal,
                    sampleSize=sampleSize) #xOffset=xOffset, yOffset=yOffset)

    e = time.time()
    print("Elapsed minutes: %5.2f" % ((e - s) / 60.))

    print("Masking data and fitting parabola.")
    s = time.time()

    maskedData = maskLeicaData(processedPath, n=N)

    diff = maskedData['fitResidual']
    xRot, yRot, zRot = maskedData['rotated']

    e = time.time()
    print("Elapsed minutes: %5.2f" % ((e - s) / 60.))

    print("Regriding data to a uniformly sampled grid.")
    s = time.time()

    xReg,yReg,diffReg = regridXYZ(xRot, 
                                  yRot, 
                                  diff.filled(np.nan), 
                                  n=N, verbose=False)
    
    _,_,retroMask = regridXYZ(xRot, 
                              yRot, 
                              np.ma.masked_where(diff.mask, diff.mask.astype(float)).filled(np.nan), 
                              n=N, verbose=False)

    diffRegMasked = np.ma.masked_where(retroMask, diffReg)

    e = time.time()
    print("Elapsed minutes: %5.2f" % ((e - s) / 60.))

    # Write the final results to disk.
    finalFile = "%s.processedMasked" % fileBasename
    print("Saving {} to disk".format(finalFile))
    np.savez(finalFile, xs=xReg, ys=yReg, diffs=diffRegMasked)

    return xReg, yReg, diffRegMasked

def processLeicaScan(fpath,
                     N=512,
                     rot=None,
                     xOffset=None,
                     yOffset=None,
                     sampleSize=None,
                     parabolaFit=None,
                     useFittingWeights=False,
                     simSignal=None):
    """
    High level function for processing leica data:
       * processes PTX file
       * smoothes it by calling gpu code
       * fits parabolas to data
       * regrids final data
    Final processed scan is ready for difference between
    this and a ref or signal scan.   
    """
    
    assert os.path.isfile(fpath)

    fileBasename = os.path.basename(fpath)

    # we'll provide primitive timing reports
    s = time.time()

    # removes headers, does basic rotations, etc.
    print("Processing PTX file ...")
    if xOffset is None:
        xOffset = -8.
        #xOffset = -6.
    if yOffset is None:    
        yOffset = 50.0
    if rot is None:
        rot = 0.
    processedPath = fpath + ".csv"
    # if useFittingWeights:    
    #     # TBF: for now read the data from the CSV file
    #     print "Skipping processing and loading previous values"
    #     xyz = np.loadtxt(processedPath, delimiter=',')
    # else:
    if True:
        xyz = processNewPTX(fpath,
                    rot=rot,
                    xOffset=xOffset,
                    yOffset=yOffset,
                    rFilter=True,
                    iFilter=False,
                    parabolaFit=parabolaFit,
                    simSignal=simSignal,
                    sampleSize=sampleSize) #xOffset=xOffset, yOffset=yOffset)

    e = time.time()
    print("Elapsed minutes: %5.2f" % ((e - s) / 60.))

    # reduces our data via GPUs
    s = time.time()
    print("Smoothing data ...")

    weights = None
    if useFittingWeights:
        # smoothedFiles, weights = smoothWithWeights(processedPath, xyz, N=N)
        smoothedFiles, weights = weightSmooth(processedPath, xyz)
    else:  
        smoothedFiles = smooth(processedPath, N=N)

    #   fn = smoothedFiles[0]
    #   assert fn[-5:] == 'x.csv'
    #   fn = fn[:-6]
    #   weights = getWeightsFromInitialSmoothing(fn, processedPath)

    e = time.time()
    print("Elapsed minutes: %5.2f" % ((e - s) / 60.))
    
    # now fit this data to a rotated parabola.
    # this function just takes the first part of our outputs
    # and figures out the rest
    s = time.time()
    print("Fitting paraboloa ...")
    fn = smoothedFiles[0]
    assert fn[-5:] == 'x.csv'
    fn = fn[:-6]
    print("Fitting data found in files:", fn)
    # print "NOT USING WEIGHTS!"
    wc = copy(weights)
    # weights = None
    diff, x, y = fitLeicaScan(fn,
                              numpy=False,
                              N=N,
                              rFilter=False,
                              # inSpherical=useFittingWeights,
                              weights=weights)

    e = time.time()
    print("Elapsed minutes: %5.2f" % ((e - s) / 60.))

    s = time.time()
    print("Regriding data ...")
    filename = "%s.regrid" % fileBasename
    xs, ys, diffs = smoothXYZGpu(x, y, diff, N, filename=filename)

    e = time.time()
    print("Elapsed minutes: %5.2f" % ((e - s) / 60.))

    xs.shape = ys.shape = diffs.shape = (N, N)

    imagePlot(diffs, "Regridded Diff")

    diffsLog = np.log(np.abs(np.diff(diffs)))
    imagePlot(diffsLog, "Regridded Diff Log")
    
    # write these final results to disk
    finalFile = "%s.processed" % fileBasename
    np.savez(finalFile, xs=xs, ys=ys, diffs=diffs)

    return xs, ys, diffs, wc

def loadProcessedData(filename):
    "Loads the results of processLeicaScan from file"
    d = np.load(filename)
    return d["xs"], d["ys"], d["diffs"]

def maskLeicaData(filename, n=512, **kwargs):
    """
    Given a GPU smoothed file, it will try and find bumps in the surface and mask them.

    :param filename: file with the GPU smoothed data.
    :param n: number of samples in the GPU smoothed data. Default n=512.
    :param kwargs: keyword arguments passed to astropy.stats.sigma_clip.
    """

    orgData, cleanData = loadLeicaData(filename, n=n, numpy=False)
    
    xf = cleanData[0]
    yf = cleanData[1]
    zf = cleanData[2]

    # Assemble initial guess.
    f = 60.
    v1x = 0.
    v1y = 0.
    v2 = 0.
    xTheta = 0.
    yTheta = 0.
    guess = [f, v1x, v1y, v2, xTheta, yTheta]

    # Fit a parabola to the data.
    fitresult = fitLeicaData(xf, yf, zf, guess, weights=None)
    
    # Subtract the fitted parabola from the data.
    # The difference should be flat.
    c = fitresult.x
    newX, newY, newZ = newParabola(orgData[0], orgData[1], orgData[2], c[0], c[1], c[2], c[3], c[4], c[5])
    newX.shape = newY.shape = newZ.shape = (n, n)
    xrr, yrr, zrr = rotateData(orgData[0], orgData[1], orgData[2], c[4], c[5])
    xrr.shape = yrr.shape = zrr.shape = (n, n)
    diff = zrr - newZ
    
    mask = (((xrr - 2.)**2 + (yrr - 50.)**2) < 45**2)
    mdiff = np.ma.masked_where(~mask, diff)
    # Mask any pixels which deviate from the noise.
    # This should mask out retroreflectors, misaligned panels and sub-scans where the TLS moved due to wind.
    mcdiff = sigma_clip(mdiff)
    
    # Apply the mask to the original data and repeat once more.
    # In the end also fit a parabola to each row in the data, and mask outliers.
    # This allows to find more subtle features in the data.
    xf = np.ma.masked_where(mcdiff.mask, orgData[0])
    yf = np.ma.masked_where(mcdiff.mask, orgData[1])
    zf = np.ma.masked_where(mcdiff.mask, orgData[2])

    f = 60.
    v1x = 0.
    v1y = 0.
    v2 = 0.
    xTheta = 0.
    yTheta = 0.
    guess = [f, v1x, v1y, v2, xTheta, yTheta]

    masked_fitresult = fitLeicaData(xf.compressed(), yf.compressed(), zf.compressed(),
                                       guess, weights=None)
    
    c = masked_fitresult.x
    newXm, newYm, newZm = newParabola(orgData[0], orgData[1], orgData[2], c[0], c[1], c[2], c[3], c[4], c[5])
    newXm.shape = newYm.shape = newZm.shape = (n, n)
    xrrm, yrrm, zrrm = rotateData(orgData[0], orgData[1], orgData[2], c[4], c[5])
    xrrm.shape = yrrm.shape = zrrm.shape = (n, n)
    masked_diff = zrrm - newZm
    mask = (((xrr - 2.)**2 + (yrr - 50.)**2) < 45**2)
    masked_diff = np.ma.masked_where(~mask, masked_diff)
    mcdiff2 = masked_diff
    
    # Final mask.
    map_mask = np.zeros((n,n), dtype=bool)

    x = np.linspace(0,n,n)

    # Loop over rows fitting a parabola and masking any pixels that deviate from noise.
    for i in range(0,n):

        y = mcdiff2[i]

        if len(x[~y.mask]) > 3:

            poly_c = np.polyfit(x[~y.mask], y[~y.mask], 2)
            poly_f = np.poly1d(poly_c)

            res = np.ma.masked_invalid(y - poly_f(x))
            res_sc = sigma_clip(res, **kwargs)

            map_mask[i] = res_sc.mask

        else:

            map_mask[i] = True

    # Prepare output
    origMaskedData = (np.ma.masked_where(map_mask, orgData[0]),
                      np.ma.masked_where(map_mask, orgData[1]),
                      np.ma.masked_where(map_mask, orgData[2]))

    rotatedData = (xrrm, yrrm, zrrm)

    #fitResidual = np.ma.masked_where(map_mask, orgData[2] - newZm)
    fitResidual = np.ma.masked_where(map_mask, zrrm - newZm)

    parabolaFit = (newXm, newYm, newZm) 

    outData = {'origMasked': origMaskedData,
               'rotated': rotatedData,
               'fitResidual': fitResidual,
               'parabolaFit': parabolaFit,
               'parabolaFitCoeffs': c}

    return outData 

def processLeicaScanPair(filename1,
                         filename2,
                         processed=False,
                         rot=None,
                         rFilter=False,
                         parabolaFit=None,
                         fitZernikies=True):
    "Process each scan, find diffs, and fit for zernikies"

    if processed:
        # load results from file; these files should exist
        # here in the CWD
        fn1 = "%s.processed.npz" % os.path.basename(filename1)
        print("Loading processed data from file:", fn1)
        xs1, ys1, diff1 = loadProcessedData(fn1)
        fn2 = "%s.processed.npz" % os.path.basename(filename2)
        print("Loading processed data from file:", fn2)
        xs2, ys2, diff2 = loadProcessedData(fn2)
        imagePlot(np.log(np.abs(np.diff(diff1))), "first scan log")
        imagePlot(np.log(np.abs(np.diff(diff2))), "second scan log")

    else:
        # we need to process each scan
        # WARNING: each scan takes about 10 minutes    
        xs1, ys1, diff1, weights1 = processLeicaScan(filename1, rot=rot, parabolaFit=parabolaFit)
        xs2, ys2, diff2, weights2 = processLeicaScan(filename2, rot=rot, parabolaFit=parabolaFit)

    print("Finding difference between scans ...")
    N = 512
    diffData = diff1 - diff2

    if rFilter:
        # TBF: which x, y to use?
        print("xs1 dims", np.min(xs1), np.max(xs1), np.min(xs1) + ((np.max(xs1) - np.min(xs1))/2))
        print("ys1 dims", np.min(ys1), np.max(ys1), np.min(ys1) + ((np.max(ys1) - np.min(ys1))/2))
        print("xs2 dims", np.min(xs2), np.max(xs2), np.min(xs2) + ((np.max(xs2) - np.min(xs2))/2))
        print("ys2 dims", np.min(ys2), np.max(ys2), np.min(ys2) + ((np.max(ys2) - np.min(ys2))/2))
        rLimit = 45.5
        xOffset = np.min(xs1) + ((np.max(xs1) - np.min(xs1))/2)
        yOffset = np.min(ys1) + ((np.max(ys1) - np.min(ys1))/2)
        print("Center (%f, %f), Removing points close to edge: radius=%f" % (xOffset, yOffset, rLimit))
        diffData = radialReplace(xs1.flatten(),
                                 ys1.flatten(),
                                 diffData.flatten(),
                                 xOffset,
                                 yOffset, 
                                 rLimit,
                                 np.nan)
        diffData.shape= (N, N)
        

    # find the difference of the difference!
    diffData.shape = (N, N)
    imagePlot(diffData, "Surface Deformations")
    diffDataLog = np.log(np.abs(diffData))
    imagePlot(diffDataLog, "Surface Deformations Log")

    print("Mean of diffs: ", np.nanmean(diffData))
    print("Std of diffs: ", np.nanstd(diffData))

    # find the zernike
    if not fitZernikies:
        return (xs1, ys1, xs2, ys2), diffData

    print("Fitting difference to zernikies ...")

    # replace NaNs with zeros
    diffDataOrg = copy(diffData)
    diffData[np.isnan(diffData)] = 0.

    # print scaling up data for z-fit by 1000.
    diffDataM = diffData * 1000.

    # find the first 12 Zernike terms
    numZsFit = 36
    fitlist,C1 = opticspy.zernike.fitting(diffDataM,
                                          numZsFit,
                                          remain2D=1,
                                          barchart=1)
    print("fitlist: ", fitlist)
    C1.listcoefficient()
    C1.zernikemap()

    print("Converting from Noll to Active Surface ANSI Zernikies ...")
    # and now convert this to active surface zernike convention
    # why does the fitlist start with a zero? for Z0??  Anyways, avoid it
    nollZs = fitlist[1:(numZsFit+1)]
    asAnsiZs = noll2asAnsi(nollZs)
    print("nolZs")
    printZs(nollZs)
    print("active surface Zs")
    printZs(asAnsiZs)

    return (xs1, ys1, xs2, ys2), diffData

def regridXYZ(x, y, z, n=512., verbose=False, xmin=False, xmax=False, ymin=False, ymax=False):
    """
    Regrids the XYZ data to a regularly sampled grid.

    :param x: vector with the x coordinates.
    :param y: vector with the y coordinates.
    :param z: vector with the z coordinates.
    :param n: number of samples in the grid.
    :param verbose: verbose output?
    """
    
    # Set the grid limits.
    if not xmin:
        xmin = np.nanmin(x)
    if not xmax:
        xmax = np.nanmax(x)
    if not ymin:
        ymin = np.nanmin(y)
    if not ymax:
        ymax = np.nanmax(y)

    if verbose:
        print("Limits: ", xmin, xmax, ymin, ymax)

    # Set the grid spacing.
    dx = (xmax - xmin)/n
    dy = (ymax - ymin)/n
    
    # Make the grid.
    grid_xy = np.mgrid[xmin:xmax:dx,
                       ymin:ymax:dy]
    if verbose:
        print("New grid shape: ", grid_xy[0].shape)

    # Regrid the data.
    reg_z = griddata(np.array([x[~np.isnan(z)].flatten(),y[~np.isnan(z)].flatten()]).T, 
                     z[~np.isnan(z)].flatten(), 
                     (grid_xy[0], grid_xy[1]), method='linear', fill_value=np.nan)
    
    return grid_xy[0], grid_xy[1], reg_z.T

def simulateSignal(sigFn,
                   refFn,
                   sigType,
                   rScale=10., # for use with bump
                   zScale=1.0, # for use with bump
                   xOffset=-8., # for use with z's and guassian
                   yOffset=53., # for use with z's and guassian
                   zAmplitude=None, # for use with zernikes
                   gAmplitude=.0017, # for use with gaussian
                   gWidth=0.2, # for use with guassian
                   ):
    """
    Like processLeicaScanPair, but instead we are 
    injecting a simulated signal into the signal
    file data after the smoothing stage.
    The reference scan we will use processed already.
    """

    N = 512

    # use the name of the signal scan to determine
    # where to find the smoothed results
    # GPU_PATH = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing"
    sigFn2 = "%s.ptx.csv" % sigFn
    sigPath = os.path.join(GPU_PATH, sigFn2)

    # and fit them
    diffSig, xSig, ySig = fitLeicaScan(sigPath,
                                       numpy=False,
                                       N=N,
                                       rFilter=False)

    # add a simulated signal
    if sigType == 'bump':
        from copy import copy
        diffSigorg = copy(diffSig)
        #from processPTX import addCenterBump
        diffSigS = addCenterBump(xSig.flatten(),
                                 ySig.flatten(),
                                 diffSig.flatten(),
                                 rScale=rScale,
                                 zScale=zScale)
        # print x18.shape, y18.shape, diff18.shape, diff18b1.shape
        diffSigS.shape = diffSig.shape

        imagePlot(diffSigS, 'diffSigsS (added bump)')
        imagePlot(np.log(np.abs(np.diff(diffSigS))), 'diffSigS log (added bump)')
        # scatter3dPlot(x18, y18, diff18b1, 'diff18b1', sample=10.)        
    elif sigType == 'z5':

        z5 = zernikeFive(xSig,
                         ySig,
                         -xOffset, #0., #-8,
                         -yOffset, #0., #53.,
                         amplitude=1e-4,
                         scale=1.)
        imagePlot(z5, 'z5')
        surface3dPlot(xSig, ySig, z5, 'z5')
        
        diffSigS = diffSig + z5

        imagePlot(diffSigS, 'diffSigsS (added z5)')
        imagePlot(np.log(np.abs(np.diff(diffSigS))), 'diffSigS log (added z5)')

    elif sigType == 'z4':

        z4 = zernikeFour(xSig,
                         ySig,
                         -xOffset, #0., #-8,
                         -yOffset,
                         amplitude=zAmplitude)
        imagePlot(z4, 'z4')
        surface3dPlot(xSig, ySig, z4, 'z4')
        
        diffSigS = diffSig + z4

        imagePlot(diffSigS, 'diffSigsS (added z4)')
        imagePlot(np.log(np.abs(np.diff(diffSigS))), 'diffSigS log (added z4)')

    elif sigType == 'gaussian':

        zg = gaussian(xSig,
                      ySig,
                      gAmplitude,
                      xOffset,
                      yOffset,
                      gWidth)

        imagePlot(zg, 'gaussian')
        surface3dPlot(xSig, ySig, zg, 'gaussian')

        # add it to the signal
        diffSigS = diffSig + zg

        imagePlot(diffSigS, 'diffSigsS (added gaussian)')
        imagePlot(np.log(np.abs(np.diff(diffSigS))), 'diffSigS log (added gaussian)')

    else:
        print("sigType not recognized: ", sigType)
        diffSigS = diffSig

    # regrid to get into evenly spaced x y
    regridFn = "%s.%s.regridded" % (sigFn, sigType)
    xSigr, ySigr, diffsSigS = smoothXYZGpu(xSig,
                                           ySig,
                                           diffSigS,
                                           N,
                                           filename=regridFn)

    diffsSigS.shape = (N, N)
    imagePlot(difflog(diffsSigS), 'sig + sim regridded')

    
    # now load the reference scan
    from lassiAnalysis import loadProcessedData
    refFn2 = "%s.ptx.processed.npz" % refFn
    xsRef, ysRef, diffsRef = loadProcessedData(refFn2)
    imagePlot(difflog(diffsRef), "%s: ref scan" % refFn)
    
    diffsRef.shape = (N, N)
    diffData = diffsSigS - diffsRef
    diffDataLog = np.log(np.abs(diffData))
    imagePlot(diffData, "Surface Deformations")    
    imagePlot(diffDataLog, "Surface Deformations Log")

    print("Mean: ", np.nanmean(diffData))
    print("Std; ", np.nanstd(diffData))
    
    return diffData

def main():
    fpath = "data/Baseline_STA10_HIGH_METERS.ptx"
    processLeicaScan(fpath)

if __name__ == "__main__":
    main()

