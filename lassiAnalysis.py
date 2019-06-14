"""
This is a very high level module for integrating other modules and 
analyzing a single scan of Leica data once it's available.
"""
import os
import time
from copy import copy

# import matplotlib
# matplotlib.use('agg')
import numpy as np
import opticspy

from processPTX import processPTX, processNewPTX, aggregateXYZ
from main import smoothGPUs, smoothXYZGpu, splitXYZ, cart2sph, sph2cart
from main import loadLeicaDataFromGpus
from parabolas import fitLeicaScan, imagePlot, surface3dPlot, radialReplace
from zernikeIndexing import noll2asAnsi, printZs
from simulateSignal import addCenterBump, zernikeFour
from simulateSignal import zernikeFive, gaussian
from plotting import sampleXYZData

# where is the code we'll be running?
GPU_PATH = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing"

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
        print "GPUs created file: ", dimPath

    return outfiles

def writeGPUoutput(gpuPath, fn, dim, data):

    fn = "%s.%s.csv" % (fn, dim)
    fpath = os.path.join(gpuPath, fn)
    # here we make sure each element gets its own line
    # with as much precision as what I think the GPU is doing
    print "writeGPUoutput to ", fpath
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
    print "Converting to spherical ..."
    x, y, z = splitXYZ(xyz)
    s = 1.0
    print "working with %f percent of data" % s
    x, y, z = sampleXYZData(x, y, z, s)
    r, el, az = cart2sph(x, y, z)
    sph = aggregateXYZ(r, az, el)

    # write this to the file format expected by the GPUs,
    # using 'sph' to denote the coord system
    assert fpath[-4:] == '.csv'
    outf = fpath[:-4] + ".sph.csv"
    print "Saving spherical to file: ", outf
    np.savetxt(outf, sph, delimiter=",")    

    # TBF: GPU code is still labeling the output files
    # as [x,y,z] even though we are in spherical
    smoothSphFiles = smooth(outf, spherical=True)

    # retrieve the radial values for the variance calculation
    rs, azs, els = loadLeicaDataFromGpus(fpath)

    # convert these back to cartesian, since this is what
    # we'll need for the subsequent fittings
    xs, ys, zs = sph2cart(els, azs, rs)
    smoothedFiles = []
    for data, dim in [(xs, 'x'), (ys, 'y'), (zs, 'z')]:
        fn = writeGPUoutput(GPU_PATH, data, dim)
        smoothedFiles.append(fn)

    # write them to files as if the GPUs created them:

    # now we need to smooth again, but this time using
    # our radial values SQUARED.  So write this to file
    sph2 = aggregateXYZ(r**2, az, el)
    outf2 = fpath + ".sph2.csv"
    np.savetxt(outf2, sph2, delimiter=",")    

    smoothSphFiles2 = smooth(outf2, spherical=True)

    # retrieve the radial values for the variance calculation
    r2s, azs, els = loadLeicaDataFromGpus(fpath)

    # now we can calculate the variance!
    # make sure small negative numerical errors are dealt with
    sigma2 = np.abs(r2s - (rs**2))

    # not normalized weights
    Ws_Not_Norm= 1/sigma2

    # make sure NaNs induced by zeros in sigma2 are dealt with
    Ws_Not_Norm[sigma2 == 0.0] = 0.0

    # normalize weights, making sure Nans in sum are dealt with
    ws = (Ws_Not_Norm) / np.sum(Ws_Not_Norm[np.logical_not(np.isnan(sigma2))])

    return smoothFiles, ws

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

def processLeicaScan(fpath,
                     N=512,
                     rot=None,
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
    print "Processing PTX file ..."
    # xOffset = -8.
    xOffset = -6.
    yOffset = 50.0
    if rot is None:
        rot = 0.
    processedPath = fpath + ".csv"
    if useFittingWeights:    
        # TBF: for now read the data from the CSV file
        print "Skipping processing and loading previous values"
        xyz = np.loadtxt(processedPath, delimiter=',')
    else:
        xyz = processNewPTX(fpath,
                    rot=rot,
                    rFilter=True,
                    iFilter=False,
                    parabolaFit=parabolaFit,
                    simSignal=simSignal,
                    sampleSize=sampleSize) #xOffset=xOffset, yOffset=yOffset)

    e = time.time()
    print "Elapsed minutes: %5.2f" % ((e - s) / 60.)

    # reduces our data via GPUs
    s = time.time()
    print "Smoothing data ..."


    weights = None
    if useFittingWeights:
        smoothedFiles, weights = smoothWithWeights(processedPath, xyz, N=N)
    else:  
        smoothedFiles = smooth(fpath, xyz, N=N)

    #   fn = smoothedFiles[0]
    #   assert fn[-5:] == 'x.csv'
    #   fn = fn[:-6]
    #   weights = getWeightsFromInitialSmoothing(fn, processedPath)

    e = time.time()
    print "Elapsed minutes: %5.2f" % ((e - s) / 60.)
    
    # now fit this data to a rotated parabola.
    # this function just takes the first part of our outputs
    # and figures out the rest
    s = time.time()
    print "Fitting paraboloa ..."
    fn = smoothedFiles[0]
    assert fn[-5:] == 'x.csv'
    fn = fn[:-6]
    diff, x, y = fitLeicaScan(fn,
                              numpy=False,
                              N=N,
                              rFilter=False,
                              inSpherical=useFittingWeights,
                              weights=weights)

    e = time.time()
    print "Elapsed minutes: %5.2f" % ((e - s) / 60.)

    s = time.time()
    print "Regriding data ..."
    filename = "%s.regrid" % fileBasename
    xs, ys, diffs = smoothXYZGpu(x, y, diff, N, filename=filename)

    e = time.time()
    print "Elapsed minutes: %5.2f" % ((e - s) / 60.)

    xs.shape = ys.shape = diffs.shape = (N, N)

    imagePlot(diffs, "Regridded Diff")

    diffsLog = np.log(np.abs(np.diff(diffs)))
    imagePlot(diffsLog, "Regridded Diff Log")
    
    # write these final results to disk
    finalFile = "%s.processed" % fileBasename
    np.savez(finalFile, xs=xs, ys=ys, diffs=diffs)

    return xs, ys, diffs

def loadProcessedData(filename):
    "Loads the results of processLeicaScan from file"
    d = np.load(filename)
    return d["xs"], d["ys"], d["diffs"]

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
        print "Loading processed data from file:", fn1
        xs1, ys1, diff1 = loadProcessedData(fn1)
        fn2 = "%s.processed.npz" % os.path.basename(filename2)
        print "Loading processed data from file:", fn2
        xs2, ys2, diff2 = loadProcessedData(fn2)
        imagePlot(np.log(np.abs(np.diff(diff1))), "first scan log")
        imagePlot(np.log(np.abs(np.diff(diff2))), "second scan log")

    else:
        # we need to process each scan
        # WARNING: each scan takes about 10 minutes    
        xs1, ys1, diff1 = processLeicaScan(filename1, rot=rot, parabolaFit=parabolaFit)
        xs2, ys2, diff2 = processLeicaScan(filename2, rot=rot, parabolaFit=parabolaFit)

    print "Finding difference between scans ..."
    N = 512
    diffData = diff1 - diff2

    if rFilter:
        # TBF: which x, y to use?
        print "xs1 dims", np.min(xs1), np.max(xs1), np.min(xs1) + ((np.max(xs1) - np.min(xs1))/2)
        print "ys1 dims", np.min(ys1), np.max(ys1), np.min(ys1) + ((np.max(ys1) - np.min(ys1))/2)
        print "xs2 dims", np.min(xs2), np.max(xs2), np.min(xs2) + ((np.max(xs2) - np.min(xs2))/2)
        print "ys2 dims", np.min(ys2), np.max(ys2), np.min(ys2) + ((np.max(ys2) - np.min(ys2))/2)
        rLimit = 45.5
        xOffset = np.min(xs1) + ((np.max(xs1) - np.min(xs1))/2)
        yOffset = np.min(ys1) + ((np.max(ys1) - np.min(ys1))/2)
        print "Center (%f, %f), Removing points close to edge: radius=%f" % (xOffset, yOffset, rLimit)
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

    print "Mean of diffs: ", np.nanmean(diffData)
    print "Std of diffs: ", np.nanstd(diffData)

    # find the zernike
    if not fitZernikies:
        return (xs1, ys1, xs2, ys2), diffData

    print "Fitting difference to zernikies ..."

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
    print "fitlist: ", fitlist
    C1.listcoefficient()
    C1.zernikemap()

    print "Converting from Noll to Active Surface ANSI Zernikies ..."
    # and now convert this to active surface zernike convention
    # why does the fitlist start with a zero? for Z0??  Anyways, avoid it
    nollZs = fitlist[1:(numZsFit+1)]
    asAnsiZs = noll2asAnsi(nollZs)
    print "nolZs"
    printZs(nollZs)
    print "active surface Zs"
    printZs(asAnsiZs)

    return (xs1, ys1, xs2, ys2), diffData

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
    GPU_PATH = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing"
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
        print "sigType not recognized: ", sigType
        
    # regrid to get into evenly spaced x y
    regridFn = "%s.%s.regridded" % (sigFn, sigType)
    xSigr, ySigr, diffsSigS = smoothXYZGpu(xSig,
                                           ySig,
                                           diffSigS,
                                           N,
                                           filename=regridFn)

    diffsSigS.shape = (N, N)
    imagePlot(diffsSigS, 'sig + sim regridded')

    
    # now load the reference scan
    from lassiAnalysis import loadProcessedData
    refFn2 = "%s.ptx.processed.npz" % refFn
    xsRef, ysRef, diffsRef = loadProcessedData(refFn2)
    imagePlot(np.log(np.abs(diffsRef)), "%s: ref scan" % refFn)
    
    diffsRef.shape = (N, N)
    diffData = diffsSigS - diffsRef
    diffDataLog = np.log(np.abs(diffData))
    imagePlot(diffData, "Surface Deformations")    
    imagePlot(diffDataLog, "Surface Deformations Log")

    return diffData

def main():
    fpath = "data/Baseline_STA10_HIGH_METERS.ptx"
    processLeicaScan(fpath)

if __name__ == "__main__":
    main()

