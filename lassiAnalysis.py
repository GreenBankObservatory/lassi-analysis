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

from processPTX import processPTX, processNewPTX
from main import smoothGPUs, smoothXYZGpu
from parabolas import fitLeicaScan, imagePlot
from zernikeIndexing import noll2asAnsi, printZs
import opticspy

# where is the code we'll be running?
GPU_PATH = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing"

def smooth(fpath, N=512):

    assert os.path.isfile(fpath)

    # we need to specify the aboslute path of our inputs
    abspath = os.path.abspath(fpath)

    # our output will be same file name, but with appendages
    outfile = os.path.basename(fpath)

    smoothGPUs(GPU_PATH, abspath, outfile, N)

    # make sure the output is where it should be
    outfiles = []
    for dim in ['x', 'y', 'z']:
        dimFile = "%s.%s.csv" % (outfile, dim)
        dimPath = os.path.join(GPU_PATH, dimFile)
        outfiles.append(dimPath)
        assert os.path.isfile(dimPath)

    return outfiles

def processLeicaScan(fpath, N=512, rot=None, sampleSize=None):
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
    xOffset = -8.
    yOffset = 50.0
    if rot is None:
        rot = 0.
    processNewPTX(fpath,
                  rot=rot,
                  rFilter=True,
                  iFilter=False,
                  sampleSize=sampleSize) #xOffset=xOffset, yOffset=yOffset)
    processedPath = fpath + ".csv"

    e = time.time()
    print "Elapsed minutes: %5.2f" % ((e - s) / 60.)

    # reduces our data via GPUs
    s = time.time()
    print "Smoothing data ..."
    smoothedFiles = smooth(processedPath, N=N)

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
                              rFilter=False)

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

def processLeicaScanPair(filename1, filename2, processed=False, fitZernikies=True):
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
    else:
        # we need to process each scan
        # WARNING: each scan takes about 10 minutes    
        xs1, ys1, diff1 = processLeicaScan(filename1)
        xs2, ys2, diff2 = processLeicaScan(filename2)

    # find the difference of the difference!
    print "Finding difference between scans ..."
    N = 512
    diffData = diff1 - diff2
    diffData.shape = (N, N)
    diffDataLog = np.log(np.abs(diffData))
    imagePlot(diffDataLog, "Surface Deformations")

    # find the zernike
    if not fitZernikies:
        return 

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

def main():
    fpath = "data/Baseline_STA10_HIGH_METERS.ptx"
    processLeicaScan(fpath)

if __name__ == "__main__":
    main()

