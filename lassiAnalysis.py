"""
This is a very high level module for integrating other modules and 
analyzing a single scan of Leica data once it's available.
"""
import os
import time

# import matplotlib
# matplotlib.use('agg')

from processPTX import processPTX
from main import smoothGPUs, smoothXYZGpu
from parabolas import fitLeicaScan
 
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

def processLeicaScan(fpath, N=512):
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

    # we'll provide primitive timing reports
    s = time.time()

    # removes headers, does basic rotations, etc.
    print "Processing PTX file ..."
    processPTX(fpath)
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
    diff, x, y = fitLeicaScan(fn, numpy=False, N=N)

    e = time.time()
    print "Elapsed minutes: %5.2f" % ((e - s) / 60.)

    s = time.time()
    print "Regriding data ..."
    xs, ys, diffs = smoothXYZGpu(x, y, diff, N)

    e = time.time()
    print "Elapsed minutes: %5.2f" % ((e - s) / 60.)

    return xs, ys, diffs

def main():
    fpath = "data/Baseline_STA10_HIGH_METERS.ptx"
    processLeicaScan(fpath)

if __name__ == "__main__":
    main()

