"""
In this module we do some high level
analysis if the LASSI data taken March 27, 2019,
our first test run, with an emphasis on external
factors such as wind
"""

import os

import numpy as np
import matplotlib.pylab as plt
import pytz

from parabolas import imagePlot
from march27scans import march27scans, getStartEndTimes
from lassiAnalysis import loadProcessedData
from SamplerData import SamplerData

w3sd = SamplerData("Weather-Weather3-weather3", True)

def dt2tuple(dt):
    return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second

def dt2utc(dt):
    "converts naive dt assumed in EST to UTC"
    assert dt.tzinfo is None
    utc=pytz.utc
    eastern=pytz.timezone('US/Eastern')
    estDt = eastern.localize(dt)
    return estDt.astimezone(utc)

def plotWindSpeeds(sDt, eDt):

    start = dt2tuple(sDt)          
    end = dt2tuple(eDt)          
    dates = (start,end)
    cols = (0,1)
    exprs = ('X','Y1')
    
    dmjds, vels = w3sd.GetPlotData(dates,cols,exprs) 

    f = plt.figure()
    ax = f.gca()
    ax.plot(dmjds, vels)
    plt.xlabel('DMJD')
    plt.ylabel('m/s')
    plt.title('W3 Wind Speeds')

    
    # print analysis
    print "Wind Speeds (m/s)"
    print "UTC: %s - %s" % (sDt, eDt)
    print "Minutes: ", (sDt - eDt).seconds / 60.
    print "Min: %f, Max: %f" % (np.min(vels), np.max(vels))
    print "Mean: %f, Std: %f" % (np.mean(vels), np.std(vels))

def summarizeCleanedData():
    scanNums = sorted(march27scans.keys())    

    # plot the wind speeds for all scans together
    firstScan = scanNums[0]
    lastScan = scanNums[-1]
    s, e = getStartEndTimes(firstScan, lastScan)
    plotWindSpeeds(dt2utc(s), dt2utc(e))

    for scanNum in scanNums:
        fn = "Clean%d.ptx.processed.npz" % scanNum

        # nothing to do if we haven't proceessed it yet
        if not os.path.isfile(fn):
            continue

        print ""
        print "*"*80
        typ = march27scans[scanNum]['type']
        print "Scan: %d, %s" % (scanNum, typ)

        xs, ys, diffs = loadProcessedData(fn)
        imagePlot(np.log(np.abs(np.diff(diffs))),
                  "Scan %d Log (%s)" % (scanNum, typ))

        nextScanNum = scanNum + 1
        if nextScanNum in scanNums:
            sDt, eDt = getStartEndTimes(scanNum, nextScanNum)    
            print "Time (EST) between scan %d and %d:" % (scanNum, nextScanNum)
            print "%s - %s" % (sDt, eDt)
            plotWindSpeeds(dt2utc(sDt), dt2utc(eDt))

def main():
    summarizeCleanedData()

if __name__=='__main__':
    main()
