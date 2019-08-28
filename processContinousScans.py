"""
Process continous scans in the style done on June 11
for the 24 hour test scans.
"""
import os
from datetime import datetime

def getScans(path):
    "Return sorted list of PTX files in path"

    # but how to sort?
    # scans have filename right now of <scan>_<datetime>.ptx
    # so if we trust our scan numbering, just use that
    fs = sorted(os.listdir(path))
    fs = [f for f in fs if f[-3:] == 'ptx']

    scans = []
    for f in fs:
        scanNum, dt = parseScanFilename(f)
        if scanNum is None or dt is None:
            continue
        scans.append((scanNum, dt, f))

    return sorted(scans)
        
def getScanPairs(fs):
    "[1,2,3] -> [(1,2), (2,3)]"

    pairs = []    
    # for i, f in enumerate(fs[:-1]):
    for i in range(len(fs[:-1])):
        thisScan = fs[i]
        nextScan = fs[i+1]
        pairs.append((thisScan, nextScan))
    return pairs

def parseScanFilename(fn):

    # parts = fn.split('_')
    # scanNum = int(parts[0])
    idx = fn.find("_")
    try:
        scanNum = int(fn[:idx])
        dt = datetime.strptime(fn[idx+1:-4], "%Y-%m-%d_%H:%M:%S")
    except:
        print("ERROR parsing scan", fn)
        scanNum = dt = None    
    return scanNum, dt

def processScanPair(scan1, scan2, path):

    
    print("Processing: ")
    print(scan1)
    print(scan2)

    scanNum1, dt1, fn1 = scan1 #parseScanFilename(fn1)
    scanNum2, dt2, fn2 = scan2 #parseScanFilename(fn2)

    print("Scans: %d vs %d" % (scanNum1, scanNum2))
    print("Times: %s vs %s" % (dt1, dt2))

    if scanNum1 >= scanNum2:
        print("WARNING: Scans are not incrementing:", scanNum1, scanNum2)
    if dt2 > dt1:
        print("WARNING: Timestamps are reversed!!!")

    print("")

def processScan(path, filename, scanNum, dt):

    print("Processing: ", scanNum, dt, filename)
    fpath = os.path.join(path, filename)

    # processLeicaScan(fpath, rot=-10.)

def processContinousScans(path):
    "For given path, process each neighboring scan as a pair"

    fs = getScans(path)

    # iterate through scan pairs
    #pairs = getScanPairs(fs)

    #for scan1, scan2 in pairs:
    #    processScanPair(scan1, scan2, path)
    for scanNum, dt, fn in fs:
        processScan(path, fn, scanNum, dt)

def main():
    path = "/home/scratch/pmargani/LASSI/scannerData"
    processContinousScans(path)

if __name__ == '__main__':
    main()

    # started + 15:44:12
    # ended = 15:45:40
