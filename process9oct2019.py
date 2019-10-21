import os
from datetime import datetime
from glob import glob
import pickle
from pprint import pprint

import numpy as np

from lassiAnalysis import processLeicaScan, loadProcessedData, processLeicaScanPair
from plotting import imagePlot

def showProcessedScans():

    fs = getProcessedFiles()

    for dt, fn in fs:
        print dt, fn
        xs, ys, zs = loadProcessedData(fn)
        # imagePlot(zs, fn)
        imagePlot(np.log(np.abs(zs)), fn)

def getFileByScanNumber(scanNumber, processed=False):

    ext = "ptx"
    if processed:
        ext = "processed.npz"

    path = "/home/sandboxes/pmargani/LASSI/data/9oct2019"
    if processed:
        path = "."

    fs = glob('%s/%d_*.%s' % (path,scanNumber,ext))

    # fs = [f for f in fs if f[-3:] == "ptx" and "_times"]
    if len(fs) == 0:
        print "No files found for", scanNumber
        f = None
    elif len(fs) > 1:
        print "Many files found for ", scanNumber, fs
        return fs[0]
    elif len(fs) == 1:
        return fs[0]
    else:
        print "fuck!"
        return None

def processScanPair(scanNum1, scanNum2):

        f1 = getFileByScanNumber(scanNum1, processed=True)
        f2 = getFileByScanNumber(scanNum2, processed=True)

        return processLeicaScanPair(f1[:-14], f2[:-14], processed=True)

def processScans(scanNums):

    for scanNum in scanNums:
        f = getFileByScanNumber(scanNum)
        print "file: ", f
        if f is not None:
            #d = processLeicaScan(f, xOffset=-50, yOffset=-7.5, rot=80.)
            # d = processLeicaScan(f, xOffset=-50, yOffset=-7.5, rot=80.)
            d = processLeicaScan(f, xOffset = -45, yOffset=-5., rot=80.)

def getProcessedFiles():
    "Find the processed files from Oct 9, 2019"

    fs = os.listdir('.')
    fs = sorted([f for f in fs if f[-13:] == "processed.npz"])

    # now, only look at those that are from our dates, so
    # they should look like <scan Number>_2019_10_09_*
    # fss = []
    # for f in fs:
    #     ind = f.find("2019-10")
    #     if ind == -1:
    #         continue

        # ps = f[ind:].split('_')
        # year = ps[0]
        # assert year = "2019"
        # month = ps[1]
        # if month != "10":
        #     continue
        # day =     
    fs = [f for f in fs if f.find("2019-10") != -1]
    
    return sorted([(filename2dt(f), f) for f in fs])

def filename2dt(f):
    "100_2019-10-09_21:29:27.ptx.processed.npz -> datetime"
    s = f.find("2019")
    e = f.find(".ptx")
    dtStr = f[s:e]
    return datetime.strptime(dtStr, "%Y-%m-%d_%H:%M:%S")

def notProcessed(path, f):
    "If the file has an associated .csv, it's been processed"
    return not os.path.isfile(os.path.join(path, f + ".csv"))

def process():
    "Find the files, and process them"

    # first, find the files that haven't been processed yet
    path = "/home/sandboxes/pmargani/LASSI/data/9oct2019"
    fs = os.listdir(path)
    print len(fs)
    fs = sorted([f for f in fs if f[-4:] == ".ptx"])
    print len(fs)

    fs = [f for f in fs if notProcessed(path, f)]
    print len(fs)

    # and process them
    for i, f in enumerate(fs):
        fpath = os.path.join(path, f)
        print "processing: ", i, fpath
        d = processLeicaScan(fpath,xOffset = -50, yOffset=-7.5, rot=80.) 

def showAllScans():
    """
    {2: {'AS FITS': {'DATE-OBS': '2019-10-10T16:03:28',
                 'SCAN': 49,
                 'ZERNIKES': [],
                 'filename': '2019_10_10_16:03:28.fits',
                 'proj': 'JUNK'},
     'AS Scan #': 49,
     'AS in SC?': 0,
     'configuration': {'az_fov': 180.0,
                       'cntr_az': 279.0,
                       'cntr_el': 45.0,
                       'el_fov': 90.0,
                       'project': ' 9oct2019',
                       'resolution': ' 63mm@100m',
                       'scan_mode': ' Speed',
                       'sensitivity': ' Normal'},
     'dt': datetime.datetime(2019, 10, 10, 12, 3, 36),
     'elevation': 50.0,
     'ptx': '2_2019-10-10_12:06:30.ptx',
     'scan #': 2,
     'type': 'REF',
     'wind(m/s)': 0.2,
     'zernike index': None,
     'zernike value': None},
    """

    scans = getScanMap()

    scanNums = sorted(scans.keys())

    path = "/home/sandboxes/pmargani/LASSI/data/9oct2019"

    for scanNum in scanNums:
        scanInfo = scans[scanNum]
        if "ptx" not in scanInfo:
            print "Missing PTX file!", scanNum
            continue
        ptx = scanInfo["ptx"]
        # if notProcessed('.', ptx):
        # fn = os.path.join(path, ptx + ".processed.npz")
        fn = ptx + ".processed.npz"
        if not os.path.isfile(fn):
            continue    
        print scanNum, ptx
        pprint(scanInfo)
        xs, ys, zs = loadProcessedData(fn)
        # imagePlot(zs, fn)
        imagePlot(np.log(np.abs(zs)), fn)

def processConfigurationTests():

    def get(d, k):
        x = d.get(k, None)

        if x is not None and type(x) == type("string"):
            x = x.strip()
        return x

    scans = range(24, 38)

    with open("lassiScans9oct2019.pickle", 'r') as f:
        d = pickle.load(f)

    print ""
    print "%-5s %-27s %-12s %-8s %-6s %-12s %-6s %-6s %-5s" % ('scan', 'filename', 'res', 'sen', 'mode', 'bytes', 'scan', 'export', 'type')
    for scan in scans:
        s = d[scan]
        config = s['configuration']
        res = get(config, 'resolution')
        sen = get(config, 'sensitivity')
        mode = get(config, 'scan_mode')
        fn = get(s, 'ptx')
        ptxBytes = get(s, 'ptx (bytes)')
        scanMins = get(s, 'scan (mins)')
        exportMins = get(s, 'export (mins)')   
        stype = get(s, 'type')   
        if ptxBytes is None:
            continue
        print "%-5d %-27s %-12s %-8s %-6s %-12s %-6s %-6s %-5s" % \
            (scan, fn, res, sen, mode, ptxBytes, scanMins, exportMins, stype)

def main():
    # process()
    # print getProcessedFiles()
    # scans = [135, 136]
    # scans.extend(range(27,38))
    # scans = range(323, 329)
    #scans = [340, 341]
    #scans = [362, 363]
    # scans = range(440, 455)
    # print scans
    # processScans(scans)
    # showAllScans()
    processConfigurationTests()

if __name__ == '__main__':
    main()
