import os
import pytz
from datetime import datetime

from astropy.io import fits

def dt2utc(dt):
    "converts naive dt assumed in EST to UTC"
    assert dt.tzinfo is None
    utc=pytz.utc
    eastern=pytz.timezone('US/Eastern')
    estDt = eastern.localize(dt)
    return estDt.astimezone(utc)

def getLassiScanTimes(path):

    fs = getFiles(path, "ptx")

    scans = {}
    for f in fs:
        idx = f.find('.')
        scanNum = int(f[5:idx])
        fpath = os.path.join(path, f)
        timeEst = datetime.fromtimestamp(os.path.getmtime(fpath))
        print f, timeEst
        scans[scanNum] = dt2utc(timeEst)
    return scans
                       
def getActiveSurfaceTimes(path):

    # path = "/home/gbtdata/TINT/ActiveSurfaceMgr/"
    fs = getFiles(path, "fits")

    scans = {}
    for f in fs:
        fpath = os.path.join(path, f)
        scanStartUtc = datetime.strptime(f[:-5], "%Y_%m_%d_%H:%M:%S")
        # scanStartUtc.tzinfo = pytz.utc
        # scanStartUtc.astimezone(pytz.utc)
        scanStartUtc = pytz.utc.localize(scanStartUtc)
        hdus = fits.open(fpath)
        scanNum = int(hdus[0].header['SCAN'])
        scans[scanNum] = scanStartUtc

    return scans
                         
def getFiles(path, ext):
    files = os.listdir(path)
    return sorted([f for f in files if f[-len(ext):] == ext])

def getActSrfScanNum(lassiScanNum):
    # if lassiScanNum == 16:
        # return None
    if lassiScanNum >= 8 and lassiScanNum < 16:
        return lassiScanNum - 4
    if lassiScanNum >= 17 and lassiScanNum < 23:
        return lassiScanNum - 5
    return None    
    

def getSep16TestScanTimes():
    scanPath = "/home/sandboxes/pmargani/LASSI/data/17sep2019/"
    fitsPath = "/home/gbtdata/TINT/ActiveSurfaceMgr/"

    print "LASSI scans map to Act. Surface scans:"
    for i in range(25):
        print "scans: ", i, getActSrfScanNum(i)

    asScans = getActiveSurfaceTimes(fitsPath)
    lassiScans = getLassiScanTimes(scanPath)

    scans = {}
    for scanNum, scanEndTime in lassiScans.items():
        asScanNum = getActSrfScanNum(scanNum)
        asScanTime = asScans[asScanNum] if asScanNum is not None else None
        scans[scanNum] = (asScanTime, scanEndTime)
    
    return scans

def main():
    scans = getSep16TestScanTimes()
    # print scans
    for scanNum in sorted(scans.keys()):
        start, end = scans[scanNum]
        durMins = None
        if start is not None:
            durMins = (end - start).seconds / 60.
        print scanNum, scans[scanNum], durMins

if __name__ == '__main__':
    main()
