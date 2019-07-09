"Module for class ProjectScanLog"

import os
import logging
import pprint
from datetime import datetime, timedelta

from astropy.io import fits
from astropy.time import Time
from astropy import units
from astropy.coordinates import Angle

#logger = logging.getLogger(__name__)

class CheapLogging:
    def error(self, msg):
        print "ERROR", msg
    def debug(self, msg):
        print "DEBUG", msg

logger = CheapLogging()        
        
class ProjectScanLog:

    """
    This class reads the ScanLog.fits, and provides a convenient
    map of it's content so we can easily map scans to filenames.
    """

    def __init__(self, projPath):

        self.projPath = projPath
        self.project = os.path.basename(projPath)

        self.scans = {}
        self.scanTimes = {}

        self.projStartDts = None
        self.projEndDts = None

        self.projStartDt = None
        self.projEndDt = None

    def open(self):
        "Reads ScanLog.fits and creates simple map of scans and devices"

        scanLogPath = os.path.join(self.projPath, "ScanLog.fits")
        try:
            hdus = fits.open(scanLogPath)
        except IOError:
            logger.error("No ScanLog.fits at: %s" % scanLogPath)
            return

        scanInfo = hdus[1].data

        start = end = None

        for _, scanNum, filepath in scanInfo:

            if scanNum not in self.scans:
                self.scans[scanNum] = {}
                startDt = endDt = None
                self.scanTimes[scanNum] = {
                    'start': startDt,
                    'end': endDt,
                    'durSecs': None
                }    

            # skip the 'SCAN STARTING' and 'SCAN FINISHED' rows
            if 'SCAN' not in filepath:
                # should be of the form /project/device/timestamp.fits
                # strip out the device name
                fileParts = filepath.split("/")
                device = fileParts[2]
                filename = fileParts[3]
                self.scans[scanNum][device] = filename
            else:
                # convert these lines to start stop times:
                # SCAN STARTING AT 58559  2:43:28
                if 'STARTING' in filepath:
                    try:
                        startDt = self.stringToDt(filepath)
                    except:
                        startDt = None
                        print "error formating: ", filepath
                        continue    
                    self.scanTimes[scanNum]['start'] = startDt 
                if 'FINISHED' in filepath:
                    try:
                        endDt = self.stringToDt(filepath)
                    except:
                        endDt = None
                        print "error formating: ", filepath
                        continue    
                    self.scanTimes[scanNum]['end'] = endDt
                if startDt is not None and endDt is not None:
                    self.scanTimes[scanNum]['durSecs'] = (endDt - startDt).seconds

        # now try to figure out what the project date range is
        startDts = [ts['start'] for _, ts in self.scanTimes.items() if ts['start'] is not None]
        endDts = [ts['end'] for _, ts in self.scanTimes.items() if ts['end'] is not None]
        self.projStartDts = startDts
        self.projEndDts = endDts
        if len(startDts) > 0:
            self.projStartDt = min(startDts)
        if len(endDts) > 0:    
            self.projEndDt = max(endDts)

    def stringToDt(self, string):
        "'SCAN STARTING AT 58559  2:43:28' -> datetime object"

        # print "string", string
        assert 'STARTING' in string or 'FINISHED' in string

        mjdTimeStr = ' '.join(string.split(' ')[3:])
        mjd = int(mjdTimeStr.split(' ')[0])
        timeStr = mjdTimeStr.split(' ')[-1]

        # mjd to datetime
        date = self.mjd2utc(mjd)

        fmt = "%Y-%m-%d %H:%M:%S"
        dtfmt = "%Y-%m-%d"
        dtStr = date.strftime(dtfmt) + " " + timeStr
        # print dtStr
        return datetime.strptime(dtStr, fmt)

    def mjd2utc(self, mjd):
        "Converts MJD values to UTC datetime objects"
        t = Time(mjd, format='mjd')
        t.format = 'datetime'
        return t.value

    def getScanFilename(self, scanNum):
        "Returns the shared filename (timestamp) for all FITS files"

        # TBF: what to do about Receiver FITS names?

        # make sure there is really only one shared filename
        fns = list(set([fn for device, fn in self.scans[scanNum].items()]))
        if len(fns) == 1:
            return fns[0]
        elif not fns:
            logger.error("No files produced for scan: %s" % scanNum)
            return None
        else:
            # more then one file name!
            logger.error("More then one filename for this scan: %d" % scanNum)
            logger.error(fns)
            return None

    def getDeviceFilename(self, device, scanNum, logError=True):
        "General access method for filenames"
        if scanNum not in self.scans:
            logger.error("Scan missing from project: %d" % scanNum)
            return None
        if device not in self.scans[scanNum]:
            # raise an error in the logs?
            msg = "Device %s missing from scan: %d" % (device, scanNum)
            if logError:
                logger.error(msg)
            else:
                logger.debug(msg)
            return None
        return self.scans[scanNum][device]

    def getAntennaFilename(self, scanNum, logError=True):
        "shortcut for retrieving FITS file name for this device and scan"
        return self.getDeviceFilename('Antenna', scanNum, logError=logError)

    def getGOFilename(self, scanNum):
        "shortcut for retrieving FITS file name for this device and scan"
        return self.getDeviceFilename('GO', scanNum)


if __name__ == '__main__':
    path = "/home/gbtdata/AGBT18A_478_32"
    p = ProjectScanLog(path)
    p.open()
    pprint.pprint(p.scans)
