"Module for class ProjectScanLog"

import os
import logging
import pprint

from astropy.io import fits

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

        self.scans = {}

    def open(self):
        "Reads ScanLog.fits and creates simple map of scans and devices"

        scanLogPath = os.path.join(self.projPath, "ScanLog.fits")
        try:
            hdus = fits.open(scanLogPath)
        except IOError:
            logger.error("No ScanLog.fits at: %s" % scanLogPath)
            return

        scanInfo = hdus[1].data

        for _, scanNum, filepath in scanInfo:

            if scanNum not in self.scans:
                self.scans[scanNum] = {}

            # skip the 'SCAN STARTING' and 'SCAN FINISHED' rows
            if 'SCAN' not in filepath:
                # should be of the form /project/device/timestamp.fits
                # strip out the device name
                fileParts = filepath.split("/")
                device = fileParts[2]
                filename = fileParts[3]
                self.scans[scanNum][device] = filename

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
