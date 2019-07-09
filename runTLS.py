import time
import os
import sys
import logging
import logging.config
from datetime import datetime

import runTLSLogging

from pyTLS import TLSaccess

# Initialize the logging configuration
logging.config.dictConfig(runTLSLogging.config)
logger = logging.getLogger(__name__)


def runScans(a):
    inScan = False
    prevTimes = None
    #for i in range(2):
    while True:
        time.sleep(5)
        status = a.get_status()
        logger.debug("State: %s, Scan: %d" % (status.state, status.scan_number))
        if status.state == 'Ready':
            # do we need to save off the previous scan?
            if inScan:
                scanNum = status.scan_number
                if scanNum > currScanNum:
                    logger.debug("Ready to save off new scan")
                    time.sleep(5)
                    logger.debug("Exporting Data")
                    a.export_data()
                    
                    # wait for shit to show up
                    keys = a.get_results().keys()
                    #while len(keys) < 6:
                    while len(keys) < 5:
                        logger.debug("not enough keys: %s" % keys)
                        time.sleep(1)
                        keys = a.get_results().keys()
                    logger.debug("Got all our keys: %s" % keys)


                    # make sure we aren't getting stale data
                    results = a.get_results()
                    times = results['TIME_ARRAY']
                    if prevTimes is not None:
                        if times[0] == prevTimes[0]:
                            logger.error('ERROR: we got stale data!')
                            return
                    prevTimes = times

                    time.sleep(3)
                    now = datetime.now()
                    ts = now.strftime("%Y-%m-%d_%H:%M:%S")
                    # TBF: why are we getting scan nums incremented by 2?
                    # fn = "%s_%s.ptx" % (scanNum, ts)
                    fn = "%s_%s.ptx" % (scanNum, ts)
                    path = "/home/scratch/pmargani/LASSI/scannerData"
                    fpath = os.path.join(path, fn)
                    logger.debug("Saving to file: %s" % fpath)
                    
                    logger.debug("Exporting to PTX")
                    a.export_ptx_results(fpath)
                    time.sleep(3)
                else:
                    logger.debug("scan number did not advance: %d vs %d" % (scanNum, currScanNumber))
            currScanNum = status.scan_number
            a.start_scan()
            logger.debug("running scan, waiting to get out of Ready")
            #time.sleep(3)
            inScan = True
        
def runOneScan(a):


    a.start_scan()
    time.sleep(3)
    state = "unknown"
    while state != "Ready":
        status = a.get_status()
        state = status.state
        time.sleep(1)
    # print "sleeping for 60 secs "
    # time.sleep(60)
    self.logger("exporting data after a quick sleep")
    time.sleep(5)
    a.export_data()
    print("Result keys: ", a.get_results().keys())
    keys = a.get_results().keys()
    while len(keys) < 5:
        print("not enough keys: ", keys)
        time.sleep(1)
        keys = a.get_results().keys()
    print("We have all our keys now: ", keys)
    print("exporting to file")
    a.export_ptx_results("/tmp/delme2.ptx")    


def thisCB(key, data):
    logger.debug("thisCB: %s" % key)

def main():

    a=TLSaccess("lassi.ad.nrao.edu")

    status = a.get_status()
    if status.state != 'Ready':
        logger.debug("Is not ready, not running scan")
        a.cntrl_exit()
        sys.exit(1)

    a.subscribe(frame_type="X_ARRAY", cb_fun=thisCB)
    a.subscribe(frame_type="Y_ARRAY", cb_fun=thisCB)
    a.subscribe(frame_type="Z_ARRAY", cb_fun=thisCB)
    a.subscribe(frame_type="I_ARRAY", cb_fun=thisCB)
    a.subscribe(frame_type="TIME_ARRAY", cb_fun=thisCB)

    # reconfigure? why not
    proj = "11jun2019_24hrTests"
    res = "63mm@100m"
    #res = "31mm@100m"
    sensitivity = "Normal"
    scan_mode = "Speed"
    cntr_az = 352
    cntr_el = 45
    az_fov = 180
    el_fov = 90
    msg = "Proj: %s, Resolution: %s, Sensitivity: %s, Scan Mode: %s, cntr_az: %f, cntr_el: %f, az_fov: %f, el_fov: %f" % (proj, res, sensitivity, scan_mode, cntr_az, cntr_el, az_fov, el_fov)
    logger.debug(msg)
    #logger.debug("Configuring to: delme, 31mm@100m, Normal, Speed, 352, 0, 180, 180")
    #a.configure_scanner("delme", "31mm@100m", "Normal", "Speed", 352, 45, 180, 90)
    #a.configure_scanner("delme", "63mm@100m", "Normal", "Speed", 352, 45, 180, 90)
    a.configure_scanner(proj, res, sensitivity, scan_mode, cntr_az, cntr_el, az_fov, el_fov)

    #logger.debug("Configuring Scanner ...")
    #a.configure_scanner("delme", "31mm@100m", "Normal", "Speed", 352, 0, 5, 5)
    try:
        logger.debug("Running Scans!")
        runScans(a)
    except KeyboardInterrupt:
        a.cntrl_exit()
    finally:
        a.cntrl_exit()

    a.cntrl_exit()
    sys.exit(1)

if __name__=='__main__':
    main()
