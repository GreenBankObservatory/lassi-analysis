import time
import os
import sys
import logging
import logging.config
import numpy as np
from datetime import datetime

import runTLSLogging

from pyTLS import TLSaccess

# Initialize the logging configuration
logging.config.dictConfig(runTLSLogging.config)
logger = logging.getLogger(__name__)


def runScans(a, path=None):

    if path is None:
        path = "/home/scratch/pmargani/LASSI/scannerData"

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


                    logger.debug(a.get_results()['HEADER'])

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

                    # save off the time values
                    # path = "/home/scratch/pmargani/LASSI/scannerData"
                    fn = "%s_%s_times.csv" % (scanNum, ts)
                    fpath = os.path.join(path, fn)
                    np.savetxt(fpath, times, delimiter=",")

                    # TBF: why are we getting scan nums incremented by 2?
                    # fn = "%s_%s.ptx" % (scanNum, ts)
                    fn = "%s_%s.ptx" % (scanNum, ts)
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
        
def runOneScan(a, path=None):


    s = time.time()

    a.start_scan()
    status = a.get_status()
    scanNum = status.scan_number
    logger.debug( "Running scan with scan number: i%d" % scanNum)

    time.sleep(3)
    state = "unknown"
    while state != "Ready":
        status = a.get_status()
        state = status.state
        print "State: ", state
        print datetime.now()
        time.sleep(1)

    print "Scan Done"
    scanSecs = time.time() - s

    # print "sleeping for 60 secs "
    # time.sleep(60)

    s = time.time()

    logger.debug("exporting data after a quick sleep")
    time.sleep(5)
    a.export_data()

    exportCallSecs = time.time() - s
    s = time.time()

    print("Result keys: ", a.get_results().keys())
    keys = a.get_results().keys()
    while len(keys) < 5:
        print("not enough keys: ", keys)
        time.sleep(1)
        keys = a.get_results().keys()
    print "We have all our keys now: ", keys

    logger.debug(a.get_results()['HEADER'])

    gotKeysSecs = time.time() - s
    s = time.time()

    #filename = "/tmp/test-%d.ptx" % scanNum
    if path is None:
        path = "/home/sandboxes/pmargani/LASSI/data"

    results = a.get_results()
    times = results['TIME_ARRAY']

    # time.sleep(3)
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d_%H:%M:%S")

    fn = "%s_%s_times.csv" % (scanNum, ts)
    fpath = os.path.join(path, fn)
    logger.debug( "saving times to file %s" % fpath)
    np.savetxt(fpath, times, delimiter=",")

    ptxfn = "%s_%s.ptx" % (scanNum, ts)

    # ptxfn = "test-%d.ptx" % scanNum
    filenamePtx = os.path.join(path, ptxfn)    
    # filename = "/home/sandboxes/pmargani/LASSI/data/17sep2019/test-%d.ptx" % scanNum
    logger.debug( "exporting to file %s" % filenamePtx)
    a.export_ptx_results(filenamePtx)    

    exportPtxSecs = time.time() - s
    s = time.time()

    #csvfn = "test-%d.csv" % scanNum
    csvfn = "%s_%s.csv" % (scanNum, ts)
    filenameCsv = os.path.join(path, csvfn)    
    logger.debug( "exporting to file %s" % filenameCsv)
    a.export_csv_results(filenameCsv)    
    
    exportCsvSecs = time.time() - s

    totalExportSecs = exportCallSecs + gotKeysSecs + exportPtxSecs

    # report on Scan dimensions:
    logger.debug( "PTX Bytes: %d" % os.path.getsize(filenamePtx))
    logger.debug( "CSV Bytes: %d" % os.path.getsize(filenameCsv))

    logger.debug( "Scan Mins: %5.2f" %  (scanSecs / 60.))
    logger.debug( "Total Export Mins: %5.2f" % (totalExportSecs / 60.))

    return filenamePtx

def testScanRange(a, proj, cntr_az, cntr_el, az_fov, el_fov, path=None):

    res = "63mm@100m"
    sensitivity = "Normal"
    scan_mode = "Speed"

    logger.debug( "Scanning cntr (az, el): %5.2f, %5.2f" % (cntr_az, cntr_el))
    logger.debug( "Scanning fov (az, el): %5.2f %5.2f" % ( az_fov, el_fov) )
    
    a.configure_scanner(proj, res, sensitivity, scan_mode, cntr_az, cntr_el, az_fov, el_fov)

    fn = runOneScan(a)

    # report scan dimensions
    logger.debug( "Scanned square degrees: %5.2f" % ( az_fov * el_fov))

    return fn
    
def thisCB(key, data):
    logger.debug("thisCB: %s" % key)

def configureScanner(a, proj, res, sensitivity, scan_mode, cntr_az, cntr_el, az_fov, el_fov):
    "Wraps configure_scanner so that parameters are logged"

    # validate some inputs
    resValues = ["500mm@100m", 
                 "250mm@100m",
                 "125mm@100m", 
                 "63mm@100m",
                 "31mm@100m",
                 "16mm@100m",
                 "8mm@100m"]
    assert res in resValues             

    sensValues = ['Normal', 'High']
    assert sensitivity in sensValues

    modeValues = ["Speed", "Range", "Medium Range", "Long Range"]
    assert scan_mode in modeValues

    logger.debug("Configuring Scanner:")
    logger.debug("Project: %s", proj)
    logger.debug("Resolution: %s", res)
    logger.debug("Sensitivity: %s", sensitivity)
    logger.debug("Scan Mode: %s", scan_mode)

    logger.debug("cntr_az: %f", cntr_az)    
    logger.debug("cntr_el: %f", cntr_el)    
    logger.debug("az_fov: %f", az_fov)    
    logger.debug("el_fov: %f", el_fov)    

    a.configure_scanner(proj, res, sensitivity, scan_mode, cntr_az, cntr_el, az_fov, el_fov)

def runThisScan():


    a=TLSaccess("lassi.ad.nrao.edu")

    status = a.get_status()
    if status.state != 'Ready':
        logger.debug("Is not ready, not running scan")
        a.cntrl_exit()
        sys.exit(1)

    # for latest version, this is no longer necessary
    #a.subscribe(frame_type="X_ARRAY", cb_fun=thisCB)
    #a.subscribe(frame_type="Y_ARRAY", cb_fun=thisCB)
    #a.subscribe(frame_type="Z_ARRAY", cb_fun=thisCB)
    #a.subscribe(frame_type="I_ARRAY", cb_fun=thisCB)
    #a.subscribe(frame_type="TIME_ARRAY", cb_fun=thisCB)

    # reconfigure? why not
    #proj = "11jun2019_24hrTests"
    #proj = "14aug2019_test"
    # proj = "17sep2019_test"
    proj = "test"
    res = "63mm@100m"
    #res = "31mm@100m"
    sensitivity = "Normal"
    scan_mode = "Speed"
    # scnasd 4 - 15
    #cntr_az = 270
    #cntr_el = 45
    #az_fov = 180
    #el_fov = 90
    # scans > 15
    cntr_az = 270
    cntr_el = 45
    az_fov = 45
    el_fov = 45

    # make this a short scan
    #az_fov = 30
    #el_fov = 30
    msg = "Proj: %s, Resolution: %s, Sensitivity: %s, Scan Mode: %s, cntr_az: %f, cntr_el: %f, az_fov: %f, el_fov: %f" % (proj, res, sensitivity, scan_mode, cntr_az, cntr_el, az_fov, el_fov)
    logger.debug(msg)
    #logger.debug("Configuring to: delme, 31mm@100m, Normal, Speed, 352, 0, 180, 180")
    #a.configure_scanner("delme", "31mm@100m", "Normal", "Speed", 352, 45, 180, 90)
    #a.configure_scanner("delme", "63mm@100m", "Normal", "Speed", 352, 45, 180, 90)
    # a.configure_scanner(proj, res, sensitivity, scan_mode, cntr_az, cntr_el, az_fov, el_fov)
    configureScanner(a, proj, res, sensitivity, scan_mode, cntr_az, cntr_el, az_fov, el_fov)

    #logger.debug("Configuring Scanner ...")
    #a.configure_scanner("delme", "31mm@100m", "Normal", "Speed", 352, 0, 5, 5)
    try:
        logger.debug("Running Scans!")
        runOneScan(a)
    except KeyboardInterrupt:
        a.cntrl_exit()
    finally:
        a.cntrl_exit()

    a.cntrl_exit()
    sys.exit(1)

def main():
    runThisScan()
    
if __name__=='__main__':
    main()
