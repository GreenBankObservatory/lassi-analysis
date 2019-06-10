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
        print "State:", status.state
        if status.state == 'Ready':
            # do we need to save off the previous scan?
            if inScan:
                scanNum = status.scan_number
                if scanNum > currScanNum:
                    print "Ready to save off new scan"
                    time.sleep(5)
                    print "Exporting Data"
                    a.export_data()
                    
                    # wait for shit to show up
                    keys = a.get_results().keys()
                    while len(keys) < 6:
                        print "not enough keys: ", keys
                        time.sleep(1)
                        keys = a.get_results().keys()
                    print "Got all our keys: ", keys


                    # make sure we aren't getting stale data
                    results = a.get_results()
                    times = results['TIME_ARRAY']
                    if prevTimes is not None:
                        if times[0] == prevTimes[0]:
                            print 'ERROR: we got stale data!'
                            return
                    prevTimes = times

                    time.sleep(3)
                    now = datetime.now()
                    ts = now.strftime("%Y-%m-%d_%H:%M:%S")
                    fn = "%s_%s.ptx" % (scanNum, ts)
                    path = "/home/scratch/pmargani/LASSI/scannerData"
                    fpath = os.path.join(path, fn)
                    print "Saving to file", fpath
                    
                    print "Exporting to PTX"
                    a.export_ptx_results(fpath)
                    time.sleep(3)
                else:
                    print "scan number did not advance: ", scanNum, currScanNumber
            currScanNum = status.scan_number
            a.start_scan()
            print "running scan, waiting to get out of Ready"
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
    print "exporting data after a quick sleep"
    time.sleep(5)
    a.export_data()
    print "Result keys: ", a.get_results().keys()
    keys = a.get_results().keys()
    while len(keys) < 5:
        print "not enough keys: ", keys
        time.sleep(1)
        keys = a.get_results().keys()
    print "We have all our keys now: ", keys
    print "exporting to file"
    a.export_ptx_results("/tmp/delme2.ptx")    


def thisCB(key, data):
    print "thisCB:", key

def main():

    a=TLSaccess("lassi.ad.nrao.edu")

    status = a.get_status()
    if status.state != 'Ready':
        print "Is not ready, not running scan"
        a.cntrl_exit()
        sys.exit(1)

    a.subscribe(frame_type="X_ARRAY", cb_fun=thisCB)
    a.subscribe(frame_type="Y_ARRAY", cb_fun=thisCB)
    a.subscribe(frame_type="Z_ARRAY", cb_fun=thisCB)
    a.subscribe(frame_type="I_ARRAY", cb_fun=thisCB)
    a.subscribe(frame_type="TIME_ARRAY", cb_fun=thisCB)

    try:
        print "Running Scans!"
        runScans(a)
    except KeyboardInterrupt:
        a.cntrl_exit()

    sys.exit(1)

if __name__=='__main__':
    main()
