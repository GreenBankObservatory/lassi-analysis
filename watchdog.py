"""
simple module for making sure we are still writing to files
in given location
"""

import time
import os
from datetime import datetime

def getLatestFile(path):

    fs = os.listdir(path)

    # only want ptx files
    fs = [f for f in fs if f[-3:] == 'ptx']

    # rejoin path
    fs = [os.path.join(path, f) for f in fs]

    fs.sort(key=lambda x: os.path.getmtime(x))

    if len(fs) > 0:
        ts = os.path.getmtime(fs[-1])
        dt = datetime.fromtimestamp(ts)
        return fs[-1], dt
    else:
        return None

def main():
    sleepSecs = 60
    tolMins = 30.
    path = "/home/scratch/pmargani/LASSI/scannerData"
    while True:
        f = getLatestFile(path)
        if f is not None:
            fn, dt = f
            print fn, dt
        else:
            print "Could not find any files!", path
            continue
        now = datetime.now()
        print "compare to: ", now
        td = now - dt
        ageMins = td.seconds / 60.        
        print "No new file for %f minutes" % ageMins
        if ageMins > tolMins:
            msg = """
            WARNING:
            Scanner may be hung up - no new files in path %s for
            the last %f minutes.  Last file to be written: %s
            """ % (path, ageMins, fn)
            print msg
            with open("errorMsg.out", 'w') as f:
                f.write(msg)
            os.system('sendWatchdogMsg')    
        time.sleep(sleepSecs)    

if __name__ == '__main__':
    main()
