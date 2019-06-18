import os
from datetime import datetime

import numpy as np

def parseLog(path, f):
    "Parse given logfile, and return elapsed time (seconds)"
    
    fpath = os.path.join(path, f)
     
    with open(fpath, 'r') as f:
        ls = f.readlines()

    if len(ls) < 2:
        print "Not enough lines in", fpath
        return None

    # first line should look like:
    # Starting oofshell Fri Sep 28 10:42:25 EDT 2018
    l1 = ls[0]
    if ' '.join(l1.split(' ')[:2]) != 'Starting oofshell':
        print "Unexpected first liine: ", l1
        return None
    # see if we can parse the time
    ts = l1.split(' ')[3:]
    # Fri Sep 28 10:42:25 EDT 2018 ->
    # 2018 Sep 28 10:42:25
    # strip carriage return
    year = ts[-1][:-1]
    timeStr = ' '.join([year, ts[0], ts[1], ts[2]])
    fmt = "%Y %b %d %H:%M:%S"
    try:
        startDt = datetime.strptime(timeStr, fmt)
    except:
        print "Could not parse date from line", l1
        print timeStr
        return None

    # last two lines should look like:
    # Wrote file =  <path>
    # DONE
    # example path:
    # /home/sandboxes/monctrl/workspace/poof/release_18.2/test/TPTCSOOF_080921/OOF/s19-1-db-000/Solutions/Plots/s19-1-plot12fits.png
    if ls[-1] != "DONE\n":
        print "Unexpected last line", ls[-1]
        return None

    ls2 = ls[-2]
    exp = ' '.join(ls2.split(' ')[:3])
    if exp != "Wrote file =":
        print "Unexpected next to last line: ", ls2
        return None

    # remove carriage return from filename
    fn = ls2.split(' ')[4][:-1]

    # ignore stuff done by unit tests because the file won't even be there
    # in fact, ignore ANYTHING not put in the data dir
    if "/home/gbtdata" not in fn:
        return None

    # get creation time of this file
    if not os.path.isfile(fn):
        print "Could not find file", fn
        return None

    mtime = os.path.getmtime(fn)
    mdt = datetime.fromtimestamp(mtime)

    # print "file modified on", fn, mdt

    # return the elapsed time from when the log think it started
    # and when this file got written
    return (mdt - startDt).seconds        


def mineOOFLogs(path):
    "Go through all OOF logs in given path, gather and report"

    # gather logs
    fs = os.listdir(path)

    # our log files match the patter oof-<username>-<timestamp>.out
    fs = [f for f in fs if f[:3] == 'oof' and f[-3:] == 'out']

    print "Num logs in path", path, len(fs)

    # parse logs
    secs = []
    for i, f in enumerate(fs):
        print i, f
        td = parseLog(path, f)
        if td is not None:
            secs.append(td)

    # report results
    print "Computed elapsed time (secs) for %d OOFs" % len(secs)
    print "min: %7.2f, max: %7.2f, mean: %7.2f, std: %7.2f" % (np.min(secs),
                                                               np.max(secs),
                                                               np.mean(secs),
                                                               np.std(secs))

def main():
    path = "/home/oof/logs"
    mineOOFLogs(path)

if __name__ == "__main__":
    main()
