import os

from astropy.io import fits
import numpy as np

from ProjectScanLog import ProjectScanLog

def processGO(fpath):
    # print fpath
    info = {}

    if not os.path.isfile(fpath):
        print "WARNING: no GO Fits at: ", fpath
        return info

    hs = fits.open(fpath)
    hd = hs[0].header


    keys = [
        'SCAN',
        'RECEIVER',
        'PROCNAME',
        'PROCTYPE',
        'PROCSCAN',
        'PROCSIZE',
        'PROCSEQN'
    ]

    for key in keys:
        if key in hd:
            info[key] = hd[key]
        else:
            print "Header missing key in ", key, fpath    

    if info == {}:
        print "WARNING: no info found in GO FITS", fpath

    return info            

def processProject(path, projDir):
    projPath = os.path.join(path, projDir)
    proj = ProjectScanLog(projPath)
    proj.open()
    if proj.scans == {}:
        print "Nothing to process in ", projPath
        return

    goInfos = []
    for scanNum, devices in proj.scans.items():
        if 'GO' in devices:
            goFn = devices['GO']
            # processGO(goFn)
            # print "goFN", goFn
            goPath = os.path.join(projPath, 'GO', goFn)
            goInfo = processGO(goPath)
            goInfo['PROJ'] = projDir
            goInfo['FILE'] = goFn
            # update this with the duration
            if scanNum in proj.scanTimes:
                goInfo['DURSECS'] = proj.scanTimes[scanNum]['durSecs']
            goInfos.append(goInfo)
        else:
            print "scanNum does not have a GO FITS", scanNum
            continue

    return goInfos

def mineAstridProcedures(path):
    goInfos = []

    # gather data
    projDirs = os.listdir(path)
    print "Found %d Projects" % len(projDirs)
    for i, projDir in enumerate(projDirs):
        print "Project: %d %s" % (i, projDir)
        goInfo = processProject(path, projDir)
        if goInfo is not None:
            goInfos.extend(goInfo)

    # mine data
    oofData = {}
    pntData = {}
    fcsData = {}
    for goInfo in goInfos:
        if 'PROCTYPE' in goInfo and 'RECEIVER' in goInfo and 'PROCNAME' in goInfo:
            if goInfo['PROCTYPE'] == 'OOFMAP':
                rx = goInfo['RECEIVER']
                secs = goInfo['DURSECS']
                if rx not in oofData:
                    oofData[rx] = []
                oofData[rx].append((goInfo['PROJ'], goInfo['FILE'], secs))
            if goInfo['PROCTYPE'] == 'POINTING':
                rx = goInfo['RECEIVER']
                secs = goInfo['DURSECS']
                if rx not in pntData:
                    pntData[rx] = []
                pntData[rx].append((goInfo['PROJ'], goInfo['FILE'], secs))
            if goInfo['PROCTYPE'] == 'CALIBRATION' and goInfo['PROCNAME']=='FocusSubreflector':
                rx = goInfo['RECEIVER']
                secs = goInfo['DURSECS']
                if rx not in fcsData:
                    fcsData[rx] = []
                fcsData[rx].append((goInfo['PROJ'], goInfo['FILE'], secs))

                
    # report results
    print "All Times in Seconds:"  
    print ""
    print "OOF Results: "
    print "%20s %8s %9s %9s %9s" % ("RX", "# scans", "total", "mean", "std")
    for rx, info in oofData.items():
        # print rx, info
        durSecs = [i[2] for i in info]
        print "%20s %8d %9.2f %9.2f %9.2f" % (rx, len(durSecs), np.sum(durSecs), np.mean(durSecs), np.std(durSecs))                    

    print ""
    print "POINTING Results: "
    print "%20s %8s %9s %9s %9s" % ("RX", "# scans", "total", "mean", "std")
    for rx, info in pntData.items():
        # print rx, info
        durSecs = [i[2] for i in info]
        print "%20s %8d %9.2f %9.2f %9.2f" % (rx, len(durSecs), np.sum(durSecs), np.mean(durSecs), np.std(durSecs))                    

    print ""
    print "FOCUS Results: "
    print "%20s %8s %9s %9s %9s" % ("RX", "# scans", "total", "mean", "std")
    for rx, info in fcsData.items():
        # print rx, info
        durSecs = [i[2] for i in info]
        print "%20s %8d %9.2f %9.2f %9.2f" % (rx, len(durSecs), np.sum(durSecs), np.mean(durSecs), np.std(durSecs))                    

def main():
    path = "/home/gbtdata"
    mineAstridProcedures(path)

if __name__ == '__main__':
    main()
