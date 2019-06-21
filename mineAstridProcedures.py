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

    return proj, goInfos

def getOOFProjInfo(project, goInfo):
    """
    Analyze when OOF scans run in a project.
       * project is a ProjectScanLog object
       * goInfo is a list of dictionaries with info from GO FITS
    """

    # get info on whole project
    scanNums = sorted(project.scanTimes.keys())
    # startTimes = sorted([ts['start'] for scanNums, ts in project.scanTimes.items()])
    startTimes = project.projStartDts
    startTime = min(startTimes)
    endTime = max(startTimes)

    # get info on just OOF scans
    # and group together adjacent scans
    # oofScans = []
    oofGroups = None
    oofGroup = None
    for go in goInfo:
        if 'PROCTYPE' not in go:
            continue
        # print go['SCAN'], go['PROCTYPE']    
        if go['PROCTYPE'] != 'OOFMAP':
            if oofGroup is not None:
                if oofGroups is None:
                    oofGroups = []
                oofGroups.append(oofGroup)
            oofGroup = None
            continue
        scanNum = go['SCAN']
        rx = go['RECEIVER']
        startDt = project.scanTimes[scanNum]['start']
        endDt = project.scanTimes[scanNum]['end']
        # oofScans.append((scanNum, rx, startDt, endDt))
        if oofGroup is None:
            # start the new group!
            print "new group for: ", project.projPath
            oofGroup = []
        oofGroup.append((scanNum, rx, startDt, endDt))    

    return oofGroups

def mineAstridProcedures(path):
    goInfos = []
    oofProjInfos = []

    #/home/gbtdata/AGBT19A_092_07
    # gather data
    projDirs = os.listdir(path)
    # projDirs = ['AGBT19A_092_07']
    # projDirs = ['AGBT17B_101_07']
    # projDirs = ['AGBT18B_383_04']
    # projDirs = ['AGBT17B_325_39']
    print "Found %d Projects" % len(projDirs)
    for i, projDir in enumerate(projDirs):
        print "Project: %d %s" % (i, projDir)
        proc = processProject(path, projDir)
        if proc is not None:
            proj, goInfo = proc
            goInfos.extend(goInfo)
            oofProjInfo = getOOFProjInfo(proj, goInfo)
            if oofProjInfo is not None:
                oofProjInfos.append((proj, oofProjInfo))

    # print oofProjInfos
    rxOofGaps = {}
    rxOofDurs = {}
    gapProjs = noGapProjs = 0
    for proj, projOofInfo in oofProjInfos:
        print "for project:"
        # print projOofInfo
        if len(projOofInfo) > 1:
            gapProjs += 1
            gaps = []
            currGap = None
            for i in range(len(projOofInfo[:-1])):
                thisGroup = projOofInfo[i]
                nextGroup = projOofInfo[i+1]
                sn1, rx, _, gapStartDt = thisGroup[-1]
                sn2, _, gapEndDt, _ = nextGroup[0]
                gaps.append((rx, gapStartDt, gapEndDt))
                print gapStartDt, gapEndDt
                gapDurHrs = (gapEndDt - gapStartDt).seconds/(60.*60.)    
                print "gap (hrs): ", rx, sn1, sn2, gapDurHrs
                if rx not in rxOofGaps:
                    rxOofGaps[rx] = []
                rxOofGaps[rx].append(gapDurHrs)    
        elif len(projOofInfo) == 1:
            print projOofInfo
            noGapProjs += 1
            group = projOofInfo[0][-1]
            sn, rx, startDt, endDt = group
            if rx not in rxOofDurs:
                rxOofDurs[rx] = []
            durHrs = (proj.projEndDt - endDt).seconds/(60.*60.) 
            rxOofDurs[rx].append(durHrs)  

    print ""
    print "Num gap Projs vs. no gap projs: ", gapProjs, noGapProjs
    print ""
    print "Gaps (hrs) between OOF scan groups (> 1 hr):"
    print "%20s %7s %7s %7s %7s %7s" % ("Rx", "#projs", "Min", "Max", "Mean", "STD")
    for rx, durs in rxOofGaps.items():
        # print "rx gaps: ", rx, np.min(durs), np.max(durs), np.mean(durs), np.std(durs)
        # now filter out suspicious outliers:
        fdurs = [d for d in durs if d > 1.0]
        if len(fdurs) > 0:
            print "%20s %7d %7.2f %7.2f %7.2f %7.2f" % (rx,
                                                   len(fdurs),     
                                                   np.min(fdurs),
                                                   np.max(fdurs),
                                                   np.mean(fdurs),
                                                   np.std(fdurs))

    print ""
    print "Time (hrs) between OOF scan groups and end of project (> 1 hr):"
    print "%20s %7s %7s %7s %7s %7s" % ("Rx", "#projs", "Min", "Max", "Mean", "STD")
    for rx, durs in rxOofDurs.items():
        fdurs = [d for d in durs if d > 1.0]
        if len(fdurs) > 0:
            print "%20s %7d %7.2f %7.2f %7.2f %7.2f" % (rx,
                                                        len(fdurs),
                                                   np.min(fdurs),
                                                   np.max(fdurs),
                                                   np.mean(fdurs),
                                                   np.std(fdurs))

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
    print ""
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
