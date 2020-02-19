import os

from astropy.io import fits

from ProjectScanLog import ProjectScanLog

def getNonZeroZernikes(ext):
    names = ext.data.field('NAME')
    values = ext.data.field('VALUE')

    nzDict = {}
    # print non-zero values
    nzValues = [(i, v) for i, v in enumerate(values) if v > 0.0 ]
    if len(nzValues) > 0:
        # print("Non-zero zernikies for", title)
        for i, v in nzValues:
            print("%d: %5.2f" % (i, v))
            nzDict[i] = v
    return nzDict
            
def getLassiInfo(projPath, path):
    p = os.path.join(projPath, "LASSI", path)

    if not os.path.isfile(p):
        print ("Does not exist: '%s'" % p)
        return {}

    hdus = fits.open(p)

    refScan = hdus[0].header["REFSCAN"]

    return {"refScan": refScan}

def getActiveInfo(projPath, path):
    p = os.path.join(projPath, "ActiveSurfaceMgr", path)
    if not os.path.isfile(p):
        print ("Does not exist: '%s'" % p)
        return {}

    hdus = fits.open(p)

    return getNonZeroZernikes(hdus[1])
    

def viewLassiTestScans(path):
    p = ProjectScanLog(path)
    p.open()

    scans = {}

    print (p.scans)
    scanNums = sorted(list(p.scans.keys()))
    # for scanNum, devices in p.scans.items():
    for scanNum in scanNums:

        # lassiScan = None
        print ("")
        print("Scan Num: ", scanNum)
        devices = p.scans[scanNum]
        deviceList = list(devices.keys())
        if len(deviceList) == 1:
            print(deviceList)
            if deviceList[0] == 'ActiveSurfaceMgr':
                # lassiScan = True
                actInfo = getActiveInfo(path, devices["ActiveSurfaceMgr"])      
            elif deviceList[0] == 'LASSI':
                # lassiScan = False
                lassiInfo = getLassiInfo(path, devices["LASSI"])
                print(lassiInfo)    
            else:
                print("shit")

            # if lassiScan is not None and lassiScan:
            # if lassiScan is not None and not lassiScan:
        # scans[scanNum] = {
        #     "device": devices[0],
        #     ""
        # }
        

def main():
    # path = "/export/simdata/TINT_200203"
    # path = "/export/simdata/TINT_3"
    # path = "/export/simdata/TINT_200210"
    #path = "/export/simdata/TINT_200211"
    # path = "/home/gbtdata/TINT_200214"
    # path = "/home/gbtdata/TLASSI_200219"
    import sys
    path = sys.argv[1]
    viewLassiTestScans(path)

if __name__ == '__main__':
    main()

