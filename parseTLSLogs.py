from glob import glob
from pprint import pprint
from datetime import datetime
import os
import pickle

from astropy.io import fits

LASSI_SCAN = "Running scan with scan number:"
LASSI_EXPORT = "exporting to file"
PTX_SIZE = "PTX Bytes"
LASSI_SCAN_TIME = "Scan Mins"
LASSI_EXPORT_TIME = "Total Export Mins"
ACTSRF_SCAN = "Active Surface ref scan number:"
LASSI_CONFIGURE = "Configuring Scanner:"
ACTSRF_EL = "Active Surface Elevation:"
ACTSRF_SC = "Active Surface in Scan Coordinator?:"
ACTSRF_Z = "Active Surface setting zernike"
WIND = "Weather3 Wind Velocity (m/s):"
SIGNAL = "Running signal scan"
REF = "Running reference scan"
FLAT = "two flat scans"

def readActiveSurfaceFITS(fpath):


    info = {}

    hdus = fits.open(fpath)
    hdr = hdus[0].header

    info['SCAN'] = hdr['SCAN']
    info['DATE-OBS'] = hdr['DATE-OBS']
    
    info['ZERNIKES'] = []

    if len(hdus) <= 1:
        return info

    try:
        d = hdus['THRMZERN']
    except KeyError:
        return info

    d = d.data

    zs = d['Name']
    values = d['Value']
    nonZeros = []
    for i, z in enumerate(zs):
        if values[i] != 0.:
            nonZeros.append((z, values[i]))
    info['ZERNIKES'] = nonZeros

    return info        


def checkActiveSurfaceFITS(proj):

    dpath = "/home/gbtdata"    
    # proj = "TINT"
    fitsPath = os.path.join(dpath, proj, 'ActiveSurfaceMgr')

    # get fits files
    fs = os.listdir(fitsPath)
    print fs
    fs = sorted([f for f in fs if f[-4:] == 'fits'])
    print "fits files: ", fs

    infos = []
    for f in fs:
        fpath = os.path.join(fitsPath, f)
        info = readActiveSurfaceFITS(fpath)
        print info
        info["filename"] = os.path.basename(fpath)
        info["proj"] = proj
        infos.append(info)
    return infos
        
def getZernike(l):
    zStr = getValue(l, ACTSRF_Z)
    i = zStr.find("to")
    return int(zStr[:(i-1)]), float(zStr[(i+3):])

def getActiveSurfaceScanNumber(l):
    # i = l.find(ACTSRF_SCAN)
    # if i == -1:
    #     return None
    # s = i + len(ACTSRF_SCAN) + 2
    # return int(l[s:-1])
    return int(getValue(l, ACTSRF_SCAN))

def getDateTime(l):
    "2019-10-11 11:56:42,410 - * -> datetime(2019, 10, 11, 11, 56, 42)"
    dtStr = l[:19]
    return datetime.strptime(dtStr, "%Y-%m-%d %H:%M:%S")

def getValue(l, key):
    "* - key: value -> value"
    i = l.find(key)
    if i == -1:
        return None
    s = i + len(key) + 1
    return l[s:-1]

def getFileName(l):
    "2019-10-11 11:56:42,410 - runTLS.runOneScan - DEBUG - exporting file to <path> -> <path>"
    # i = l.find(LASSI_EXPORT)
    # if i == -1:
    #     return None
    # s = i + len(LASSI_EXPORT) + 2
    # return l[s:-1]
    return getValue(l, LASSI_EXPORT)

def getScanNum(l):
    "2019-10-11 11:56:42,410 - runTLS.runOneScan - DEBUG - Running scan with scan number: i501 -> 501"

    i = l.find(LASSI_SCAN)
    if i == -1:
        return (None, None)
    s = i + len(LASSI_SCAN) + 2

    
    return int(l[s:]), getDateTime(l)
        
def getMsg(l):

    ps = l.split(" ")
    return " ".join(ps[7:])

def getConfigValue(l):
    "key: value -> value"
    configStr = getMsg(l)
    i = configStr.find(":")
    return configStr[(i+1):-1]

def getConfiguration(i, l, ls):
    assert LASSI_CONFIGURE in l
    if len(ls) < i+10:
        return {}
    proj = getConfigValue(ls[i+1])
    res = getConfigValue(ls[i+2])
    sen = getConfigValue(ls[i+3])
    mod = getConfigValue(ls[i+4])
    az = float(getConfigValue(ls[i+5]))
    el = float(getConfigValue(ls[i+6]))
    azFov = float(getConfigValue(ls[i+7]))
    elFov = float(getConfigValue(ls[i+8]))

    return {
        "project" : proj,
        "resolution" : res,
        "sensitivity" : sen,
        "scan_mode" : mod,
        "cntr_az" : az,
        "cntr_el" : el,
        "az_fov" : azFov,
        "el_fov" : elFov
    }

def parseLog(fpath, scans):

    with open(fpath, 'r') as f:
        ls = f.readlines()

    currentScan = prevScan = None

    config = stype = elev = inSC = windMps = ptxFn = asScanNum = None
    z = zValue = None

    for i, l in enumerate(ls):
        print l
        if LASSI_CONFIGURE in l:
            config = getConfiguration(i, l, ls)
        if ACTSRF_EL in l:
            print "elevation: ", l
            elev = float(getValue(l, ACTSRF_EL))
            print elev
        if ACTSRF_SC in l:
            print l
            print getMsg(l)
            inSC = int(getValue(l, ACTSRF_SC))
        if WIND in l:
            windMps = float(getValue(l, WIND))           
        if REF in l:
            stype = "REF"
        if SIGNAL in l:
            stype = "SIGNAL"
        if FLAT in l:
            stype = "FLAT"            
        if LASSI_SCAN in l:
            scanNum, dt = getScanNum(l)
            print scanNum, dt
            if scanNum is not None:
                prevScan = currentScan
                currentScan = scanNum
                # k = (scanNum, dt)
                k = scanNum
                # if k not in scans.keys():
                    # scans[k] = {}
                scans[k] = {
                    "configuration": config,
                    "elevation": elev,
                    "AS in SC?": inSC,
                    "wind(m/s)": windMps,
                    "AS Scan #": asScanNum,
                    "type": stype,
                    "scan #": scanNum,
                    "dt": dt
                }
                if stype != "SIGNAL":
                    scans[k]["zernike index"] = None
                    scans[k]["zernike value"] = None
                else:
                    scans[k]["zernike index"] = z
                    scans[k]["zernike value"] = zValue

        if LASSI_EXPORT in l:
            if l[-4:-1] == 'ptx':
                ptxFn = getFileName(l)
                if ptxFn is not None:
                    ptx = os.path.basename(ptxFn)
                    # scan_timestamp.ptx
                    ptxScanNum = int(ptx.split('_')[0])
                    assert ptxScanNum == currentScan
                    scans[currentScan]['ptx'] = ptx
        if PTX_SIZE in l:
            scans[currentScan]['ptx (bytes)'] = int(getValue(l, PTX_SIZE))
        if LASSI_SCAN_TIME in l:
            scans[currentScan]['scan (mins)'] = float(getValue(l, LASSI_SCAN_TIME))
        if LASSI_EXPORT_TIME in l:
            scans[currentScan]['export (mins)'] = float(getValue(l, LASSI_EXPORT_TIME))
        if ACTSRF_SCAN in l:
            asScanNum = getActiveSurfaceScanNumber(l)
            # this appears before the LASSI scan number:
            # if prevScan in scans.keys():
                # scans[prevScan]["activeSurfaceScan"] = asScanNum
        if ACTSRF_Z in l:
            z, zValue = getZernike(l)

    return scans

def parseLogs(path):


    # TBF: need to change name of log
    fs = sorted(glob("%s/teleSpy*.log" % path))
    # print fs

    scans = {}
    for f in fs:
        if "2019-10" in f:
            parseLog(f, scans)

    pprint(scans)
    print "num scans: ", len(scans.keys())

    # now update these with the FITS info
    fits = checkActiveSurfaceFITS("TINT")
    fits2 = checkActiveSurfaceFITS("JUNK")
    fits.extend(fits2)

    # create lookup table
    fitsInfo = {}
    for f in fits:
        fitsInfo[f['SCAN']] = f

    # now merge the two
    for scanNum, scanInfo in scans.items():

        asScanNum = scanInfo['AS Scan #']
        if asScanNum is not None:
            fi = fitsInfo[asScanNum]
            assert fi['SCAN'] == scanInfo['AS Scan #']
            if 'ZERNIKES' in fitsInfo:
                zs = fitsInfo['ZERNIKES']
                if len(zs) == 0:
                    assert scanInfo['zernike index'] is None
                    assert scanInfo['zernike value'] is None
                elif len(zs) == 1:
                    zStr, zValue = zs[0]
                    z = int(zStr[1:])
                    assert scanInfo['zernike index'] == z
                    assert scanInfo['zernike value'] == zValue

            scanInfo['AS FITS'] = fitsInfo[asScanNum]

    pprint(scans)

    print "num scans: ", len(scans.keys())
    # print fits1
    return scans

def main():
    path = "/home/scratch/pmargani/LASSI/runTLSlogs"
    scans = parseLogs(path)

    with open('lassiScans9oct2019.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(scans, f, pickle.HIGHEST_PROTOCOL)

    # l = "2019-10-11 11:56:30,831 - runTLS.configureScanner - DEBUG - Scan Mode: Speed"
    # print getMsg(l)

if __name__ == '__main__':
    main()

