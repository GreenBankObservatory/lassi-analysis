"This is the module for navigating information about March 27 scans"

from datetime import datetime

REF = 'ref'

# Here we organize things by scan number, including whatever
# we may know about each scan, including:
#   * 'ActSrfScan': we ran an Active Surface scan for each LASSI scan,
#      this is it's M&C scan number
#   * ActSrfFile: this is the FITS file name for the Active Surface scan
#   * dt: this is the creation timestamp of the FITS file (EST)
#   * type: 'ref' for reference, or 'Z4' for a type of signal
march27scans = {
    9: {
        'ActSrfScan':3, 
        'dt': 'Mar 27 11:46',
        'ActSrfFile': '2019_03_27_15:46:00.fits',
        'type': REF
        },
    10: {
        'ActSrfScan':4, 
        'dt': 'Mar 27 11:50',
        'ActSrfFile': '2019_03_27_15:50:45.fits',
        'type': REF,
        },
    11: {
        'ActSrfScan':5, 
        'dt': 'Mar 27 11:55',
        'ActSrfFile': '2019_03_27_15:55:28.fits',
        'type': 'Z4',
        },
    12: {
        'ActSrfScan':6, 
        'dt': 'Mar 27 12:00',
        'ActSrfFile': '2019_03_27_16:00:50.fits',
        'type': REF,
        },
    13: {
        'ActSrfScan':7, 
        'dt': 'Mar 27 12:06',
        'ActSrfFile': '2019_03_27_16:06:08.fits',
        'type': 'Z21',
        },
    14: {
        'ActSrfScan':8, 
        'dt': 'Mar 27 12:10',
        'ActSrfFile': '2019_03_27_16:10:05.fits',
        'type': REF,
        },
    15: {
        'ActSrfScan':9, 
        'dt': 'Mar 27 12:17',
        'ActSrfFile': '2019_03_27_16:17:17.fits',
        'type': 'Z8',
        },
    16: {
        'ActSrfScan':10,
        'dt': 'Mar 27 12:21',
        'ActSrfFile': '2019_03_27_16:20:59.fits',
        'type': REF,
        },
    17: {
        'ActSrfScan':11,
        'dt': 'Mar 27 12:25',
        'ActSrfFile': '2019_03_27_16:25:48.fits',
        'type': 'Z13',
        },
    18: {
        'ActSrfScan':12,
        'dt': 'Mar 27 12:31',
        'ActSrfFile': '2019_03_27_16:31:02.fits',
        'type': REF,
        },
    19: {
        'ActSrfScan':13,
        'dt': 'Mar 27 12:40',
        'ActSrfFile': '2019_03_27_16:40:48.fits',
        'type': 'bump',
        },
    20: {
        'ActSrfScan':14,
        'dt': 'Mar 27 12:48',
        'ActSrfFile': '2019_03_27_16:48:56.fits',
        'type': REF,
        },
}

projPath = "/home/gbtdata/TINT/ActiveSurfaceMgr"

def dtStr2dt(dtStr):
    "'Mar 27 11:46' -> datetime"
    return datetime.strptime('2019 ' + dtStr, '%Y %b %d %H:%M')


def getStartEndTimes(thisScan, nextScan):
    return (dtStr2dt(march27scans[thisScan]['dt']),
            dtStr2dt(march27scans[nextScan]['dt']))
