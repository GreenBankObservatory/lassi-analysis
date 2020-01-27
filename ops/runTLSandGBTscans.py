import logging
import time
import os

from astropy.io import fits

from .pyTLS import TLSaccess
from .runTLS import runOneScan, configureScanner
from ygor.ActiveSurfaceDevice import ActiveSurface
from ygor.GrailClient import GrailClient
from . import runTLSLogging

# Initialize the logging configuration
logging.config.dictConfig(runTLSLogging.config)
logger = logging.getLogger(__name__)

def checkOnConditions(activeSurface):

    # check on the active surface elevation
    el = float(activeSurface.GetValue("elevation"))
    logger.debug("Active Surface Elevation: %f" % el)

    # make sure the Scan Coordinator does not include the active surf
    gc = GrailClient(host = "wind")
    sc = gc.create_manager("ScanCoordinator")
    ss = sc.get_parameter("subsystemSelect")
    asInSc = int(ss['subsystemSelect']['subsystemSelect']['ActiveSurface']['value'])
    logger.debug("Active Surface in Scan Coordinator?: %d" % asInSc)
    if asInSc != 0:
        logger.error("Active Surface is in the Scan Coordinator!!!")

    # check on the wind velocity
    w3 = gc.create_manager("Weather,Weather3")
    w = w3.get_sampler("weather3")
    windMps = float(w['weather3']['weather3']['WINDVEL_1']['value'])
    logger.debug("Weather3 Wind Velocity (m/s): %f" % windMps)

def runConfigurations():

    resValues = ["500mm@100m", 
                 "250mm@100m",
                 "125mm@100m", 
                 "63mm@100m",
                 "31mm@100m",
                 "16mm@100m",
                 "8mm@100m"]

    sensValues = ['Normal', 'High']

    modeValues = ["Speed", "Range", "Medium Range", "Long Range"]

    # I think we can leave sensitivity and scan mode as they are (Normal and Speed), but it might be interesting to
    # try 31mm@100m (it should take four times as long to complete a scan)
    # or 125mm@100m (three times faster), and see if it improves or degrades things. 

    # res = ["31mm@100m", "63mm@100m", "125mm@100m"]
    res = ["63mm@100m", "125mm@100m"]
    for r in res:
        assert r in resValues

    # do a dumb ass easy to see zernike
    zs = [(5, 5000.)]
    for r in res:
        runTLSandGBTscans(zs, res=r, repeat=False)



def runOtherConfigurations():

    # do a dumb ass easy to see zernike
    zs = [(5, 5000.)]

    sensValues = ['Normal', 'High']

    modeValues = ["Speed", "Range"]

    for sensitivity in sensValues:
        for mode in modeValues:
            runTLSandGBTscans(zs,
                              scan_mode=mode,
                              sensitivity=sensitivity,
                              repeat=False)
    
def runTLSandGBTscans(zernikes,
                      twoFlat=False,
                      repeat=True,
                      scan_number=None,
                      res=None,
                      scan_mode=None,
                      sensitivity=None):

    lassiPath = "/home/sandboxes/pmargani/LASSI/data/9oct2019"

    a = TLSaccess("lassi.ad.nrao.edu")
    print (a.get_status())

    if scan_number is not None:
        a.set_scan_number(scan_number)

    # configure scanner
    if res is None:
        res="63mm@100m"
    if scan_mode is None:    
        scan_mode = "Speed"
    if sensitivity is None:    
        sensitivity = "Normal"

    # fuck: I fat fingered this: it should be 270, but 
    # there's no changing it now ...
    # cntr_az=279
    cntr_az=270
    cntr_el=45

    # az_fov = 180
    az_fov = 360
    el_fov = 90

    proj = "9oct2019"

    configureScanner(a, proj, res, sensitivity, scan_mode, cntr_az, cntr_el, az_fov, el_fov)

    # set the scan number?

    # configure Active Surface
    asm = ActiveSurface()
    asm.TurnOnThermalZernikes()
    asm.ZeroThermalZernikes()

    keepGoing = True
    while keepGoing:
        try:    
            # if twoFlat:
                # runTwoFlatMntScans(a=a)

            for zi, zValue in zernikes:
                
                # reference scan: make sure active surface is set properly
                checkOnConditions(asm)

                # deactivate all zernikes
                asm.ZeroThermalZernikes()
                asm.startScan()

                # wait for scan to finish
                time.sleep(8)    
                asScanNumber = int(asm.GetValue("scanNumber"))
                logger.debug("Active Surface ref scan number: %d" % asScanNumber)

                logger.debug("Running reference scan")
                refPtxFile = runOneScan(a, path=lassiPath)

                # process the ref scan - save name of results

                # Signal scan:
                checkOnConditions(asm)

                # now inject the appropriate zernikie
                logger.debug("Active Surface setting zernike %d to %f" % (zi, zValue))
                asm.SetThermalZernike(zi, zValue)
                asm.startScan()

                # wait for scan to finish
                time.sleep(8)    
                asScanNumber = int(asm.GetValue("scanNumber"))
                logger.debug("Active Surface ref scan number: %d" % asScanNumber)

                logger.debug("Running signal scan")
                sigPtxFile = runOneScan(a, path=lassiPath)

                # process the signal scan - save name of all results
                # in theory, this reproduces the zernike we sent in above

                # now send the difference of the commanded and calculates z's
                # to the active surface

                # now run a third lassi scan - no real time processing needing!

                # keep going?
                if not repeat:
                    keepGoing = False

        except KeyboardInterrupt:
            print ("User Break")
            a.cntrl_exit()
            keepGoing = False
        # finally:
            # a.cntrl_exit()
            # keepGoing = False


    a.cntrl_exit()

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
    print (fs)
    fs = sorted([f for f in fs if f[-4:] == 'fits'])
    print ("fits files: ", fs)

    for f in fs:
        fpath = os.path.join(fitsPath, f)
        info = readActiveSurfaceFITS(fpath)
        print (info)


def runTwoFlatMntScans(a=None,
                      scan_number=None,
                      res=None,
                      scan_mode=None,
                      sensitivity=None):

    lassiPath = "/home/sandboxes/pmargani/LASSI/data/9oct2019"

    if a is None:
        a = TLSaccess("lassi.ad.nrao.edu")
        print (a.get_status())

    if scan_number is not None:
        a.set_scan_number(scan_number)

    # configure scanner
    if res is None:
        res="63mm@100m"
    if scan_mode is None:    
        scan_mode = "Speed"
    if sensitivity is None:    
        sensitivity = "Normal"

    # fuck: I fat fingered this: it should be 270, but 
    # there's no changing it now ...
    # cntr_az=279
    cntr_az=270
    cntr_el=45

    az_fov = 180
    # az_fov = 360
    el_fov = 90

    proj = "9oct2019"


    # # get the first half of the dish
    logger.debug("Running first of two flat scans.")
    configureScanner(a, proj, res, sensitivity, scan_mode, cntr_az, cntr_el, az_fov, el_fov)
    runOneScan(a, path=lassiPath)

    # get the first half of the dish
    cntr_az = 90    
    configureScanner(a, proj, res, sensitivity, scan_mode, cntr_az, cntr_el, az_fov, el_fov)
    logger.debug("Running second of two flat scans.")
    runOneScan(a, path=lassiPath)

    a.cntrl_exit()

def main():

    # runTwoFlatMntScans()
    # checkActiveSurfaceFITS("TINT")
    # return

    # zs = [(5, 5000.)]
    # runTLSandGBTscans(zs, repeat=False, scan_number=342)
    # return

    # We could try the following terms:

    # Z7 at 1000 and 500 (asymmetrical)
    # Z13 at 1000 and 500 (symmetrical)
    # Z15 at 1000 and 500 (asymmetrical)     
    zis = [7, 13, 15]
    zamps = [1000., 500., 150.]
    zs = []
    for zi in zis:
        for zamp in zamps:
            zs.append((zi, zamp)) 
    print ("running with zernikes: ",zs)
    # zs = [(5, 5000)]        
    runTLSandGBTscans(zs, twoFlat=False, repeat=True, scan_number=517)
    #checkActiveSurfaceFITS("JUNK")
   
    #checkOnConditions(ActiveSurface())
    # testOtherConfigurations()

if __name__ == '__main__':
    main()
