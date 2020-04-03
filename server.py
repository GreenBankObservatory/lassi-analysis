import shutil
import zmq
import random
import sys
import os
import time
from datetime import datetime
from multiprocessing import Process, Value
import subprocess
import logging, logging.config

import numpy  as np
import msgpack
import msgpack_numpy

import lassiTestSettings as settings
from ops.pyTLS import TLSaccess
from processPTX import getRawXYZ, processNewPTXData
from lassiAnalysis import extractZernikesLeicaScanPair
from ZernikeFITS import ZernikeFITS
from ops.getConfigValue import getConfigValue
from plotting import plotZernikes
from SmoothedFITS import SmoothedFITS
from utils.utils import splitXYZ
from gpus import loadParallelGPUFiles, loadGPUFiles
from lassiAnalysis import imageSmoothedData

from settings import GPU_MULTI_HOSTS, GPU_MULTI_PATHS

import ops.runTLSLogging as runTLSLogging

# Init the logging configuration
logging.config.dictConfig(runTLSLogging.config)
logger = logging.getLogger(__name__)

# Get a number of settings from the config files:
# DATADIR = "/home/sandboxes/pmargani/LASSI/data"
# from system.conf:
DATADIR = getConfigValue(".", "YGOR_DATA")
logger.debug("Writing FITS files to YGOR_DATA: %s" % DATADIR)

# from LASSI.conf:
# TLS_HOST = "galileo.gb.nrao.edu"
lc = "LASSI.conf"
TLS_HOST = getConfigValue(".", "tlsServerHost", configFile=lc)
SIM_RESULTS = 1 == int(getConfigValue(".", "analysisResultsSimulated", configFile=lc))
SIM_INPUTS = 1 == int(getConfigValue(".", "analysisInputsSimulated", configFile=lc))
SIM_REF_INPUT = getConfigValue(".", "analysisRefInput", configFile=lc)
SIM_SIG_INPUT = getConfigValue(".", "analysisSigInput", configFile=lc)
SIM_REF_SMOOTH_RESULT = getConfigValue(".", "analysisSmoothRefResult", configFile=lc)
SIM_SIG_SMOOTH_RESULT = getConfigValue(".", "analysisSmoothSigResult", configFile=lc)
SIM_ZERNIKE_RESULT = getConfigValue(".", "analysisZernikeResult", configFile=lc)
SIM_ZERNIKE_PNG = getConfigValue(".", "analysisZernikePng", configFile=lc)
PORT = int(getConfigValue(".", "analysisServerPort", configFile=lc))
PUB_PORT = int(getConfigValue(".", "analysisPublishPort", configFile=lc))

# let user know how we're setup
logger.debug("Starting analysis server using sim results: %s" % SIM_RESULTS)
logger.debug("Starting analysis server using sim  inputs: %s" % SIM_INPUTS)
logger.debug("For smoothing using these hosts: %s" % GPU_MULTI_HOSTS)
logger.debug("from these installations: %s" % GPU_MULTI_PATHS)

# states
READY = 0 #"READY"
WAITING_FOR_SCAN_END = 1
WAITING_FOR_DATA = 2
PROCESSING = 3 #"PROCESSING"

PROCESS_STATES = [WAITING_FOR_SCAN_END, WAITING_FOR_DATA, PROCESSING]

stateMap = {
        READY: "Ready",
        WAITING_FOR_SCAN_END: "Waiting for scan end",
        WAITING_FOR_DATA: "Waiting for data",
        PROCESSING: "Processing"
        }

def setupServerSocket(port):
    "Returns a ZMQ publishing socket using the given port"

    # pubPort = 9001
    # pubUrl = "tcp://%s:%d" % (pubHost, pubPort)
    pubUrl = "tcp://*:%d" % (port)
    logger.debug("Publishing from: %s" % pubUrl)

    ctx = zmq.Context()
    pubck = ctx.socket(zmq.PUB)
    pubck.bind(pubUrl)

    return pubck

def publishData(x, y, z):
    "Publishes the given data over a ZMQ socket"

    # we must create our server socket here because
    # everything must be in the same thread.  We can't
    # create this socket in the main thread, pass it
    # in here and use it.
    pubSocket = setupServerSocket(PUB_PORT)

    x = msgpack.packb(x, default=msgpack_numpy.encode)
    y = msgpack.packb(y, default=msgpack_numpy.encode)
    z = msgpack.packb(z, default=msgpack_numpy.encode)

    # TBF: this is a kluge to 'wake up' the client subscriber,
    # which often doesn't receive the first thing published
    logger.debug("publishing primer")
    pubSocket.send_multipart([b"PRIMER"])
    time.sleep(3)

    logger.debug("publishing data")
    data = [b"HEADER",
      # TBF: replace with actual header info
      b"HEADER_DATA",
      # the time info is not used, but must be here
      b"TIME_ARRAY",
      b"TIME_ARRAY_DATA",
      b"X_ARRAY",
      x,
      b"Y_ARRAY",
      y, 
      b"Z_ARRAY",
      z,
      # This data is not used, but must be present,
      b"I_ARRAY",
      x]
                          
    pubSocket.send_multipart(data)
    logger.debug("published!")

def getMissingFiles(files):
    "Which of the given files does not exist?"
    missing = []
    for f in files:
        if not os.path.isfile(f):
            missing.append(f)
    return missing
            
def processLeicaDataStream(x,
                           y,
                           z,
                           i,
                           dts,
                           hdr,
                           ellipse,
                           rot,
                           project,
                           dataDir,
                           filename,
                           plot=True,
                           smoothOutputs=None,
                           test=False):
    """
    x, y, z: data streamed from scanner
    i: intensity values
    dts: datetime values
    hdr: header dictionary from scanner stream
    ellipse: used for filtering data (dependent on scanner position)
    dataDir: where data files get written to (eg, /home/gbtdata)
    project: same sense as GBT project (eg, TGBT19A_500_01)
    filename: basename of file products (eg, 2019_09_26_01:35:43)
    smoothOutputs: optionally override where smoothing outputs go
    test: dont smooth, it's a unit test!
    """

    # TBF: make this an optional arg as well?
    # We can lower this for testing purposes
    N = 512

    # do basic processing first: remove bad data, etc.
    # TBF: backwards compatible (bad) interface
    lines = None
    xyzi = (x, y, z, i)
    xyz, dts, intensities = processNewPTXData(lines,
                                 xyzi=xyzi,
                                 rot=rot,
                                 ellipse=ellipse,
                                 rFilter=True,
                                 iFilter=False)

    logger.debug("done pre-processing data, ready to smooth")

    # optionally override settings 
    if smoothOutputs is None:
        outputs = getGpuOutputPaths(GPU_MULTI_PATHS, "outfile")
        logger.debug("getGpuOutputPaths %s %s" % (GPU_MULTI_PATHS, outputs))
    else:
        outputs = smoothOutputs

    # when running unit tests, we may not
    # have access to GPUs for smoothing
    if not test:
        xOutputs = ["%s.x.csv" % f for f in outputs]
        logger.debug("xOutputs: %s" % xOutputs)
        # make sure that output isn't there yet
        for fn in xOutputs:
            if os.path.isfile(fn):
                logger.debug("Removing previous results: %s" % fn)
                os.remove(fn)
        
        # now publish the input to smoothing
        logger.debug("splitting data")
        x, y, z = splitXYZ(xyz)
        publishData(x, y, z)

        # and wait for it to show up.
        missing = getMissingFiles(xOutputs)
        while len(missing) > 0:
            logger.debug("Waiting on smoothing file %s" % missing[0])
            time.sleep(5)
            missing = getMissingFiles(xOutputs)

    # read in those files
    x, y, z = loadGPUFiles(outputs)

    # save this off for later use
    fitsio = SmoothedFITS()
    fitsio.setData(x, y, z, N, hdr, dataDir, project, filename)
    fitsio.write()

    smoothedFitsFilename = fitsio.getFilePath()

    if plot:
        # how should we save the image of our processed data?
        ext = "smoothed.fits"
        fn = smoothedFitsFilename[:-len(ext)] + "processed.png"
        logger.debug("Processing smoothed data, imaging to: %s" % fn)
        # process a little more and create a surface plot for diagnosis
        imageSmoothedData(x, y, z, N, filename=fn)

    return fitsio.getFilePath()

def getFITSFilePath(proj, filename):
    return os.path.join(DATADIR, proj, "LASSI", filename + ".fits")

def getRefScanFileName(scans, proj, refScanNum):
    """
    Retrieves the full path to the smoothed FITS file
    for the given reference scan number.
    proj: given project, as in /home/gbtdata/<proj>
    refScanNum: reference scan number.  Can be None.
    scans: a dictionary of scan information that we will
    mine to get the desired file name. Looks like:
       {project: { scanNumber: {refScan: true, ...}}}
    """

    if proj not in scans:
        logger.error("Proj not in scans: %s %s" % (proj, scans.keys()))
        return None

    # if the ref scan number HAS NOT BEEN specified
    if refScanNum is None or refScanNum == 0:
        # then we need to find the most recent ref scan
        pScans = scans[proj]
        newestRefScan = None
        for scanNum, scanInfo in pScans.items():
            # if it's a refScan, how old is it?
            if bool(scanInfo['refScan']):
                if newestRefScan is None:
                    newestRefScan = scanInfo
                
                elif scanInfo['timestamp'] > newestRefScan['timestamp']:
                    newestRefScan = scanInfo
        if newestRefScan is None:
            # TBF: go to the file system to look for it?
            logger.error("Cannot find refScan from %s %s %s" % (proj, refScanNum, scans))
            return None
        else:
            logger.debug("getRefScanFileName: %s" % newestRefScan)
            return newestRefScan['filepathSmoothed']

    # if it has, can we use it?
    if refScanNum not in scans[proj]:
        logger.error("Scan not in proj: %s %s" % (refScanNum, scans[proj].keys()))
        return None            

    scan = scans[proj][refScanNum]
    logger.debug(scan)

    return scan['filepathSmoothed']

def waitForScanEnd(state, a):
    "The TLS is scanning, we need to wait for that to end"

    # set the new state
    logger.debug ("in WaitForScanEnd")
    state.value = WAITING_FOR_SCAN_END
    logger.debug(stateMap[WAITING_FOR_SCAN_END])

    logger.debug ("can we get the scanner state?")
    logger.debug (a.get_status())
    logger.debug ("yes we can")

    # here we wait until the scanner's state has gotten to Ready
    # time.sleep(3)
    state = "unknown"
    while state != "Ready":
        status = a.get_status()
        state = status.state
        logger.debug("State: %s" % state)
        time.sleep(1)
    logger.debug ("done waiting for scan end")
    

def waitForData(state, a):
    "The scan is over, but we now need to get the data"

    # set the new state
    state.value = WAITING_FOR_DATA
    logger.debug(stateMap[WAITING_FOR_DATA])

    keys = a.get_results().keys()
    while len(keys) < 5:
        logger.debug("not enough keys: %s" % keys)
        time.sleep(1)
        keys = a.get_results().keys()
    logger.debug ("We have all our keys now: %s" % keys)
    return a.get_results()

def processing(state, results, proj, scanNum, refScan, refScanNum, refScanFile, filename, test=False):
    
    # make sure we've made it clear we've entered this state
    state.value = PROCESSING
    logger.debug(stateMap[PROCESSING])

    # here we can finally process the data
    time.sleep(3)

    # get x, y, z and intesity from results
    if not SIM_INPUTS:
        x = results['X_ARRAY']
        y = results['Y_ARRAY']
        z = results['Z_ARRAY']
        i = results['I_ARRAY']
        dts = results['TIME_ARRAY']
        hdrObj = results['HEADER']

        hdr = hdrObj.asdict() if not test else {}
        # hdr = hdrObj.asdict()
    else:
        simFile = SIM_REF_INPUT if bool(refScan) else SIM_SIG_INPUT
        logger.debug("Using simulated input: %s" % simFile)
        # this is not smoothed data, but it still can be read
        f = SmoothedFITS()
        f.read(simFile)

        x = f.x      
        y = f.y      
        z = f.z
        i = f.hdus[1].data.field("INTENSIT")
        dts = f.hdus[1].data.field('DMJD')
        hdr = dict(f.hdr)      

    # update the header with more metadata
    hdr['mc_project'] = proj
    hdr['mc_scan'] = scanNum
    hdr['REF'] = refScan
    hdr['REFSCAN'] = refScanNum

    # then being processing
    lines = None
    xyzi = (x, y, z, i)


    # TBF: pass this in?  From settings file?
    s = settings.SETTINGS_27MARCH2019
    # s = settings.SETTINGS_19FEB2020
    # s = settings.SETTINGS_11OCTOBER2019

    # TBF: in production this might come from config file?        
    ellipse, rot = settings.getData(s)

    # TBF: we'll get these from the manager?
    # proj = "TEST"
    # dataDir = "/home/sandboxes/pmargani/LASSI/data"
    dataDir = DATADIR

    if not SIM_RESULTS:
        fitsFile = processLeicaDataStream(x,
                           y,
                           z,
                           i,
                           dts,
                           hdr,
                           ellipse,
                           rot,
                           proj,
                           dataDir,
                           filename,
                           test=test)
    else:
        # cp the smoothed file to the right locatoin
        dest = os.path.join(dataDir, proj, 'LASSI', filename)
        dest = dest + ".smoothed.fits"
        # which file?
        simFile = SIM_REF_SMOOTH_RESULT if refScan else SIM_SIG_SMOOTH_RESULT
        logger.debug("simulating smoothed results from %s to %s" % (simFile, dest))
        shutil.copy(simFile, dest)
        fitsFile = dest

    # any more processing?
    if refScan:
        # if this is a ref scan we are done
        logger.debug("This is a ref scan, we are done")
        return


    # if it is a signal scan, we
    # need to compute Zernike's
    # find the previous refscan:
    # logger.debug("look for file for scan", refScanNum)
    sigScanFile = filename

    # TBF: where to specify this?
    #N = 100
    N = 512


    logger.debug("files: %s %s %s" % (refScanFile, sigScanFile, fitsFile))

    if not SIM_RESULTS:
        xs, ys, zs, zernikes = extractZernikesLeicaScanPair(refScanFile,
                                                            fitsFile,
                                                            n=N,
                                                            nZern=36)

        # write results to final fits file                                                        nZern=36)
        fitsio = ZernikeFITS()
        # make it clear where the ref scan came from
        hdr['REFSCNFN'] = os.path.basename(refScanFile)
        # make sure the zernikes are in microns, not meters
        hdr['ZUNITS'] = 'microns'
        logger.debug(zernikes)
        logger.debug(type(zernikes))
        # zernikes = zernikes * 1e6
        zernikes = [z * 1e6 for z in zernikes]
        fitsio.setData(xs, ys, zs, N, hdr, dataDir, proj, filename)
        fitsio.setZernikes(zernikes)
        logger.debug ("Writing Zernikes to: %s" % fitsio.getFilePath())
        fitsio.write()

        """
        # Confirm with Paul that it is alright to get rid of this.
        # now make sure there's a plot for this fits file too
        zDict = {}
        for i in range(len(zernikes)):
            k = "Z%d" % (i + 1)
            zDict[k] = zernikes[i]
        """

        # Ask for Paul's approval.
        fn = fitsio.getFilePath()[:-4] + "png"
        logger.debug ("Plotting and writing Zernikes png to: %s" % fn)
        title = "%s:%s" % (proj, scanNum)
        plotZernikes(xs, ys, zernikes, n=N, title=title, filename=fn)  

    else:
        # cp the files to the right locatoin
        # first the zernike FITS file
        dest = os.path.join(dataDir, proj, 'LASSI', filename)
        dest = dest + ".zernike.fits"
        logger.debug("simulating zernike FITS results from %s to %s" % (SIM_ZERNIKE_RESULT, dest))
        shutil.copy(SIM_ZERNIKE_RESULT, dest)

        # then copy over the image file
        dest = dest[:-4] + "png"
        logger.debug("simulating zernike PNG results from %s to %s" % (SIM_ZERNIKE_PNG, dest))
        shutil.copy(SIM_ZERNIKE_PNG, dest)

def process(state, proj, scanNum, refScan, refScanNum, refScanFile, filename, test=False):
    """
    This is the processing spawned off by the main server thread.
    state: server state
    proj: as in /home/gbtdata/<proj>
    refScan: boolean
    refScanNum: the scan number of the desired ref scan
    refScanFilename: the path to the desired ref scan.  Needed if this is
    not a ref scan and we want to process a ref/sig pair of scan.
    filename: path to this scans eventual data file
    test: for testing, avoid interacting with lassi_daq
    """
    logger.debug("starting process, with state: %s" % state.value)

    if test:
        # skip all interactions with scanner
        processing(state, {}, proj, scanNum, refScan, refScanNum, refScanFile, filename, test=test)
        state.value = READY
        logger.debug ("done, setting state: %s" % state.value)
        return

    # a = TLSaccess("lassi.ad.nrao.edu")
    a = TLSaccess(TLS_HOST)

    waitForScanEnd(state, a)
    r = waitForData(state, a)

    a.cntrl_exit()
    processing(state, r, proj, scanNum, refScan, refScanNum, refScanFile, filename)

    # done!
    state.value = READY
    logger.debug ("done, setting state: %s" % state.value)


def serve():
    """
    This is the server entry point, where a while loop
    receives messages, parses them, and acts accordingly.
    Actually data processing may be spawned from here as
    a separate Process.
    """

    # initiallize our bookkeeping
    state = Value('i', READY)
    proj = None
    scanNum = None
    refScan = True
    refScanNum = None
    filename = None

    # setup the server REPLY socket
    port = PORT
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)

    # initialize the Process object
    p = None

    # initialize our 'memory' of what scans we
    # have processed
    scans = {}

    # TBF: terminate with a server command as well?
    # run till a keyboard interrupt
    run = True

    while run:
        try:
            #logger.debug ("waiting for message")
            msg = socket.recv()
            #logger.debug ("got msg")
            #logger.debug (msg)
            msgStr = "".join( chr(x) for x in msg)

            # ************ Process Data
            if msg == b"process":
                if state.value == READY:
                    # record the current state of all parameters so
                    # we know what scans are being processed
                    filepath = getFITSFilePath(proj, filename)
                    if proj not in scans:
                        scans[proj] = {}
                    scans[proj][scanNum] = {
                        "scanNum": scanNum,
                        "timestamp": datetime.now(),
                        "refScan": refScan,
                        "refScanNum": refScanNum,
                        "filename": filename,
                        "filepath": filepath,
                        "filepathSmoothed": filepath.replace(".fits", ".smoothed.fits")
                    }

                    logger.debug("scans list: %s" % scans)
                    if not refScan:
                        # this is a signal scan, so get the filename of our 
                        # reference scan
                        refScanFile = getRefScanFileName(scans, proj, refScanNum)
                    else:
                        refScanFile = None
                        
                    logger.debug("processing!")
                    p = Process(target=process, 
                                args=(state,
                                      proj, 
                                      scanNum, 
                                      refScan,
                                      refScanNum, 
                                      refScanFile, 
                                      filename))
                    p.start()
                    #state.value = PROCESSING
                    # socket.send_string("Started Processing")
                    logger.debug("Started Processing")
                    socket.send_string("OK")
                else:
                    logger.debug("processing already!")
                    socket.send_string("Already processing")

            # ******** STOP processing data        
            elif msg == b"stop":
                if state.value not in PROCESS_STATES:
                    socket.send_string("Nothing to stop")
                else:
                    if p is not None:
                        logger.debug ("terminating process")
                        logger.debug (stateMap[state.value])
                        p.terminate()
                        state.value = READY
                        logger.debug ("process terminated")
                        socket.send_string("OK")
                    else:
                        logger.debug ("can't terminate, p is none")
                        socket.send_string("can't terminate, p is none")

            # ************ return our current STATE            
            elif msg == b"get_state": # or msg == "get_state":
                #socket.send_string("READY" if state.value is 0 else "PROCESSING")    
                socket.send_string(stateMap[state.value])

            # ************* SET a parameter    
            elif msgStr[:4] == "set:":
                # set what to what?
                # set: key=value
                ps = msgStr[4:].split("=")
                if len(ps) != 2:
                    socket.send_string("Cant understand: %s" % msgStr)
                else:
                    key = ps[0]
                    value = ps[1]
                    if key == 'proj':
                        proj = value
                    elif key == 'scanNum':
                        scanNum = int(value)
                    elif key == 'refScan':
                        # TBF: settle on bool or int type?
                        # refScan = value == 'True' 
                        refScan = int(value) == 1 
                    elif key == 'refScanNum':
                        refScanNum = int(value)
                    elif key == 'filename':
                        filename = value
                    else:
                        logger.debug("unknonw key: %s" % key)                    
                    # socket.send_string("setting %s to %s" % (key, value))    
                    logger.debug("setting %s to %s" % (key, value))
                    socket.send_string("OK")

            # ********* RAISE an ERROR!            
            else:
                logger.debug("what?")
                logger.debug(msg)
                socket.send_string("Dont' understand message")

            # no need for the server to spin too fast
            time.sleep(1)

        except KeyboardInterrupt:

            # exit gracefully
            logger.debug("KeyboardInterrupt")
            if p is not None:
                p.terminate()
            # socket.send_string("server exiting")
            run = False
        
    logger.debug("Exiting server")   

def getGpuOutputPaths(paths, outfile):
    "Where will the output go?"
    # Ex: [/home/sandboxes/pmargani/gpus/gpu1/outfile.1]
    return ["%s.%d" % (os.path.join(path,outfile), i+1) for i, path in enumerate(paths)]


def main():
    serve()

if __name__ == '__main__':
    main()
