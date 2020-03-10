import shutil
import zmq
import random
import sys
import os
import time
from datetime import datetime
from multiprocessing import Process, Value

import numpy  as np
import msgpack
import msgpack_numpy

import lassiTestSettings as settings
from ops.pyTLS import TLSaccess
from processPTX import getRawXYZ, processNewPTXData
# from lassiAnalysis import processLeicaDataStream
from lassiAnalysis import extractZernikesLeicaScanPair
from ZernikeFITS import ZernikeFITS
from ops.getConfigValue import getConfigValue
from plotting import plotZernikes
from SmoothedFITS import SmoothedFITS
from utils.utils import splitXYZ
from gpus import loadParallelGPUFiles
from lassiAnalysis import imageSmoothedData

from settings import GPU_MULTI_HOSTS, GPU_MULTI_PATHS

# Get a number of settings from the config files:
# DATADIR = "/home/sandboxes/pmargani/LASSI/data"
# from system.conf:
DATADIR = getConfigValue(".", "YGOR_DATA")
print("Writing FITS files to YGOR_DATA: ", DATADIR)

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
print("Starting analysis server using sim results: ", SIM_RESULTS)
print("Starting analysis server using sim  inputs: ", SIM_INPUTS)
print("For smoothing using these hosts: ", GPU_MULTI_HOSTS)
print("from these installations: ", GPU_MULTI_PATHS)

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
    print("Publishing from: ", pubUrl)

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
    print("publishing primer")
    pubSocket.send_multipart([b"PRIMER"])
    time.sleep(3)

    print ("publishing data")
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
    print("published!")

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
                           plot=True):
    """
    x, y, z: data streamed from scanner
    i: intensity values
    dts: datetime values
    hdr: header dictionary from scanner stream
    ellipse: used for filtering data (dependent on scanner position)
    dataDir: where data files get written to (eg, /home/gbtdata)
    project: same sense as GBT project (eg, TGBT19A_500_01)
    filename: basename of file products (eg, 2019_09_26_01:35:43)
    """

    # do basic processing first: remove bad data, etc.
    # TBF: backwards compatible (bad) interface
    lines = None
    xyzi = (x, y, z, i)
    xyz, dts = processNewPTXData(lines,
                                 xyzi=xyzi,
                                 rot=rot,
                                 ellipse=ellipse,
                                 rFilter=True,
                                 iFilter=False)

    print("done pre-processing data, ready to smooth")

    # fpathBase = os.path.join(dataDir, project, "LASSI", filename)
    # processedPath = "%s.csv" % fpathBase
    # print ("writing filtered data to CSV file: ", processedPath)
    # np.savetxt(processedPath, xyz, delimiter=",")

    # Next: smooth
    # where will the smoothing output go?
    # TBF: don't hard code the file output path!!!
    outFile = "outfile"
    outPath = "/users/pmargani/tmp"
    fn = os.path.join(outPath, outFile)
    fn = "%s.x.csv" % fn
    # make sure that output isn't there yet
    if os.path.isfile(fn):
        os.remove(fn)
    
    # now publish the input to smoothing
    print("splitting data")
    x, y, z = splitXYZ(xyz)
    publishData(x, y, z)

    # We can lower this for testing purposes
    N = 512

    #x, y, z = smoothGPUParallel(processedPath, N)

    # and wait for it to show up.
    # TBF: how to handle mutliple parallel GPUs???
    while not os.path.isfile(fn):
        print("Waiting on smoothing", fn)
        time.sleep(5)

    # read in those files
    x, y, z = loadParallelGPUFiles(outFile, [outPath])

    # save this off for later use
    fitsio = SmoothedFITS()
    fitsio.setData(x, y, z, N, hdr, dataDir, project, filename)
    fitsio.write()

    smoothedFitsFilename = fitsio.getFilePath()

    if plot:
        # how should we save the image of our processed data?
        ext = "smoothed.fits"
        fn = smoothedFitsFilename[:-len(ext)] + "processed.png"
        print("Processing smoothed data, imaging to:", fn)
        # process a little more and create a surface plot for diagnosis
        imageSmoothedData(x, y, z, N, filename=fn)

    return fitsio.getFilePath()

def getFITSFilePath(proj, filename):
    return os.path.join(DATADIR, proj, "LASSI", filename + ".fits")

def getRefScanFileName(scans, proj, refScanNum):

    if proj not in scans:
        print ("Proj not in scans: ", proj, scans.keys())
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
            print("ERROR: cannot find refScan from", proj, refScanNum, scans)
            return None
        else:
            print("getRefScanFileName: ", newestRefScan)
            return newestRefScan['filepathSmoothed']

    # if it has, can we use it?
    if refScanNum not in scans[proj]:
        print ("Scan not in proj: ", refScanNum, scans[proj].keys())
        return None            

    scan = scans[proj][refScanNum]
    # assert scans[proj][scan]["refScan"]
    # print(scans[proj][scan]["refScan"])
    print(scan)

    return scan['filepathSmoothed']

def waitForScanEnd(state, a):
    print ("in WaitForScanEnd")
    state.value = WAITING_FOR_SCAN_END
    print(stateMap[WAITING_FOR_SCAN_END])

    print ("can we get the scanner state?")
    print (a.get_status())
    print ("yes we can")

    # here we wait until the scanner's state has gotten to Ready
    # time.sleep(3)
    state = "unknown"
    while state != "Ready":
        status = a.get_status()
        state = status.state
        print("State: ", state)
        time.sleep(1)
    print ("done waiting for scan end")
    

def waitForData(state, a):
    state.value = WAITING_FOR_DATA
    print(stateMap[WAITING_FOR_DATA])

    # here we call export to data, then wait until we actually see all of it
    # time.sleep(3)
    #a.export_data()
    keys = a.get_results().keys()
    while len(keys) < 5:
        print("not enough keys: ", keys)
        time.sleep(1)
        keys = a.get_results().keys()
    print ("We have all our keys now: ", keys)
    return a.get_results()
    # return None

def processing(state, results, proj, scanNum, refScan, refScanNum, refScanFile, filename):
    state.value = PROCESSING
    print(stateMap[PROCESSING])

    # here we can finally process the data
    time.sleep(3)

    # test = True

    # if test:
    #     results = {
    #         'X_ARRAY': None,
    #         'Y_ARRAY': None,
    #         'Z_ARRAY': None,
    #         'TIME_ARRAY': None,
    #         'I_ARRAY': None,
    #         'HEADER': {},
    #     }

    # get x, y, z and intesity from results
    if not SIM_INPUTS:
        x = results['X_ARRAY']
        y = results['Y_ARRAY']
        z = results['Z_ARRAY']
        i = results['I_ARRAY']
        dts = results['TIME_ARRAY']
        hdrObj = results['HEADER']

        # hdr = hdrObj.asdict() if not test else {}
        hdr = hdrObj.asdict()
    else:
        simFile = SIM_REF_INPUT if bool(refScan) else SIM_SIG_INPUT
        print("Using simulated input: ", simFile)
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

    # xyz, dts = processNewPTXData(lines,
    #                   xyzi=xyzi,
    #                   dts=dts,
    #                   plotTest=False)

    # TBF: pass this in?  From settings file?
    s = settings.SETTINGS_27MARCH2019
    # s = settings.SETTINGS_19FEB2020
    # s = settings.SETTINGS_11OCTOBER2019

    # if test:
    #     # read data from previous scans
    #     path = s['dataPath']

    #     # determine which scan to read depending on
    #     # whether this is a reference or signal scan
    #     fn = settings.SCAN9 if refScan else settings.SCAN11

    #     fpath = os.path.join(path, fn)
    #     with open(fpath, 'r') as f:
    #         ls = f.readlines()

    #     # finally, substitue the data!
    #     x, y, z, i = getRawXYZ(ls)

    #     # fake the datetimes
    #     dts = np.zeros(len(x))
    
    # TBF: in production this might come from config file?        
    ellipse, rot = settings.getData(s)

    # TBF: we'll get these from the manager
    # proj = "TEST"
    # dataDir = "/home/sandboxes/pmargani/LASSI/data"
    dataDir = DATADIR
    # filename = "test"

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
                           filename)
    else:
        # cp the smoothed file to the right locatoin
        dest = os.path.join(dataDir, proj, 'LASSI', filename)
        dest = dest + ".smoothed.fits"
        # which file?
        simFile = SIM_REF_SMOOTH_RESULT if refScan else SIM_SIG_SMOOTH_RESULT
        print("simulating smoothed results from %s to %s" % (simFile, dest))
        shutil.copy(simFile, dest)
        fitsFile = dest

    # any more processing?
    if refScan:
        # if this is a ref scan we are done
        print("This is a ref scan, we are done")
        return


    # if it is a signal scan, we
    # need to compute Zernike's
    # find the previous refscan:
    # print("look for file for scan", refScanNum)
    sigScanFile = filename
    #N = 100
    N = 512


    print("files: ", refScanFile, sigScanFile, fitsFile)

    if not SIM_RESULTS:
        #testSig = '/home/sandboxes/pmargani/LASSI/gpus/versions/devenv-hpc1/1.csv'
        #testRef = '/home/sandboxes/pmargani/LASSI/gpus/versions/devenv-hpc1/2.csv'
        # extractZernikesLeicaScanPair(refScanFile, sigScanFile, n=512, nZern=36, pFitGuess=[60., 0., 0., -50., 0., 0.], rMaskRadius=49.)
        # extractZernikesLeicaScanPair(refScanFile, sigScanFile, n=N, nZern=36)
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
        print(zernikes)
        print(type(zernikes))
        # zernikes = zernikes * 1e6
        zernikes = [z * 1e6 for z in zernikes]
        fitsio.setData(xs, ys, zs, N, hdr, dataDir, proj, filename)
        fitsio.setZernikes(zernikes)
        print ("Writing Zernikes to: ", fitsio.getFilePath())
        fitsio.write()

        # now make sure there's a plot for this fits file too
        zDict = {}
        for i in range(len(zernikes)):
            k = "Z%d" % (i + 1)
            zDict[k] = zernikes[i]

        title = "%s:%s" % (proj, scanNum)
        p = plotZernikes(zDict, title=title)
        
        # change extension from .fits to .png
        fn = fitsio.getFilePath()[:-4] + "png"
        print ("Writing Zernikes png to: ", fn)
        p.savefig(fn)    

    else:
        # cp the files to the right locatoin
        # first the zernike FITS file
        dest = os.path.join(dataDir, proj, 'LASSI', filename)
        dest = dest + ".zernike.fits"
        print("simulating zernike FITS results from %s to %s" % (SIM_ZERNIKE_RESULT, dest))
        shutil.copy(SIM_ZERNIKE_RESULT, dest)

        # then copy over the image file
        dest = dest[:-4] + "png"
        print("simulating zernike PNG results from %s to %s" % (SIM_ZERNIKE_PNG, dest))
        shutil.copy(SIM_ZERNIKE_PNG, dest)

def process(state, proj, scanNum, refScan, refScanNum, refScanFile, filename):
    print("starting process, with state: ", state.value)

    # test = True
    test = False
    if test:
        # skip all interactions with scanner
        processing(state, {}, proj, scanNum, refScan, refScanNum, refScanFile, filename)
        state.value = READY
        print ("done, setting state: ", state.value)
        return

    # a = TLSaccess("lassi.ad.nrao.edu")
    a = TLSaccess(TLS_HOST)

    waitForScanEnd(state, a)
    r = waitForData(state, a)

    a.cntrl_exit()
    processing(state, r, proj, scanNum, refScan, refScanNum, refScanFile, filename)

    # done!
    state.value = READY
    print ("done, setting state: ", state.value)


def serve():

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

    while True:
        #print ("waiting for message")
        msg = socket.recv()
        #print ("got msg")
        #print (msg)
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

                print("scans list:", scans)
                if not refScan:
                    # this is a signal scan, so get the filename of our 
                    # reference scan
                    refScanFile = getRefScanFileName(scans, proj, refScanNum)
                else:
                    refScanFile = None
                    
                print("processing!")
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
                print("Started Processing")
                socket.send_string("OK")
            else:
                print("processing already!")
                socket.send_string("Already processing")

        # ******** STOP processing data        
        elif msg == b"stop":
            if state.value not in PROCESS_STATES:
                socket.send_string("Nothing to stop")
            else:
                if p is not None:
                    print ("terminating process")
                    print (stateMap[state.value])
                    p.terminate()
                    state.value = READY
                    print ("process terminated")
                    socket.send_string("OK")
                else:
                    print ("can't terminate, p is none")
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
                    print("unknonw key", key)                    
                # socket.send_string("setting %s to %s" % (key, value))    
                print("setting %s to %s" % (key, value))
                socket.send_string("OK")

        # ********* RAISE an ERROR!            
        else:
            print("what?")
            print(msg)
            socket.send_string("Dont' understand message")

        time.sleep(1)

def tryPublishFromThread():

    p = Process(target=tryPublishPreSmoothedData)

    p.start()

    while True:
        time.sleep(1)

def tryPublishPreSmoothedData():

    # # pubHost = "galileo.gb.nrao.edu"
    # # pubPort = 35564
    # pubPort = 9001
    # # pubUrl = "tcp://%s:%d" % (pubHost, pubPort)
    # pubUrl = "tcp://*:%d" % (pubPort)
    # print("Publishing from: ", pubUrl)

    # ctx = zmq.Context()
    # # pubck = zmq.Socket(ctx, zmq.PUB)
    # pubck = ctx.socket(zmq.PUB)
    # # pubck.connect(pubUrl) 
    # pubck.bind(pubUrl)
    
    # while True:
    #     print("publish!")
    #     # pubck.send_string("topic data")
    #     pubck.send_multipart([b"HEADER", b"DATA"])
    #     time.sleep(10)  


    x = np.array([float(0.) for i in range(4019821)])
    y = np.array([float(0.) for i in range(4019821)])
    z = np.array([float(0.) for i in range(4019821)])

    print("len of x data", len(x))
    
    # # x = msgpack.packb(data)
    # x = msgpack.packb(xdata, default=msgpack_numpy.encode)
    # y = msgpack.packb(ydata, default=msgpack_numpy.encode)
    # z = msgpack.packb(zdata, default=msgpack_numpy.encode)

    publishData(x, y, z)

def main():
    serve()
    # tryPublishFromThread()
    # tryPublishPreSmoothedData()

if __name__ == '__main__':
    main()
