import shutil
import zmq
import random
import sys
import os
import time
from multiprocessing import Process, Value

import numpy  as np

import lassiTestSettings as settings
from ops.pyTLS import TLSaccess
from processPTX import getRawXYZ
from lassiAnalysis import processLeicaDataStream
from lassiAnalysis import extractZernikesLeicaScanPair
from ZernikeFITS import ZernikeFITS
from ops.getConfigValue import getConfigValue

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
SIM_REF_SMOOTH_RESULT = getConfigValue(".", "analysisSmoothRefResult", configFile=lc)
SIM_SIG_SMOOTH_RESULT = getConfigValue(".", "analysisSmoothSigResult", configFile=lc)
SIM_ZERNIKE_RESULT = getConfigValue(".", "analysisZernikeResult", configFile=lc)
PORT = int(getConfigValue(".", "analysisServerPort", configFile=lc))
print("Starting analysis server using sim results: ", SIM_RESULTS)

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

state = Value('i', READY)

proj = None
scanNum = None
refScan = True
refScanNum = None
filename = None

# connect to the scanner
# a = TLSaccess("lassi.ad.nrao.edu")
# TLS_HOST = "lassi.ad.nrao.edu"

def getFITSFilePath(proj, filename):
    return os.path.join(DATADIR, proj, "LASSI", filename + ".fits")

def getRefScanFileName(scans, proj, refScanNum):

    if proj not in scans:
        print ("Proj not in scans: ", proj, scans.keys())
        return None

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
    x = results['X_ARRAY']
    y = results['Y_ARRAY']
    z = results['Z_ARRAY']
    i = results['I_ARRAY']
    dts = results['TIME_ARRAY']
    hdrObj = results['HEADER']

    # hdr = hdrObj.asdict() if not test else {}
    hdr = hdrObj.asdict()

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

    s = settings.SETTINGS_27MARCH2019

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
        # cp the file to the right locatoin
        dest = os.path.join(dataDir, proj, 'LASSI', filename)
        dest = dest + ".smoothed.fits"
        # which file?
        simFile = SIM_REF_SMOOTH_RESULT if refScan else SIM_SIG_SMOOTH_RESULT
        print("simulating smoothed results from %s to %s" % (simFile, dest))
        shutil.copy(simFile, dest)

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


    if not SIM_RESULTS:
        print("files: ", refScanFile, sigScanFile, fitsFile)
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
        fitsio.setData(xs, ys, zs, N, hdr, dataDir, proj, filename)
        fitsio.setZernikes(zernikes)
        print ("Writing Zernikes to: ", fitsio.getFilePath())
        fitsio.write()
    else:
        # cp the file to the right locatoin
        dest = os.path.join(dataDir, proj, 'LASSI', filename)
        dest = dest + ".zernike.fits"
        print("simulating zernike results from %s to %s" % (SIM_ZERNIKE_RESULT, dest))
        shutil.copy(SIM_ZERNIKE_RESULT, dest)

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

port = PORT
# port = "5557"
#port = "9020"
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)

# socket.send_string("Server ready for commands")
p = None

scans = {}



while True:
    #print ("waiting for message")
    msg = socket.recv()
    #print ("got msg")
    #print (msg)
    msgStr = "".join( chr(x) for x in msg)
    if msg == b"process":
        if state.value == READY:
            # record the current state of all parameters so
            # we know what scans are being processed
            filepath = getFITSFilePath(proj, filename)
            if proj not in scans:
                scans[proj] = {}
            scans[proj][scanNum] = {
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
    elif msg == b"get_state": # or msg == "get_state":
        #socket.send_string("READY" if state.value is 0 else "PROCESSING")    
        socket.send_string(stateMap[state.value])
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
    else:
        print("what?")
        print(msg)
        socket.send_string("Dont' understand message")

    time.sleep(1)

