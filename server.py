
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

# states
READY = 0 #"READY"
WAITING_FOR_SCAN_END = 1
WAITING_FOR_DATA = 2
PROCESSING = 3 #"PROCESSING"

PROCESS_STATES = [WAITING_FOR_SCAN_END, WAITING_FOR_DATA, PROCESSING]

stateMap = {
        READY: "Ready",
        WAITING_FOR_SCAN_END: "Waiting for scan end.",
        WAITING_FOR_DATA: "Waiting for data.",
        PROCESSING: "Processing"
        }

state = Value('i', READY)

# connect to the scanner
# a = TLSaccess("lassi.ad.nrao.edu")

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
    a.export_data()
    keys = a.get_results().keys()
    while len(keys) < 5:
        print("not enough keys: ", keys)
        time.sleep(1)
        keys = a.get_results().keys()
    print ("We have all our keys now: ", keys)
    return a.get_results()
    # return None



def processing(state, results):
    state.value = PROCESSING
    print(stateMap[PROCESSING])

    # here we can finally process the data
    time.sleep(3)

    # get x, y, z and intesity from results
    x = results['X_ARRAY']
    y = results['Y_ARRAY']
    z = results['Z_ARRAY']
    i = results['I_ARRAY']
    dts = results['TIME_ARRAY']
    hdr = results['HEADER'].asdict()

    # then being processing
    lines = None
    xyzi = (x, y, z, i)

    # xyz, dts = processNewPTXData(lines,
    #                   xyzi=xyzi,
    #                   dts=dts,
    #                   plotTest=False)

    s = settings.SETTINGS_27MARCH2019

    test = True
    if test:
        # read data from previous scans
        path = s['dataPath']
        fn = settings.SCAN9

        fpath = os.path.join(path, fn)
        with open(fpath, 'r') as f:
            ls = f.readlines()

        # finally, substitue the data!
        x, y, z, i = getRawXYZ(ls)

        # fake the datetimes
        dts = np.zeros(len(x))
    
    # TBF: in production this might come from config file?        
    ellipse, rot = settings.getData(s)

    # TBF: we'll get these from the manager
    proj = "TEST"
    dataDir = "/home/sandboxes/pmargani/LASSI/data"
    filename = "test"

    processLeicaDataStream(x,
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

    # any more processing?

    # right processed results to FITS file?

    # if this is a ref scan we are done

    # if it is a signal scan, we need to compute Zernike's

def process(state):
    print("starting process, with state: ", state.value)
 
    a = TLSaccess("lassi.ad.nrao.edu")

    waitForScanEnd(state, a)
    r = waitForData(state, a)

    a.cntrl_exit()
    processing(state, r)

    # done!
    state.value = READY
    print ("done, setting state: ", state.value)

port = "5557"
#port = "9020"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:%s" % port)

socket.send_string("Server ready for commands")
p = None

while True:
    msg = socket.recv()
    print (msg)
    msgStr = "".join( chr(x) for x in msg)
    if msg == b"process":
        if state.value == READY:
            print("processing!")
            p = Process(target=process, args=(state, ))
            p.start()
            #state.value = PROCESSING
            socket.send_string("Started Processing")
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
                socket.send_string("Stopped process")
                state.value = READY
                print ("process terminated")
            else:
                print ("can't terminate, p is none")
    elif msg == b"get_state":
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
            socket.send_string("setting %s to %s" % (key, value))    
    else:
        socket.send_string("Dont' understand message")

    time.sleep(1)

