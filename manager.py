import zmq
import random
import sys
import time

from ops.pyTLS import TLSaccess

def runScan():

    a = TLSaccess("lassi.ad.nrao.edu")

    # wait till it's ready
    status = a.get_status()
    print ("scanner status:", status)
    while status.state != "Ready":
        time.sleep(1)
        status = a.get_status()

    # configure
    proj = "test"
    res = "63mm@100m"
    sensitivity = "Normal"
    scan_mode = "Speed"
    cntr_az = 270
    cntr_el = 45
    az_fov = 15
    el_fov = 15
    a.configure_scanner(proj, res, sensitivity, scan_mode, cntr_az, cntr_el, az_fov, el_fov)

    # get the device scan started,
    a.start_scan()

    # now wait until it's NOT in ready
    while status.state == "Ready":
        print("waiting for scan to start")
        time.sleep(1)
        status = a.get_status()    

    # dissconnect! Because it only supports one control connection
    a.cntrl_exit()
    del a

    # then once it's started, get the processing going:
    # connect to the process server
    port = "5557"
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://localhost:%s" % port)

    # # see what state the server is in
    socket.send_string("get_state")
    msg = socket.recv()
    print ("process server state: ", msg)

    socket.send_string("process")
    msg = socket.recv()
    print ("process server response: ", msg)

    while True:
        try:
            # status = a.get_status()
            # print ("scanner state: ", status.state)
            # see what state the server is in
            socket.send_string("get_state")
            msg = socket.recv()
            print ("process server state: ", msg)
            time.sleep(1)
        except KeyboardInterrupt:
            # a.cntrl_exit()
            break

def main():
    runScan()

if __name__ == '__main__':
    main()
