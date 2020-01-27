from __future__ import print_function
import zmq
import msgpack
import threading
import string
import random
from time import sleep
import numpy as np
import msgpack_numpy
import inspect
from datetime import datetime
import sys

CONTROL_PORT=':35560'
DATAPUB_PORT=':35562'

SCAN_MODES = {"Speed" : 0, "Range" : 1, "Medium Range" : 2, "Long Range" : 3}
SCANSENSITIVITY = {"Normal" : 0, "High" : 1}
SCANRESOLUTION = {"500mm@100m" : 0, "250mm@100m" : 1, "125mm@100m" : 2, \
                  "63mm@100m" : 3, "31mm@100m" : 4, "16mm@100m" : 5, "8mm@100m" : 6}
RESOLUTIONTOTEXT = { 0: "500mm@100m", 1: "250mm@100m", 2:"125mm@100m", \
                     3: "63mm@100m", 4: "31mm@100m", 5:"16mm@100m",
                     6: "8mm@100m" }
SENSITIVITYTOTEXT = {0: "Normal", 1: "High" }
SCANMODETOTEXT = {0: "Speed", 1: "Range", 2: "Medium Range", 3: "Long Range" }


def mycb(a, b):
    if a in "OK_STATUS":
        print("\nScanner State=", b.state)
    else:
        print("\nReceived scan data for scan ")

class OKStatus:
    def __init__(self, g):
        self.error_msg = g[0].decode()
        self.is_ok = g[1]
        self.scan_number = g[2]
        self.state = g[3].decode()
        self.state_id = g[4]

    def __str__(self):
        return "is_ok=%s, state=%s, message=%s scan_number=%d" % (str(self.is_ok), self.state, self.error_msg,self.scan_number)

class ScanHeaderInfo:
    def __init__(self, g):
        # print("g=", g)
        if type(g) == bool:
            # Use default/dummy values
            self.center_az = 180.0
            self.center_el = 45.0
            self.fov_az = 180.0
            self.fov_el =  90.0
            self.scan_number = 1
            ns_since_1970 = int(0)
            self.scan_time = datetime.fromtimestamp(ns_since_1970/1.0E9) # DateTime from long
            self.tls_project = "None"
            self.tls_resolution = 1 # convert to string
            self.tls_scan_mode = 0  # convert to string
            self.tls_serial_number = 18880
            self.tls_sensitivity = 0 # convert to string
        else:
            self.center_az = float(g[0])
            self.center_el = float(g[1])
            self.fov_az = float(g[2])
            self.fov_el = float(g[3])
            self.scan_number = int(g[4])
            ns_since_1970 = int(g[5])
            self.scan_time = datetime.fromtimestamp(ns_since_1970/1.0E9) # DateTime from long
            self.tls_project = g[6].decode()
            resolution = int(g[7])
            if resolution in RESOLUTIONTOTEXT.keys():
                self.tls_resolution = RESOLUTIONTOTEXT[resolution]
            mode = int(g[8])
            if mode in SCANMODETOTEXT.keys():
                self.tls_scan_mode = SCANMODETOTEXT[mode]

            self.tls_serial_number = int(g[9])

            sensitivity = int(g[10]) # convert to string
            if sensitivity in SENSITIVITYTOTEXT.keys():
                self.tls_sensitivity = SENSITIVITYTOTEXT[sensitivity]
            self.tls_tilt_compensator = int(g[11])

    def asdict(self):
        d = {
            "center_az" : self.center_az,
            "center_el" : self.center_el,
            "fov_az" : self.fov_az,
            "fov_el" : self.fov_el,
            "scan_number": self.scan_number,
            "scan_time": self.scan_time,
            "project": self.tls_project,
            "resolution" : self.tls_resolution,
            "scan_mode" : self.tls_scan_mode,
            "tls_serial_number" : self.tls_serial_number,
            "sensitivity" : self.tls_sensitivity,
            "tls_tilt_compensator" : self.tls_tilt_compensator
        }
        return d


    def __str__(self):
        """
        string representation of a ScanHeaderInfo object
        :return: 
        """
        s= """
        center_az = %f
        center_el = %f
        fov_az = %f
        fov_el = %f
        scan_number %d
        scan_time= %s
        project= %s
        resolution = %s
        scan_mode = %s
        tls_serial_number = %d
        sensitivity = %s
        tls_tilt_compensator = %d
        """ % (self.center_az, self.center_el, self.fov_az, self.fov_el, self.scan_number, \
               self.scan_time, self.tls_project, self.tls_resolution, self.tls_scan_mode, \
               self.tls_serial_number, self.tls_sensitivity, self.tls_tilt_compensator)
        return s


class TLSaccess(object):
    def __init__(self, scanner_host):
        """
        Creates a TLS scanner interface object
        scanner_host: a string specifying the host name of the scanner controller.
        e.g: lassi.ad.nrao.edu (i.e NOT a ZMQ URL)
       
        """
        self.SUBSCRIBE = 10
        self.UNSUBSCRIBE = 11
        self.UNSUBSCRIBE_ALL = 12
        self.QUIT = 13
        self.PING = 14

        self.array_names = ["X_ARRAY", "Y_ARRAY", "Z_ARRAY",
                            "AZ_ARRAY", "EL_ARRAY", "R_ARRAY", "I_ARRAY",
                            "RED_ARRAY", "GREEN_ARRAY", "BLUE_ARRAY", "TIME_ARRAY"]

        url='tcp://' + scanner_host + CONTROL_PORT
        suburl = 'tcp://' + scanner_host + DATAPUB_PORT

        self.scanner_host = scanner_host

        self._ctx = zmq.Context()
        self.csock = zmq.Socket(self._ctx, zmq.REQ)
        self.csock.connect(url)

        self.subck = zmq.Socket(self._ctx, zmq.SUB)
        self.subck.connect(suburl)

        self._sub_task = None
        self.results = {}

        if not self._sub_task:
            self._sub_task = TLSaccess.PipelineSubscriberThread(self)
            self._sub_task.start()
            sleep(1)

        self.hdr_info = ScanHeaderInfo(False)
        self.add_callback("debug_callback", mycb)

    def __del__(self):
        "An attempt to shutdown the subscriber thread, which doesnt seem to work"
        self.cntrl_exit()

    def help(self):
        "A help page to guide you through"

        print("Method documentation:")
        help="""configure_scanner(project, resolution, sensitivity, scan_mode, center_az, center_el, fov_az, fov_el) """
        print(help)
        print(self.configure_scanner.__doc__)
        help="""move_az(az) """
        print(help)
        print(self.move_az.__doc__)

        print("""start_scan()""")
        print(self.start_scan.__doc__)

        print("""stop_scan()""")
        print(self.stop_scan.__doc__)

        print("""pause_scan()""")
        print(self.pause_scan.__doc__)

        print("""resume_scan()""")
        print(self.resume_scan.__doc__)

        print("""get_status()""")
        print(self.get_status.__doc__)

        print("""set_scan_number""")
        print(self.set_scan_number.__doc__)

        print("""set_simulated_data_file(ptxdatafile""")
        print(self.set_simulated_scan_data_file.__doc__)

        print("""add_callback(callback_name, cb_fun)""")
        print(self.add_callback.__doc__)

        print("""remove_callback(callback_name)""")
        print(self.remove_callback.__doc__)

        print("""get_result(which_array)""")
        print(self.get_result.__doc__)

        print("""get_results()""")
        print(self.get_results.__doc__)

        print("""shutdown_scanner()""")
        print(self.shutdown_scanner.__doc__)

        print("""export_data()""")
        print(self.export_data.__doc__)

        print("""cntrl_exit()""")
        print(self.cntrl_exit.__doc__)

        print("""export_ptx_results(filename)""")
        print(self.export_ptx_results.__doc__)

        print("""export_csv_results(filename)""")
        print(self.export_csv_results.__doc__)

        print("""list_projects()""")
        print(self.list_projects.__doc__)

        print("""list_scan_names(project_name)""")
        print(self.list_scan_names.__doc__)

        print("""export_scan_from_project(project, scan)""")
        print(self.export_scan_from_project.__doc__)

    def parse_OKStatus(self, msg):
        if msg is None:
            return None

        kk = msgpack.unpackb(msg[1])
        status = OKStatus(kk)
        return status

    def parse_listScans(self, msg):
        bin_scanlist = msgpack.unpackb(msg[1])
        scanlist = []
        for k in bin_scanlist:
            scanlist.append(k.decode())
        return scanlist

    def encode_MoveAz(self, azcmd):
        return msgpack.packb([azcmd], use_bin_type=True)

    def encode_ScanNumber(self, scannum):
        return msgpack.packb([scannum], use_bin_type=True)

    def encode_Config(self, project, res, sensitivity, mode, ctr_az, ctr_el, fov_az, fov_el):
        #Note: this order must match the ordering in lassi_daq
        return msgpack.packb([ctr_az, ctr_el, fov_az, fov_el, project, res, mode, sensitivity])

    def simple_cmd(self, cmd, args=None):
        if args is None:
            self.csock.send_string(cmd)
        else:
            self.csock.send_string(cmd, zmq.SNDMORE)
            for i in range(0,  len(args)):
                if i < len(args)-1:
                    self.csock.send_string(args[i], zmq.SNDMORE)
                else:
                    self.csock.send_string(args[i])

        rtn = self.csock.recv_multipart()
        return self.parse_OKStatus(rtn)


    # API methods below
    def configure_scanner(self, project, resolution, sensitivity, scan_mode, center_az, center_el, fov_az, fov_el):
        """
        Sends a configuration to the scanner. Fields are:
            project : a valid string for tl_scanner project (not GBT project)
            resolution : specifies the resolution as one of the strings:
                  500mm@100m, 250mm@100m, 125mm@100m, 63mm@100m, 31mm@100m, 16mm@100m, 8mm@100m
            sensitivity: specifies the sensitivity as one of the strings: Normal, High
            scan_mode: specifies Speed vs. Range or one of the strings:
                  Speed, Range, Medium Range, Long Range (Note not all are valid Speed, Range are known to work)
            center_az: center of scan in degrees
            center_el: center of scan in degrees
            fov_az: width of scan in degrees
            fov_el: height of scan in degrees
        returns an OKStatus object
        """

        if type(project) != str:
            print("project must be a string")
            return False
        if type(resolution) == str:
            if resolution in SCANRESOLUTION.keys():
                _resolution = SCANRESOLUTION[resolution]
            else:
                print("unknown scan resolution. Must be one of:")
                print(SCANRESOLUTION.keys())
                return False
        else:
            _resolution = resolution

        if type(sensitivity) == str:
            if sensitivity in SCANSENSITIVITY.keys():
                _sensitivity = SCANSENSITIVITY[sensitivity]
            else:
                print("unknown scan sensitivity. Must be one of:")
                print(SCANSENSITIVITY.keys())
                return False
        else:
            _sensitivity = sensitivity

        if type(scan_mode) == str:
            if scan_mode in SCAN_MODES.keys():
                _scan_mode = SCAN_MODES[scan_mode]
            else:
                print("unknown scan mode. Must be one of:")
                print(SCAN_MODES.keys())
                return False
        else:
            _scan_mode = scan_mode


        msg = self.encode_Config(project, _resolution, _sensitivity, _scan_mode, \
                                 center_az, center_el, fov_az, fov_el)
        self.csock.send_string('Configure', zmq.SNDMORE)
        self.csock.send(msg)

        rtn = self.csock.recv_multipart()
        return self.parse_OKStatus(rtn)

    def move_az(self, az):
        """
        Sends a command to move the scanner base to the specified (instrument relative) azimuth.

        returns an OKStatus object
        """
        msg = self.encode_MoveAz(az)
        self.csock.send_string('MoveAz', zmq.SNDMORE)
        self.csock.send(msg)

        rtn = self.csock.recv_multipart()
        return self.parse_OKStatus(rtn)

    def set_scan_number(self, scannum):
        """
        Sets the user defined scan number. The scan number is incremented every time a start_scan()
        is issued. The scan number is now a field in the OK_State feedback.
        e.g print(a.get_result("OK_STATUS").scan_number) would print the scan number and also
            print(a.get_status().scan_number) also works.
        """
        msg = self.encode_ScanNumber(scannum)
        self.csock.send_string('SetScanNumber', zmq.SNDMORE)
        self.csock.send(msg)

        rtn = self.csock.recv_multipart()
        return self.parse_OKStatus(rtn)

    def start_scan(self):
        """
        Starts a TLS scan
        :return: returns an OKStatus object
        """
        return self.simple_cmd("StartScan")

    def stop_scan(self):
        """
        Stops/aborts a TLS scan
        :return: returns an OKStatus object
        """
        return self.simple_cmd("StopScan")

    def pause_scan(self):
        """
        Pauses an active scan.
        :return: returns an OKStatus object
        """
        return self.simple_cmd("PauseScan")

    def resume_scan(self):
        """
        Resumes a previously paused scan
        :return: returns an OKStatus object
        """
        return self.simple_cmd("ResumeScan")

    def get_status(self):
        """
        Polls the scanner for the most recent status/scan number.
        :return: returns a OKStatus object which contains the state, status, and scan number
        """
        return self.simple_cmd("GetStatus")

    def shutdown_scanner(self):
        """
        This tells the scanner to shutdown cleanly, then turn off.
        Note that once this is issued, the lassi_daq cannot function sanely. Power must
        be removed, then reapplied to get the scanner to boot up.
        """
        return self.simple_cmd("ShutdownScanner")

    def export_data(self):
        """
        Sends the command to command the lassi_daq to publish the most recent scan.
        :return: returns an OKStatus object
        """
        # clear the buffer first!
        self.results.clear()
        return self.simple_cmd("Export")

    def export_scan_from_project(self, project, scan):
        """
        Sends the command to publish a specific scan from a specific project
        :param project: string name of project
        :param scan: string name of scan e.g 'Scan-21'
        :return: returns an OKStatus object
        """
        self.results.clear()
        return self.simple_cmd('ExportScanFromProject', [project, scan])

    def list_scan_names(self, project_name):
        """
        Requests a list of the scans in project named project_name.
        Use list_projects() to query the available projects.
        :param project_name: string, name of project on scanner
        :return: list of string scan names
        """
        self.csock.send_string('ListScansFromProject', zmq.SNDMORE)
        self.csock.send_string(project_name)

        rtn = self.csock.recv_multipart()
        scans = self.parse_listScans(rtn)
        return scans

    def list_projects(self):
        """
        Query the scanner for a list of the stored project names on the TLS.
        :return: list of string project names
        """
        self.csock.send_string('ListProjects')
        rtn = self.csock.recv_multipart()
        projects = self.parse_listScans(rtn)
        return projects

    def set_simulated_scan_data_file(self, simfile):
        """
        Sets the ptx file which the lassi_daq will read when in simulation mode is enabled.
        (i.e. run lassi_daq with the --simulate flag set)
        Note: The file path must be relative to where the lassi_daq is running. For
        example, running lassi_daq on windows and specifying a file in /home/sandboxes
        will not work.
        :param ptx file to use
        :return: returns an OKStatus object
        """
        return self.simple_cmd('SetSimulatedScanFile', [simfile])

    def save_result(self, frame_type, data):
        """
        Saves a published array locally. Can be used (see get_results()) instead of
        registering a callback. Of course its up to the consumer to know when the
        arrays are ready.
        :param frame_type:
        :param data:
        :return:
        """

        self.results[frame_type] = data

    def get_result(self, frame_type):
        """
        Allows access to the result cache. Note: in a future release results may
        only be saved if the callback was None. Should wrap this in a try-except
        because frame_type may not exist.
        """
        try:
            return self.results[frame_type]
        except: 
            return None

    def get_results(self):
        """
        Returns a dictionary of the saved results
        """
        return self.results

    # Other internal methods

    def _kill_subscriber_thread(self):
        """
        Terminates the subscriber thread by sending it the 'QUIT' message,
        closing the control pipe, and joining on the thread.
        """
        if self._sub_task:
            pipe = self._ctx.socket(zmq.REQ)
            pipe.connect(self._sub_task.pipe_url)
            pipe.send_pyobj(self.QUIT)
            pipe.recv_pyobj()
            pipe.close()
            self._sub_task.join()
            del self._sub_task
            self._sub_task = None

    def add_callback(self, callback_key, cb_fun):
        """
        Adds a user defined callback to be called when the lassi_daq publishes
        the scan data. The data contains the time,x,y,z and intensity
        data. This method does not need to be called unless you desire a
        function 'callback'. By default the data will be automatically stored
        and accessible via the 'get_results()' method.
        :param callback_key:
          a string identifying the callback (used as the dictionary key for the callback).

        :param cb_fun:
          the callback function, which must take 1 arg. The callback will receive
          the data as a python dictionary of numarray's for the keys
          X_ARRAY,Y_ARRAY,Z_ARRAY,I_ARRAY and TIME_ARRAY, and a python object
          ScanHeaderInfo for the key 'HEADER', containing auxilary scan information.

           keys are:
                * "X_ARRAY" - x measurement vector in meters as numpy array
                * "Y_ARRAY" - y measurement vector in meters as numpy array
                * "Z_ARRAY" - z measurement vector in meters as numpy array
                * "I_ARRAY" - intensity measurement vector as numpy array
                * "TIME_ARRAY - "MJD's for each pixel, as numpy array
                * "HEADER" - a ScanHeaderInfo python object.

        :return: 'True' if the subscription was successful, 'False' otherwise.
        """

        if cb_fun is not None:
            try:
                x = inspect.getargspec(cb_fun)

                # Should be a function that takes two parameters.
                if len(x.args) != 2:
                    return (False, 'Callback function must take 2 arguments')
            except TypeError:
                # not a function at all!
                return (False, 'Callback object is not a function!')

        if not self._sub_task:
            self._sub_task = TLSaccess.PipelineSubscriberThread(self)
            self._sub_task.start()
            sleep(1)  # give it time to start

        self._sub_task._callbacks[callback_key] = cb_fun
        return (True, None)

    def remove_callback(self, callback_name):
        """
        Removes a previously registered callback.
        :param callback_name: specifies the key the callback was registered with

        :return: True if callback was removed, False otherwise
        """
        if not self._sub_task:
            return (False, "subtask not running - error or no callbacks previously defined")

        if callback_name in self._sub_task._callbacks.keys():
            self._sub_task._callbacks.pop(callback_name)
        return (True, None)

    def my_callback(key, val):
        "An example default callback"
        print(key)

    def cntrl_exit(self):
        """
        Attempts a clean TLSaccess child thread exit. Unfortunately
        omitting this can cause python to hang when trying to exit.
        """
        if self._sub_task:
            pipe = self._ctx.socket(zmq.REQ)
            pipe.connect(self._sub_task.pipe_url)
            pipe.send_pyobj(self.QUIT)

    def export_ptx_results(self, filename):
        """
        A test method. This will write a ptx-like format file (without the header)
        for testing purposes. Note that the X_ARRAY,Y_ARRAY and Z_ARRAY's must be
        subscribed to in order for this to succeed.
        returns nothing
        """
        ident_mtrx = """0.000000 0.000000 0.000000
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000
1.000000 0.000000 0.000000 0
0.000000 1.000000 0.000000 0
0.000000 0.000000 1.000000 0
0.000000 0.000000 0.000000 1
"""

        keys = self.results.keys()
        if "X_ARRAY" in keys and "Y_ARRAY" in keys and \
           "Z_ARRAY" in keys and "I_ARRAY" in keys:
            fout=open(filename, "w+")
            fout.write("%d/%d\n" % (2450, 5600) ) # I have no idea what this is for
            fout.write("%d\n" % 5028)
            fout.write(ident_mtrx)
            x=self.results['X_ARRAY']
            y=self.results['Y_ARRAY']
            z=self.results['Z_ARRAY']
            I=self.results['I_ARRAY']
            lenx = len(x)
            leny = len(y)
            lenz = len(z)
            leni = len(I)
            if (lenx != leny) or (leny != lenx) or (lenz != lenx) or (leni != lenx):
                print("ERROR: data not correct lengths:")
                print("x=%d, y=%d, z=%d, i=%d" % (lenx, leny, lenz, leni))
                return

            for i in range(len(x)):
                fout.write("%f %f %f %f\n" % (x[i], y[i], z[i], I[i]))
            fout.close()
        else:
            print("Not all data is available. This requires that the interface be subscribed")
            print("to each of X_ARRAY, Y_ARRAY, Z_ARRAY and I_ARRAY")
            print("Try subscribing to the arrays, then run export_data() again (no need to rerun the scan)")
            print("Currently Keys are: ", keys)

    def export_csv_results(self, filename):
        """
        A test method. This will write a csv format file which paraview can import from
        for testing purposes. Note that the X_ARRAY,Y_ARRAY and Z_ARRAY's must be
        subscribed to in order for this to succeed.
        returns nothing
        """
        keys = self.results.keys()
        if "X_ARRAY" in keys and "Y_ARRAY" in keys and "Z_ARRAY" in keys:
            fout=open(filename, "w+")
            fout.write('X,Y,Z\n')
            x=self.results['X_ARRAY']
            y=self.results['Y_ARRAY']
            z=self.results['Z_ARRAY']
            for i in range(len(x)):
                fout.write("%f,%f,%f\n" % (x[i], y[i], z[i]))
            fout.close()

    # local class for handling publication/subscription to the lassi_daq server

    class PipelineSubscriberThread(threading.Thread):
        "Subscriber thread for receiving scanner pipeline data"

        def __init__(self, tls_access):
            threading.Thread.__init__(self)

            def gen_random_string(rand_len=10, chars=string.ascii_uppercase +
                                                     string.ascii_lowercase + string.digits):
                """Generates a random sequence of characters of size 'rand_len' from
                   the characters provided in 'char'
                """
                return ''.join(random.choice(chars) for _ in range(rand_len))

            self.SUBSCRIBE       = tls_access.SUBSCRIBE
            self.UNSUBSCRIBE     = tls_access.UNSUBSCRIBE
            self.UNSUBSCRIBE_ALL = tls_access.UNSUBSCRIBE_ALL
            self.QUIT            = tls_access.QUIT
            self.PING            = tls_access.PING

            self.tls_access = tls_access
            self.pub_url = "tcp://" + tls_access.scanner_host + DATAPUB_PORT
            self.end_thread = False
            self._callbacks = {}
            # A local 'in-process' pipe to handle callback subscription requests
            self.pipe_url = "inproc://" + gen_random_string(20)
            self._ctx = tls_access._ctx

        def __del__(self):
            self.end_thread = True
            self.join()

        def run(self):
            try:
                ctx = self._ctx
                poller = zmq.Poller()
                self.subsock = ctx.socket(zmq.SUB)
                self.subsock.connect(self.pub_url)
                pipe = ctx.socket(zmq.REP)
                pipe.bind(self.pipe_url)

                poller.register(pipe, zmq.POLLIN)
                poller.register(self.subsock, zmq.POLLIN)

                # sytactic sugar
                subsock = self.subsock
                subsock.setsockopt(zmq.SUBSCRIBE, "".encode())

                while not self.end_thread:
                    event = poller.poll(200)
                    #print "done waiting event is ", event
                    for e in event:
                        sock = e[0]  # e[1] always POLLIN

                        if sock == pipe:
                            msg = pipe.recv_pyobj()

                            if msg == self.QUIT:
                                #pipe.send_pyobj(True)
                                self.end_thread = True
                                continue

                            elif msg == self.PING:
                                pipe.send_pyobj(True)


                        if sock == subsock:
                            msg = subsock.recv_multipart()
                            message_header = msg[0].decode()
                            # print("KEY=", message_header)
                            if message_header in "OK_STATUS":
                                status = self.tls_access.parse_OKStatus(msg)
                                self.tls_access.save_result(message_header, status)
                                for cbkey in self._callbacks.keys():
                                    mci = self._callbacks[cbkey]
                                    if mci is not None:
                                        mci(message_header, status)

                            elif message_header in "HEADER":
                                # We should see a multipart message here formatted like:
                                # "HEADER", header data
                                # "TIME_ARRAY" time array data
                                # "{X,Y,Z,I}_ARRAY the X,Y,Z,I (in that order) data
                                # 12 frames in all

                                self.tls_access.hdr_info = ScanHeaderInfo(msgpack.unpackb(msg[1]))
                                self.tls_access.save_result(message_header, self.tls_access.hdr_info)
                                if self.tls_access.hdr_info.tls_tilt_compensator != 0:
                                    print("******** ERROR **********")
                                    print("** Tilt Compensator is ON **")
                                    print("This will cause bad data when used in inverted position - you have been warned")
                                    print("*************************")
                                for i in range(2, len(msg), 2):
                                    key = msg[i].decode()
                                    if key in self.tls_access.array_names:
                                        numpy_array = msgpack.unpackb(msg[i+1], object_hook=msgpack_numpy.decode)
                                        self.tls_access.save_result(key, numpy_array)
                                    else:
                                        continue
                                        # Might want to only save if callback is None
                                        # for now do it unconditionally


                                for cbkey in self._callbacks.keys():
                                    mci = self._callbacks[cbkey]
                                    if mci is not None:
                                        mci(message_header, self.tls_access.get_results())


                pipe.close()
                self.subsock.close()

            #except zmq.core.error.ZMQError, e:
                #print 'In subscriber thread, exception zmq.core.error.ZMQError:', e
                print("pipe_url:", self.pipe_url)
                print(" pub_url:", self.pub_url)
            except BaseException as e:
                print("Exception in subscriber thread ", e)
                self.end_thread = True

            finally:
                print("TLSaccess: Ending subscriber thread.")




