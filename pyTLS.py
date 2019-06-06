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

CONTROL_PORT=':55560'
DATAPUB_PORT=':55562'

class OKStatus:
    def __init__(self, g):
        self.error_msg = g[0]
        self.state = g[1]
        self.is_ok = g[2]

    def __str__(self):
        return "is_ok=%s, state=%d, message=%s" % (str(self.is_ok), self.state, self.error_msg)


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

    def __del__(self):
        "An attempt to shutdown the subscriber thread, which doesnt seem to work"
        self.cntrl_exit()

    def help(self):
        "A help page to guide you through"

        print("Method documentation:")
        help="""configure_scanner(project, resolution, sensitivity, scan_quality, scan_mode, center_az, center_el, fov_az, fov_el) """
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

        print("""subscribe(which_array, callback=None)""")
        print(self.subscribe.__doc__)

        print("""unsubscribe(which_array)""")
        print(self.unsubscribe.__doc__)

        print("""get_result(which_array)""")
        print(self.get_result.__doc__)

        print("""get_results()""")
        print(self.get_results.__doc__)

        print("""export_data()""")
        print(self.export_data.__doc__)

        print("""cntrl_exit()""")
        print(self.cntrl_exit.__doc__)

        print("""export_ptx_results(filename)""")
        print(self.export_ptx_results.__doc__)

        print("""export_csv_results(filename)""")
        print(self.export_csv_results.__doc__)

      

    def parse_OKStatus(self, msg):
        if msg is None or len(msg) < 1:
            return None

        kk = msgpack.unpackb(msg[1])
        status = OKStatus(kk)
        return status

    def encode_MoveAz(self, azcmd):
        return msgpack.packb([azcmd], use_bin_type=True)

    def encode_Config(self, project, res, sensitivity, scnrange, mode, ctr_az, ctr_el, fov_az, fov_el):
        return msgpack.packb([project, res, sensitivity, scnrange, mode, ctr_az, ctr_el, fov_az, fov_el])

    def simple_cmd(self, cmd):
        self.csock.send_string(cmd)
        rtn = self.csock.recv_multipart()
        return self.parse_OKStatus(rtn)

    def encode_string(self, string):
        return msgpack.packb([string])

    # API methods below

    def configure_scanner(self, project, resolution, sensitivity, scan_quality, scan_mode, center_az, center_el, fov_az, fov_el):
        """
        Sends a configuration to the scanner. Fields are:
            project : a valid string for tl_scanner project (not GBT project)
            resolution : an enum (0-6) specifying the resolution
            sensitivity: 0=normal, 1=high
            scan_quaility: 0-3
            scan_mode: enum (0-3) specifying Speed vs. Range
            center_az: center of scan in degrees
            center_el: center of scan in degrees
            fov_az: width of scan in degrees
            fov_el: height of scan in degrees
        returns an OKStatus object
        """
        msg = self.encode_Config(project, resolution, sensitivity, scan_quality, scan_mode, center_az, center_el, fov_az, fov_el)
        self.csock.send('Configure', zmq.SNDMORE)
        self.csock.send(msg)

        rtn = self.csock.recv_multipart()
        return self.parse_OKStatus(rtn)

    def move_az(self, az):
        """
        Sends a command to move the scanner base to the specified (instrument relative) azimuth.

        returns an OKStatus object
        """
        msg = self.encode_MoveAz(az)
        self.csock.send('MoveAz', zmq.SNDMORE)
        self.csock.send(msg)

        rtn = self.csock.recv_multipart()
        return self.parse_OKStatus(rtn)

    def subscribe_array(self, which_array):
        self.csock.send_string("ArraySubscribe", zmq.SNDMORE)
        msg = self.encode_string(which_array)
        self.csock.send(msg)
        rtn = self.csock.recv_multipart()
        return self.parse_OKStatus(rtn)

    def unsubscribe_array(self, which_array):
        self.csock.send_string("ArrayUnsubscribe", zmq.SNDMORE)
        msg = self.encode_string(which_array)
        self.csock.send(msg)
        rtn = self.csock.recv_multipart()
        return self.parse_OKStatus(rtn)

    def unsubscribe_array_all(self, which_array):
        for array_name in self.array_names:
            rtn = self.unsubscribe_array(array_name)
        return rtn

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
        return self.simple_cmd("GetStatus")

    def export_data(self):
        """
        Sends the command to command the lassi_daq to publish the most recent scan.
        :return: returns an OKStatus object
        """
        return self.simple_cmd("Export")

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
        """Allows access to the result cache. Note: in a future release results may
         only be saved if the callback was None. Should wrap this in a try-except
         because frame_type may not exist."""
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
        """Terminates the subscriber thread by sending it the 'QUIT' message,
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

    def subscribe(self, frame_type = None, cb_fun = None):
        """Subscribes to one or more of the pipeline data frames.
           Each frame is one vector from the latest TLS scan.
           If frame_type is None, subscriptions for x,y,z,i and time are issued.
           Frame types are:
                * "X_ARRAY" - x measurement vector as numpy array
                * "Y_ARRAY" - y measurement vector as numpy array
                * "Z_ARRAY" - z measurement vector as numpy array
                * "AZ_ARRAY" - az measurement vector as numpy array
                * "EL_ARRAY" - el measurement vector as numpy array
                * "R_ARRAY" - radial measurement vector as numpy array
                * "I_ARRAY" - intensity measurement vector as numpy array
                * "TIME_ARRAY - "MJD's for each pixel, as numpy array
            Each frame is subscribed independently, and can be mixed and matched.

        *frame_type:*
          the frame_type of interest. Must be a one of the 8 types listed above

        *cb_fun:*
          the callback function, which must take 2 args: the key, and a 1-D numpy array

        returns 'True' if the subscription was successful, 'False'
        otherwise. The function will fail if 'frame_type' is already
        subscribed.

        """

        if frame_type is None:
            self.subscribe("X_ARRAY", cb_fun)
            self.subscribe("Y_ARRAY", cb_fun)
            self.subscribe("Z_ARRAY", cb_fun)
            self.subscribe("I_ARRAY", cb_fun)
            return self.subscribe("TIME_ARRAY", cb_fun)

        # check to see if key exists
        if not frame_type in self.array_names:
            return (False, "'%s' is not a valid frame type." % frame_type)

        if cb_fun is not None:
            try:
                x = inspect.getargspec(cb_fun)

                # Should be a function that takes two parameters.
                if len(x.args) != 2:
                    return (False, 'Callback function must take 2 arguments')
            except TypeError:
                # not a function at all!
                return (False, 'Callback object is not a function!')


            # start the subscriber task if not already running
        if not self._sub_task:
            self._sub_task = TLSaccess.PipelineSubscriberThread(self)
            self._sub_task.start()
            sleep(1)  # give it time to start

            # there is already a callback, fail.
        if frame_type in self._sub_task._callbacks:
            return (False, "'%s' is already registered for a callback." % frame_type)

            # everything is good, set up the callback
        self._sub_task._callbacks[frame_type] = cb_fun
        pipe = self._ctx.socket(zmq.REQ)
        pipe.connect(self._sub_task.pipe_url)
        pipe.send_pyobj(self.SUBSCRIBE, zmq.SNDMORE)
        pipe.send_pyobj(frame_type)

        rval = pipe.recv_pyobj()
        msg = pipe.recv_pyobj()

        self.subscribe_array(frame_type)

        return (rval, msg)

    def unsubscribe(self, frame_type):
        """Unsubscribes a frame_type from the publishing interface

        *frame_type:*
          the frame_type of interest. Must be an array name as shown in
          the subscribe() section.

        returns 'True' if the frame_type was unsubscribed, 'False' if
        not. The function will fail if the frame_type was not previously
        subscribed.

        """
        self.unsubscribe_array(frame_type)

        if self._sub_task:
            pipe = self._ctx.socket(zmq.REQ)
            pipe.connect(self._sub_task.pipe_url)
            pipe.send_pyobj(self.UNSUBSCRIBE, zmq.SNDMORE)
            pipe.send_pyobj(frame_type)
            rval = pipe.recv_pyobj()
            msg = pipe.recv_pyobj()

            if frame_type in self.results.keys():
                self.results.pop(frame_type)
            return (rval, msg)
        return (False, 'No subscriber thread running!')

    def unsubscribe_all(self):
        """Causes all callbacks to be unsubscribed, and terminates the
           subscriber thread. Next call to 'subscribe' will restart
           it.

        """
        self.unsubscribe_array_all()

        if self._sub_task:
            pipe = self._ctx.socket(zmq.REQ)
            pipe.connect(self._sub_task.pipe_url)
            pipe.send_pyobj(self.UNSUBSCRIBE_ALL)
            rval = pipe.recv_pyobj()
            msg = pipe.recv_pyobj()
            self._kill_subscriber_thread()
            return (rval, msg)
        return (False, 'No subscriber thread running!')

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
            for i in range(len(x)):
                fout.write("%f %f %f %f\n" % (x[i], y[i], z[i], I[i]))
            fout.close()
        else:
            print("Not all data is available. This requires that the interface be subscribed")
            print("to each of X_ARRAY, Y_ARRAY, Z_ARRAY and I_ARRAY")
            print("Try subscribing to the arrays, then run export_data() again (no need to rerun the scan)")

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

                while not self.end_thread:
                    event = poller.poll(200)
                    #print "done waiting event is ", event
                    for e in event:
                        sock = e[0]  # e[1] always POLLIN

                        if sock == pipe:
                            msg = pipe.recv_pyobj()

                            if msg == self.SUBSCRIBE:
                                frame_type = pipe.recv_pyobj()
                                subsock.setsockopt(zmq.SUBSCRIBE, frame_type)

                                pipe.send_pyobj(True, zmq.SNDMORE)
                                pipe.send_pyobj(frame_type)

                            elif msg == self.UNSUBSCRIBE:
                                frame_type = pipe.recv_pyobj()

                                if frame_type in self._callbacks:
                                    subsock.setsockopt(zmq.UNSUBSCRIBE, frame_type)
                                    self._callbacks.pop(frame_type)

                                    pipe.send_pyobj(True, zmq.SNDMORE)
                                    pipe.send_pyobj("'%s' unsubscribed." % frame_type)
                                else:
                                    pipe.send_pyobj(False, zmq.SNDMORE)
                                    pipe.send_pyobj("'%s': No such frame_type is subscribed." % frame_type)


                            elif msg == self.UNSUBSCRIBE_ALL:
                                keys_cleared = self._callbacks.keys()

                                for frame_type in self._callbacks:
                                    subsock.setsockopt(zmq.UNSUBSCRIBE, frame_type)

                                self._callbacks.clear()
                                pipe.send_pyobj(True, zmq.SNDMORE)
                                pipe.send_pyobj('Keys cleared: %s' % ', '.join(keys_cleared))

                            elif msg == self.QUIT:
                                #pipe.send_pyobj(True)
                                self.end_thread = True
                                continue

                            elif msg == self.PING:
                                pipe.send_pyobj(True)


                        if sock == subsock:
                            msg = subsock.recv_multipart()
                            key = msg[0]
                            #print "pubsock activity"

                            if len(msg) > 1:
                                numpy_array = msgpack.unpackb(msg[1], object_hook=msgpack_numpy.decode)
                                if key in self._callbacks:
                                    mci = self._callbacks[key]
                                    if mci is not None:
                                        mci(key, numpy_array)
                                # Might want to only save if callback is None
                                # for now do it unconditionally
                                self.tls_access.save_result(key, numpy_array)
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



def mycb(a, b):
    print("\nGOT frame ", a, " of length ", len(b), " of type ", type(b))


