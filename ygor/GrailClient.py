######################################################################
#  GrailClient provides a simplified interface to the SOAP Client to
#  Grail.  In addition, it provides a callback mechanism that allows
#  a client to asynchnonously receive updated parameters and samplers
#  that it has registered to receive.
#
#  Copyright (C) 2003 Associated Universities, Inc. Washington DC, USA.
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#  General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
#
#  Correspondence concerning GBT software should be addressed as follows:
#  GBT Operations
#  National Radio Astronomy Observatory
#  P. O. Box 2
#  Green Bank, WV 24944-0002 USA
#
# $Id: GrailClient.py,v 1.21 2008/08/07 14:27:30 rcreager Exp $
#
######################################################################

from .SecurityModule import *
from select import select, error
import threading
import SOAPpy
import types

import logging


logger = logging.getLogger(__name__)

class GrailClient:

    class YgorParameter:

        def __init__(self, s):
            self.str = ""
            self.value = {}
            self.value['name'] = s['name']
            self.value['description'] = s['description']
            self.value['id'] = s['id']
            l = {}
            self.value[s['name']] = l
            self.convert_soapy_struct(s, l)

        def __repr__(self):
            self.str = str(self.value)
            return self.str

        def convert_soapy_struct(self, s, fdict):

            for i in s['field']:

                if i._asdict().has_key('field'):
                    nf = {}
                    fdict[i['name']] = nf
                    self.convert_soapy_struct(i, nf)
                elif i._asdict().has_key('value'):
                    fdict[i['name']] = i._asdict()

    class ParameterEntries:
        def add_parameters(self, entry_list):
            for i in range(len(entry_list)):
                if type(entry_list[i]) is types.ListType:
                    self.entries.append(entry_list[i])
                else:
                    self.entries.append(entry_list)

        def add_param(self, mng, name, value):
            '''Set the manager, parameter, value'''
            duplicate = 0
            if type(self.entries) is types.ListType:
                for e in self.entries:
                    if e[0] ==  mng and e[1] == name and e[2] == value:
                        duplicate = 1
                        break
            else:
                if e[0] ==  mng and e[1] == name and e[2] == value:
                    duplicate = 1

            if not duplicate:
                entry = [mng, name, value]
                self.entries.append(entry)

        def delete_entries(self):
            self.entries = []

        def _remove_duplicates(self, mng, params):
            for entry in self.entries:
                if mng == entry[0]:
                    if params.has_key(entry[1]):
                        if entry[2] == params[entry[1]]:
                            del param[entry[1]]
            return params

        def get_entries(self):
            return self.entries

    class PublishCallbacks(threading.Thread):

        def __init__(self, port):
            self.server_port = port
            self.PCBD = {}
            self.SCBD = {}
            self.__stop = threading.Event()
            self.__lock = threading.Lock()
            self.__started = threading.Event()
            threading.Thread.__init__(self)

        def wait_for_start(self):
            self.__started.wait()

        def _register(self, device, parameter, cb, cb_dict):
            '''Registers a parameter or sampler to a callback list.'''

            self.__lock.acquire()

            if device.find('.') == -1:
                if device.find(',') == -1:
                    device = device + '.' + device

            if device not in cb_dict:
                cb_dict[device] = {}

            if parameter not in cb_dict[device]:
                cb_dict[device][parameter] = set()

            cb_dict[device][parameter].add(cb);

            self.__lock.release()


        def _unregister(self, device, parameter, cb, cb_dict):
            '''Unregisters a parameter or sampler from a callback list'''

            self.__lock.acquire()

            if device.find('.') == -1:
                if device.find(',') == -1:
                    device = device + '.' + device

            if device in cb_dict:
                if parameter in cb_dict[device]:
                    cb_dict[device][parameter].discard(cb)

                    if len(cb_dict[device][parameter]) == 0:
                        del cb_dict[device][parameter]

                        if len(cb_dict[device]) == 0:
                            del cb_dict[device]

            self.__lock.release()

        def _terminate(self):
            self.__stop.set()
            threading.Thread.join(self, 10)

        def register_parameter(self, device, parameter, cb):
            self._register(device, parameter, cb, self.PCBD)

        def unregister_parameter(self, device, parameter, cb):
            self._unregister(device, parameter, cb, self.PCBD)

        def register_sampler(self, device, sampler, cb):
            self._register(device, sampler, cb, self.SCBD)

        def unregister_sampler(self, device, sampler, cb):
            self._unregister(device, sampler, cb, self.SCBD)


        def run(self):

            class callback_services:

                def __init__(self, parent):
                    self.parent = parent

                def _update_data(self, device, param, value, cb_dict):
                    if device in cb_dict:
                        if param in cb_dict[device]:
                            for i in cb_dict[device][param]:
                                a = GrailClient.YgorParameter(value)
                                i(device, param, a.value)
                            return 1
                        else:
                            return 0
                    else:
                        return 0

                def _update_field(self, device, param, value, cb_dict):
                    # print "_update_field(", device, param, value, ")"
                    if device in cb_dict:
                        # print "device in cb_dict"
                        if param in cb_dict[device]:
                            for i in cb_dict[device][param]:
                                i(device, param, value)
                            return 1
                        else:
                            return 0
                    else:
                        return 0

                def UpdateParameter(self, device, parameter, value):
                    return self._update_data(device, parameter, value, self.parent.PCBD)

                def UpdateSampler(self, device, sampler, value):
                    return self._update_data(device, sampler, value, self.parent.SCBD)

                def UpdateParameterFields(self, device, parameter, values):
                    return self._update_field(device, parameter, values, self.parent.PCBD)

                def UpdateSamplerFields(self, device, sampler, values):
                    # print "UpdateSamplerFields(", device, sampler, values, ")"
                    return self._update_field(device, sampler, values, self.parent.SCBD)

            td = threading.local()
            td.server_status = 0

            while td.server_status == 0:
                try:
                    td.server = SOAPpy.SOAPServer(("0.0.0.0", self.server_port), \
                                                  namespace = "urn:grail-client", log = 0)
                    td.server_status = 1
                except:
                    self.server_port = self.server_port + 1
                    td.server_status = 0

            self.__started.set() # release starting thread, which needs server_port value
            # comment out the following lines if not debugging
            # td.server.config.dumpSOAPIn = 1
            # td.server.config.dumpSOAPOut = 1
            td.server.registerObject(callback_services(self))

            while not self.__stop.isSet():

                try:
                    td.readable = [td.server.socket]
                    td.writable = []
                    td.error = []
                    td.r, td.w, td.e = select(td.readable, td.writable, td.error, 1.0)

                    if td.r:
                        self.__lock.acquire()
                        td.server.handle_request()
                        self.__lock.release()

                except error as e:
                    # If the error is an interrupted system call then
                    # keep on running.  If it is something else, then
                    # pass it on.  The interrupted system call error
                    # is (4, 'Interrupted system call').
                    if e[0] != 4:
                        raise
                    else:
                        print("### GrailClient: intercepted interrupted system call exception:", e)


    def __init__(self, host = None, port = 18000, cb_host = None, cb_port = 18005,
                 urn = 'urn:grail', ssh_tunnel = False):
        from socket import gethostname, getfqdn

        # If tunneling through SSH, host names can be assumed.  The
        # server host can be assumed to be 'localhost'.  On the server
        # end, the server needs to send callbacks to itself. One would
        # think that providing the remote's (server's) host name would
        # be sufficient, but that will fail if the remote host name is
        # fully qualified(!).  Thus we also make this host name
        # 'localhost'.  If not using a tunnel provide fully qualified
        # name of the client computer

        if ssh_tunnel:
            host = cb_host = 'localhost'
        else:
            if host == None:
                host = gethostname()

            if cb_host == None:
                cb_host = gethostname()

            cb_host = getfqdn(cb_host)

        self.url = 'http://' + host + ':' + str(port)
        self.urn = urn
        self.cl = SOAPpy.SOAPProxy(self.url, namespace = self.urn)
        self.security = SecurityModule()
        self.parameters = GrailClient.ParameterEntries()
        self.cb_server = GrailClient.PublishCallbacks(cb_port)
        self.cb_server.setDaemon(1)
        # IMPORTANT: The cb_server may modify the callback port
        # number.  Make sure to build self.cb_url *after* the
        # cb_server has started.
        self.cb_server.start()
        self.cb_server.wait_for_start()
        self.cb_url = 'http://' + cb_host + ':' + str(self.cb_server.server_port)

    def __del__(self):
        self.cb_server._terminate()

    def delete_entries(self):
        self.parameters.delete_entries()

    def add_parameters(self, entry_list):
        self.parameters.add_parameters(entry_list)

    def get_devices(self):
        devices = []
        for parameter in self.parameters.get_entries():
            if parameter[0] not in devices:
                 devices.append(parameter[0])
        return(devices)

    def send_to_tel(self, force = 0, prepare = 1):
        self.security.check_user()
        param_list = Types.arrayType()

        for entry in self.parameters.entries:
            item = Types.structType()
            item._typename = 'anyType'
            item._addItem('device',entry[0])
            item._addItem('path',entry[1])
            item._addItem('value',entry[2])
            param_list.append(item)
        self.cl.set_values_array(param_list, force, prepare)

    def get_values_array(self, params):
        def mkypv(d, p, v, t):
            a = SOAPpy.Types.structType()
            a._typename = "anyType"
            a._addItem("device", d)
            a._addItem("path", p)
            a._addItem("value", v)
            a._addItem("type", t)
            return a
        a = SOAPpy.Types.arrayType()

        b = []
        for i in params:
            a.append(mkypv(i[0], i[1], i[2], i[3]))

        rtrn = self.cl.get_values_array(a, " " ," ",b)

     #TBF will need to do this - or something like it when Grail is ready
      #  rtrn.error_message ='Ok'
      #  status = rtrn.error
      #  if status:
      #      faultlist = rtrn.error_message

      #  for i in rtrn.data:
      #      print i
      #  print "******"
        for value in rtrn:
            b.append((value.device, value.path, value.value, value.type))

        return b

    def reg_param(self, device, parameter, cb, fields = ''):
        self.cb_server.register_parameter(device, parameter, cb)
        self.cl.reg_parameter(device, parameter, self.cb_url, fields)

    def unreg_param(self, device, parameter, cb):
        self.cb_server.unregister_parameter(device, parameter, cb)
        self.cl.unreg_parameter(device, parameter, self.cb_url)


    def reg_sampler(self, device, sampler, cb = None, fields = ''):
        if cb != None:
            self.cb_server.register_sampler(device, sampler, cb)
            self.cl.reg_sampler(device, sampler, self.cb_url, fields)
        else:
            self.cl.reg_sampler(device, sampler)


    def unreg_sampler(self, device, sampler, cb = None):
        if cb != None:
            self.cb_server.unregister_sampler(device, sampler, cb)
            self.cl.unreg_sampler(device, sampler, self.cb_url)
        else:
            self.cl.unreg_sampler(device, sampler)


    def show_managers(self):
        return self.cl.show_managers()


    def create_manager(self, device):
        return Manager(self, device)


    def abort(self, device):
        self.security.check_user()
        return self.cl.abort(device)


    def activate(self, device):
        self.security.check_user()
        return self.cl.activate(device)


    def check(self, device):
        self.security.check_user()
        return self.cl.check(device)


    def clear_locks(self, device):
        self.security.check_user()
        return self.cl.clear_locks(device)


    def conform(self, device):
        self.security.check_user()
        return self.cl.conform(device)


    def get_parameter(self, device, param):
        a = self.YgorParameter(self.cl.get_parameter(device, param))
        return a.value

    def get(self, device, param):
        a = 'Failed'
       #TBF if param is(sampler):
       # TBF doesnt work all the time!! a = self.get_parameter(device, param)
        a = self.cl.get_value(device, param)
        #TBF  else:
       #TBF     a = self.get_sampler(device, param)
        return a

    def get_sampler(self, device, sampler):
        a = self.YgorParameter(self.cl.get_sampler(device, sampler))
        return a.value

    def balance(self, mng, option):
        self.security.check_user()
        self.set_value(mng, 'balance', option)
        self.prepare(mng)

    def get_sampler_value(self, device, path):
        return self.cl.get_sampler_value(device, path)


    def get_value(self, device, path):
        return self.cl.get_value(device, path)


    def invoke(self, device, id):
        self.security.check_user()
        return self.cl.invoke_param(device, id)


    def off(self, device):
        self.security.check_user()
        return self.cl.off(device)


    def on(self, device):
        self.security.check_user()
        return self.cl.on(device)


    def prepare(self, device):
        self.security.check_user()
        return self.cl.prepare(device)


    def recalculate(self, device):
        self.security.check_user()
        return self.cl.recalculate(device)


    def reg_change(self, device):
        self.security.check_user()
        return self.cl.reg_change(device)


    def reset(self, device):
        self.security.check_user()
        return self.cl.reset(device)


    def revert(self, device):
        self.security.check_user()
        return self.cl.revert(device)


    def send_values(self, device):
        logger.debug("device=%s", device)
        self.security.check_user()
        return self.cl.send_values(device)


    def set_lock(self, device, id):
        self.security.check_user()
        return self.cl.set_lock(device, id)


    def set_value(self, device, path, value, force = 0):
        status = 'OK'
        self.security.check_user()
        if self.cl.set_value(device, path, value, force) == 'OK':
            if self.send_values(device) != 'OK':
                status = 'Error sending parameters to M&C\n'
        else:
            status = 'Error setting parameters\n'
        return status

    def set(self, mng, params, force = 0):
        self.security.check_user()
        status = ''
        parm_value_pairs = params.items()
        for param, value in parm_value_pairs:
            ok = self.cl.set_value(mng, param, value, force)
           #status.append((mng, param, value, ok))
        if self.send_values(mng) != 'OK':
            status = 'Error sending parameters to MnC\n'
        status += self.reg_change()
        return status


    def set_values_array(self, params, force = 0, prepare = 1):

        self.security.check_user()
        def mkypv(d, p, v):
            a = SOAPpy.Types.structType()
            a._typename = "anyType"
            a._addItem("device", d)
            a._addItem("path", p)
            a._addItem("value", v)
            return a

        a = SOAPpy.Types.arrayType()

        for i in params:
            a.append(mkypv(i[0], i[1], i[2]))

        return self.cl.set_values_array(a, force, prepare)


    def show_params(self, device):
        p = self.cl.show_params(device)
        r = []

        for i in p:
            r.append({'name':i['name'], 'description':i['description']})

        return r


    def show_samplers(self, device):
        s = self.cl.show_samplers(device)
        r = []
        for i in s:
            r.append({'name':i['name'], 'description':i['description']})

        return r


    def standby(self, device):
        self.security.check_user()
        return self.cl.standby(device)

    def start_at(self, device, starttime):
        logger.debug("device=%s, starttime=%s", device, starttime)

        ts = SOAPpy.Types.structType()
        ts._typename = "anyType"
        ts.theMJD = starttime["theMJD"]
        ts.theSec = starttime["theSec"]
        ts.refFrame = starttime.setdefault("refFrame", 1)
        ts.units = starttime.setdefault("units", 1)
        return self.cl.start_at(device, ts)


    def start_sampler(self, device, sampler, cb_url = ''):
        return self.cl.start_sampler(device, sampler, cb_url)

    def stop(self, device):
        self.security.check_user()
        return self.cl.stop(device)

    def stop_sampler(self, device, sampler, cb_url = ""):
        self.cl.stop_sampler(device, sampler, cb_url)

    def stop_at(self, device, stoptime):
        ts = SOAPpy.Types.structType()
        ts._typename = "anyType"
        ts.theMJD = stoptime["theMJD"]
        ts.theSec = stoptime["theSec"]
        ts.refFrame = stoptime.setdefault("refFrame", 1)
        ts.units = stoptime.setdefault("units", 1)
        return self.cl.stop_at(device, ts)


    def sync(self, device):
        self.cl.sync(device)

    def unlock(self, device, id):
        self.security.check_user()
        self.cl.unlock(device, id)

#*********************************************************************
# Manager, allows direct access to a manager without having to specify
# the manager name.  Makes callthroughs to GrailClient using the
# 'dev' attribute.
#*********************************************************************

class Manager:

    def __init__(self, grail_client, devicename):
        self.dev = devicename
        self.cl = grail_client

    def GetDeviceName(self):
        return self.dev

    def abort(self):
        return self.cl.abort(self.dev)


    def activate(self):
        return self.cl.activate(self.dev)


    def check(self):
        return self.cl.check(self.dev)


    def clear_locks(self):
        return self.cl.clear_locks(self.dev)


    def conform(self):
        return self.cl.conform(self.dev)


    def get_parameter(self, param):
        return self.cl.get_parameter(self.dev, param)

    def get_sampler(self, sampler):
        return self.cl.get_sampler(self.dev, sampler)


    def get_sampler_value(self, path):
        return self.cl.get_sampler_value(self.dev, path)


    def get_value(self, path):
        return self.cl.get_value(self.dev, path)


    def invoke(self, id):
        return self.cl.invoke_param(self.dev, id)


    def off(self):
        return self.cl.off(self.dev)


    def on(self):
        return self.cl.on(self.dev)


    def prepare(self):
        return self.cl.prepare(self.dev)


    def recalculate(self):
        return self.cl.recalculate(self.dev)


    def reg_change(self):
        return self.cl.reg_change(self.dev)


    def reset(self):
        return self.cl.reset(self.dev)


    def revert(self):
        return self.cl.revert(self.dev)


    def send_values(self):
        return self.cl.send_values(self.dev)


    def set_lock(self, id):
        return self.cl.set_lock(self.dev, id)


    def set_value(self, path, value, force = 0):
        return self.cl.set_value(self.dev, path, value, force)


    def set_values_array(self, params, force = 0, prepare = 1):
        a = []

        for i in params:
            a.append([self.dev, i[0], i[1]])

        return self.cl.set_values_array(a, force, prepare)

    def show_params(self):
        return self.cl.show_params(self.dev)


    def show_samplers(self):
        return self.cl.show_samplers(self.dev)


    def standby(self):
        return self.cl.standby(self.dev)

    def start_at(self, starttime):
        return self.cl.start_at(self.dev, starttime)

    def stop(self):
        return self.cl.stop(self.dev)

    def stop_at(self, stoptime):
        return self.cl.stop_at(self.dev, stoptime)

    def sync(self):
        self.cl.sync(self.dev)


    def unlock(self, id):
        self.cl.unlock(self.dev, id)


    def reg_param(self, parameter, cb, fields = ''):
        self.cl.reg_param(self.dev, parameter, cb, fields)


    def unreg_param(self, parameter, cb):
        self.cl.unreg_param(self.dev, parameter, cb)


    def reg_sampler(self, sampler, cb = None, fields = ''):
        self.cl.reg_sampler(self.dev, sampler, cb, fields)


    def unreg_sampler(self, sampler, cb = None):
        self.cl.unreg_sampler(self.dev, sampler, cb)
