# Copyright (C) 2006 Associated Universities, Inc. Washington DC, USA.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 675 Mass Ave Cambridge, MA 02139, USA.
#
# Correspondence concerning GBT software should be addressed as follows:
#     GBT Operations
#     National Radio Astronomy Observatory
#     P. O. Box 2
#     Green Bank, WV 24944-0002 USA
#
# $Id

from getConfigValue import getConfigValue
from GrailClient    import GrailClient

class YgorConnection:
    """
    Buffer between the real world and this YgorDevice class.  It keeps the
    units tests away from Grail.
    """
    def __init__(self, cbPort, host = None):
        if host == None:
            grailHost = getConfigValue("localhost", "GrailHost")
        else:
            grailHost = host
            
        self.grailClient = GrailClient(grailHost, 18000, cb_port = cbPort)
        self.manager     = None

    def __del__(self):
        "Clean up."
        if self.manager is not None:
            del self.manager
            del self.grailClient

    def CreateManager(self, manager):
        self.manager = self.grailClient.create_manager(manager)

class YgorDevice:
    "Interface class for communcating with any Ygor device."

    def __init__(self, cbPort = 19101, host = None):
        "Create member variables and set up."

        # Connect to Ygor.
        self.CreateConnection(cbPort, host)
        self.CreateManager(self.GetName())

        # Initialize data members.
        self.callbacks = {}

    def __del__(self):
        "Clean up."
        for k, v in self.callbacks.iteritems():
           if v[0] == "parameter":
               self.manager.unreg_param(k, v[1])
           elif v[0] == "sampler":
               self.manager.unreg_sampler(k, v[1])
           else:
               raise Exception("Unknown callback: %s, %s" % (k, v[1]))

        del self.connection

    def __getattr__(self, name):
        "If YgorDevice doesn't have it, then try YgorConnection."
        return getattr(self.connection, name)

    def __setattr__(self, name, value):
        "Identify sanctioned attributes of YgorDevice."
        if name in ("connection", "callbacks"):
            self.__dict__[name] = value
        else:
            raise Exception("Illegal YgorConnection attribute", name)

    def CreateConnection(self, cbPort, host = None):
        # Connect to Ygor.
        self.connection = YgorConnection(cbPort, host)

    def GetName(self):
        "Which Manager are we talking about here?"
        raise Exception("This is a pure virtual method.")

    def GetParameterValue(self, parameter):
        "Returns all fields and description of desired Parameter."
        return self.manager.get_parameter(parameter)

    def GetSamplerValue(self, sampler):
        "Returns all fields and description of desired Sampler."
        return self.manager.get_sampler(sampler)

    def GetValue(self, parameter):
        "Returns a parameter field value, without parameter description fields."
        return self.manager.get_value(parameter)

    def RegisterParameter(self, name, callback, fields = ''):
        "Register for updates when a parameter values changes."
        self.manager.unreg_param(name, callback)
        self.manager.reg_param(name, callback, fields)
        self.callbacks[name] = ("parameter", callback)

    def RegisterSampler(self, name, callback, fields = ''):
        "Register for updates when a sampler values changes."
        self.manager.unreg_sampler(name, callback)
        self.manager.reg_sampler(name, callback, fields)
        self.callbacks[name] = ("sampler", callback)

    def UnregisterParameter(self, name, callback):
        "No longer listen for updates when a bank Parameter value changes."
        self.manager.unreg_param(name, callback)
        self.callbacks.pop(name)

    def UnregisterSampler(self, name, callback):
        "No longer listen for updates when a bank Sampler value changes."
        self.manager.unreg_sampler(name, callback)
        self.callbacks.pop(name)
