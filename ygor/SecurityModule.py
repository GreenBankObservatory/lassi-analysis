# Copyright (C) 2004 Associated Universities, Inc. Washington DC, USA.
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
# $Id: SecurityModule.py,v 1.7 2005/01/25 14:36:08 windyclark Exp $

import os  
import sys
from SOAPpy    import Types 
from .getConfigValue  import getConfigValue
# TBF this does not work!? from gbt.ygor  import getConfigValue

class SecurityModule:

    def _read_security_file(self):
        ygor_telescope = getConfigValue(".", "YGOR_TELESCOPE")
        fd = open(ygor_telescope + '/etc/config/securedb.txt', 'r')
        self.users = {}
        for line in fd.xreadlines():
            if line[0] != '#' and line [0] != '\n':
                name, cdr = line.split('@')
                devices = cdr.split()[1:]
                if self.users.has_key(name):
                    devices = self.users[name] + devices
                self.users[name] = devices

    def has_antenna(self, user):
        "Returns whether the given user has permission for the antenna."
        self._read_security_file()
        if not self.users.has_key(user):
            return False
        else:
            return 'Antenna' in self.users[user]

    def get_user(self):
        line = os.popen('whoami').readline() 
        user, nl = line.split('\n')
        return user

    def check_user(self):
        """ Quick security function -- interim solution for beta release""" 
        grailHost = getConfigValue("localhost", "GrailHost")
        user = self.get_user()
        self._read_security_file()
        if user in self.users.keys() or 'all' in self.users.keys():
            return True
        else:    
            sys.exit("You do not have permission to operate the telescope")
