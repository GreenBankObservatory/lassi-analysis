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

import sys
from datetime import datetime

import numpy as np
from astropy.time import Time

from .YgorDevice import YgorDevice

zernike_type = "thermal_zernike"
zernike_param = "%s_coeff" % zernike_type
num_zernikes = 36

class ActiveSurface(YgorDevice):
    """
    Interface class for communcating with the ActiveSurface.
    Currently added value is sending of Zernikes.
    """

    def __init__(self, cbPort = 19701):

        # We'd like to do this but YgorDevice messes with self
        #self.zs = None
        #self.zernike_type = "thermal_zernike"
        #self.zernike_param = "%s_coeff" % self.zernike_type

        YgorDevice.__init__(self, cbPort)

    def dt2mjd(self, dt):

        t = Time(dt, format='datetime')
        t.format = 'mjd'
        return t.value

    def startScan(self):

        now = datetime.utcnow()

        # get MJD.fractionaDay
        fmjd = self.dt2mjd(now)

        # breakup into MJD, seconds since midnight
        mjd = int(fmjd)
        fracDay = fmjd - mjd
        secs = int(fracDay * (24*60*60))

        time = {}
        time["theMJD"] = mjd
        time["theSec"] = secs
        time["flags"] = 0
        time["refFrame"] = 1
        time["units"] = 1

        self.manager.start_at(time)

    def GetName(self):
        return "ActiveSurface.ActiveSurface"

    def LoadZernikes(self, fn):
        "Get the zernikes from a numpy file"
        return np.load(fn)

    def TurnOnThermalZernikes(self):
        # turn on these zernikes
        param = "correctionSelect,%s" % zernike_type
        self.manager.set_value(param, str(1))

    def SetThermalZernike(self, zi, value, prepare=True):
        param = "%s,%d" % (zernike_param, zi)
        self.manager.set_value(param, value)

        if prepare:
            self.manager.send_values()
            self.manager.prepare()

    def ZeroThermalZernikes(self):
        # send each one
        for i in range(num_zernikes):
            zi = i + 1
            self.SetThermalZernike(zi, 0.0, prepare=False)

        self.manager.send_values()
        self.manager.prepare()

    def SendZernikes(self, fn):
        "Send the zernikes to the manager via the chosen parameter"

        zs = self.LoadZernikes(fn)

        if zs is None:
            return

        self.TurnOnThermalZernikes()

        # send each one
        for i in range(len(zs)):
            z = zs[i]
            zi = i + 1
            # param = "%s,%d" % (zernike_param, zi)
            # self.manager.set_value(param, z)
            self.SetThermalZernike(zi, z, prepare=False)

        self.manager.send_values()
        self.manager.prepare()


if __name__ == '__main__':
    zfile = sys.argv[1]
    act = ActiveSurface()
    act.SendZernikes(zfile)
