from copy import copy

import numpy as np
import opticspy

from zernikeIndexing import noll2asAnsi, printZs

def fitZernikies(data):
    "Fits zernikies to data and converts from NOLL to AS ANSI"

    # replace NaNs with zeros
    dataOrg = copy(data)
    data[np.isnan(data)] = 0.

    # print scaling up data for z-fit by 1000.
    dataM = data * 1000.

    # find the first 12 Zernike terms
    numZsFit = 36
    fitlist,C1 = opticspy.zernike.fitting(dataM,
                                          numZsFit,
                                          remain2D=1,
                                          barchart=1)
    print "fitlist: ", fitlist
    C1.listcoefficient()
    C1.zernikemap()

    print "Converting from Noll to Active Surface ANSI Zernikies ..."
    # and now convert this to active surface zernike convention
    # why does the fitlist start with a zero? for Z0??  Anyways, avoid it
    nollZs = fitlist[1:(numZsFit+1)]
    asAnsiZs = noll2asAnsi(nollZs)
    print "nolZs"
    printZs(nollZs)
    print "active surface Zs"
    printZs(asAnsiZs)

    return fitlist, asAnsiZs
