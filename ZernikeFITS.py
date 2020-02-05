import os

import numpy as np
from astropy.io import fits

from SmoothedFITS import SmoothedFITS

class ZernikeFITS(SmoothedFITS):
    
    def __init__(self):
        super().__init__()

        self.zernikes = None

        # override parent class
        self.ext = "zernike.fits"

    def setZernikes(self, zs):
        "Set the ordered list of zernike coefficients"
        self.zernikes = zs

    def makeZernikeHDU(self):
        if self.zernikes is None:
            return None

        idx = range(len(self.zernikes))

        iCol = fits.Column(name='coef', format='I', array=idx)
        zCol = fits.Column(name='value', format='D', array=self.zernikes)

        colDefs = fits.ColDefs([iCol, zCol])
        bHdu = fits.BinTableHDU.from_columns(colDefs)

        return bHdu

    def write(self):
        "Override parent method to include zernikes"    
        hdus = self.getHdus()

        # create the extension for the zernikes
        if self.zernikes is not None:
            hdu = self.makeZernikeHDU()
            print("hud type: ", type(hdu))
            print("hdus type: ", type(hdus))
            print(hdus)
            hdus.append(hdu)

        hdul = fits.HDUList(hdus)

        # finally, write to disk
        hdul.writeto(self.getFilePath())

def tryWrite():
    "simple test"
    N = 3
    x = np.zeros((N, N))
    y = np.zeros((N, N))
    z = np.zeros((N, N))
    hdr = {'TEST': 1}
    dataDir = "/users/pmargani/tmp"
    proj = "TEST"
    filenameBase = "test"

    f = ZernikeFITS()
    f.setData(x, y, z, N, hdr, dataDir, proj, filenameBase)
    f.setZernikes(range(3))
    f.write()

def tryRead():
    fn = "/home/sandboxes/pmargani/LASSI/data/TEST/LASSI/test.smoothed.fits"
    f = ZernikeFITS()
    f.read(fn)
    print (f.x.shape)
    print (f.hdr)
    print (f.hdr['N'])
    
if __name__ == '__main__':
    # main()
    # tryRead()
    tryWrite()    
