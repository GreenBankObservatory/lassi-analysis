import os

import numpy as np
from astropy.io import fits

class SmoothedFITS:

    """
    This class is responsible for writting the product of the spherical
    smoothing to a simple FITS file
    """
    def __init__(self,
                 x,
                 y,
                 z,
                 N,
                 hdr,
                 dataDir,
                 proj,
                 filenameBase
                 ):

        # we won't be writing an image, but a simple binary table
        self.x = x.flatten()
        self.y = y.flatten()
        self.z = z.flatten()

        # but the dimensions of the data are NxN
        self.N = N

        # this is a dictionary derived from the header of the data stream
        self.hdr = hdr

        # where will the FITS file be written?
        self.dataDir = dataDir
        self.proj = proj
        self.filenameBase = filenameBase
        self.deviceName = 'LASSI'

    def getPath(self):
        "Returns directory where FITS file is written to"
        return os.path.join(self.dataDir,
                            self.proj,
                            self.deviceName,
                            )    

    def getFilePath(self):
        "Returns full path of final FITS file"
        fn = "%s.smoothed.fits" % self.filenameBase
        return os.path.join(self.getPath(), fn)

    def write(self):
        "Use class attributes to write smoothed data to disk as FITS file"
        # First extension just has our header
        header = fits.Header()
        for k, v in self.hdr.items():
            if k == "scan_time":
                v = v.strftime("%Y-%m-%d %H:%M:%S")
            header[k] = v
        # additional header info
        if "N" not in self.hdr:
            header["N"] = self.N
        pHdu = fits.PrimaryHDU(header=header)

        # second extension is our binary table
        xCol = fits.Column(name='x', format='D', unit='m', array=self.x)
        yCol = fits.Column(name='y', format='D', unit='m', array=self.y)
        zCol = fits.Column(name='z', format='D', unit='m', array=self.z)

        colDefs = fits.ColDefs([xCol, yCol, zCol])
        bHdu = fits.BinTableHDU.from_columns(colDefs)

        hdul = fits.HDUList([pHdu, bHdu])

        # finally, write to disk
        hdul.writeto(self.getFilePath())

def main():
    "simple test"
    N = 3
    x = np.zeros((N, N))
    y = np.zeros((N, N))
    z = np.zeros((N, N))
    hdr = {'TEST': 1}
    dataDir = "/users/pmargani/tmp"
    proj = "TEST"
    filenameBase = "test"

    f = SmoothedFITS(x, y, z, N, hdr, dataDir, proj, filenameBase)
    f.write()

if __name__ == '__main__':
    main()
