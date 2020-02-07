import os

import numpy as np
from astropy.io import fits

class SmoothedFITS:

    """
    This class is responsible for writting the product of the spherical
    smoothing to a simple FITS file
    """

    def __init__(self):
        
        self.deviceName = 'LASSI'
        self.ext = "smoothed.fits"

    def setData(self,
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
        
    def getPath(self):
        "Returns directory where FITS file is written to"
        return os.path.join(self.dataDir,
                            self.proj,
                            self.deviceName,
                            )    

    def getFilePath(self):
        "Returns full path of final FITS file"
        fn = "%s.%s" % (self.filenameBase, self.ext)
        return os.path.join(self.getPath(), fn)

    def write(self):
        hdus = self.getHdus()

        hdul = fits.HDUList(hdus)
 
        # finally, write to disk
        hdul.writeto(self.getFilePath())


    def getHdus(self):    
        "Use class attributes to write smoothed data to disk as FITS file"
        # First extension just has our header
        header = fits.Header()
        for k, v in self.hdr.items():
            if k == "scan_time":
                v = v.strftime("%Y-%m-%d %H:%M:%S")
            if k != 'COMMENT':    
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

        return [pHdu, bHdu]

    def read(self, filepath):

        self.hdus = fits.open(filepath)

        self.hdr = self.hdus[0].header

        data = self.hdus[1].data

        self.x = data.field('x')
        self.y = data.field('y')
        self.z = data.field('z')

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

    f = SmoothedFITS()
    f.setData(x, y, z, N, hdr, dataDir, proj, filenameBase)
    f.write()

def tryRead():
    # fn = "/home/sandboxes/pmargani/LASSI/data/TEST/LASSI/test.smoothed.fits"
    fn = "/export/simdata/JUNK/LASSI/2020_02_07_18:26:20.fits"
    f = SmoothedFITS()
    f.read(fn)
    x = f.x      
    y = f.y      
    z = f.z
    i = f.hdus[1].data.field("INTENSIT")
    dts = f.hdus[1].data.field('DMJD')
    hdr = dict(f.hdr)
    print ("hdr:", hdr)     
    print (f.x.shape)
    for k, v in f.hdr.items():
        print (k, v)
    # print (f.hdr['N'])
    
    N = 512
    dataDir = "/users/pmargani/tmp"
    proj = "TEST"
    filenameBase = "test"

    f2 = SmoothedFITS()
    f2.setData(x, y, z, N, hdr, dataDir, proj, filenameBase)
    f2.write()

if __name__ == '__main__':
    # main()
    tryRead()
    #tryWrite()
