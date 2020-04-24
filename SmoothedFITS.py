import os
from copy import copy

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

        self.expHeader = {
            # set by client, standard M&C header stuff
            "PROJID" : "Manager parameter projectId",
            "SCAN" : "Manager parameter scanNumber",
            # TBF: to bad, lots of other stuff, but
            # the analysis server won't know this
            # set by client
            "REFSCAN" : "Is this a reference scan?",
            "RSCANNUM" : "Scan number of reference scan",
            "N" : "size of smoothing grid",
            # passed from TLS stream
            "TLSAZCTR" : "instrument scan center azimuth in degrees",
            "TLSELCTR" : "instrument scan center elevation in degrees",
            "TLSAZFOV" : "scan width in radians",
            "TLSELFOV" : "scan height in radians",
            "TLSSCAN"  : "TL Scanner project scan number",
            "TLSPROJE" : "TL Scanner project name",
            "TLSRESOL" : "TL Scanner resolution",
            "TLSSMODE" : "TL scanner (EDM) mode setting",
            "TLSSENSI" : "TL Scanner sensitivity setting",
            "TLSSERIA" : "TL Scanner serial number",
            "TLSQUALI" : "TL Scanner tile compensator",
            # TBF: this is not in the raw FITS file?
            # "scan_time": "TBF"

        }

        self.standardHeaders = {
            "SIMPLE" : "conforms to FITS standard",
            "BITPIX" : "array data type",
            "NAXIS"  : "number of array dimensions",
            "EXTEND" :  ""
        }

        self.x = None
        self.y = None
        self.z = None
        self.N = None
        self.hdr = {}
        self.dataDir = None
        self.proj = None
        self.filenameBase = None
        
    def setData(self,
                 x,
                 y,
                 z,
                 N,
                 hdr,
                 dataDir,
                 proj,
                 filenameBase,
                 scanNum
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
        
        self.scanNum = scanNum

        # update the header to include what we 
        # can that's like the raw FITS file
        self.hdr["PROJID"] = (self.proj, self.expHeader["PROJID"])
        self.hdr["SCAN"] = (self.scanNum, self.expHeader["SCAN"])

        # update the header with info about smoothing
        self.hdr["N"] = (self.N, "size of smoothing grid")

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

    def zmqHeader2fitsHeader(self, zmqHeader):
        """
        When the analysis server gets raw data to process, it
        comes with a header dictionary.  The manager's FITS writer
        interprets this zmq header into appropriate entries
        in the raw FITS file.
        We want to do the same thing here so that our PHDU is
        close as possible to the raw FITS file.
        """
        pass

    def write(self):
        hdus = self.getHdus()

        hdul = fits.HDUList(hdus)
 
        # finally, write to disk
        hdul.writeto(self.getFilePath())


    def checkHeader(self, addStandardKeys=False):
        "Check the current header dictionary against what we'd expect"

        # strip current header of values
        tmp = {}
        for k, v in self.hdr.items():
            tmp[k] = v[1]

        # when we write the FITS file, these keys get added
        expHeader = copy(self.expHeader)
        if addStandardKeys:
            for k, v in self.standardHeaders.items():
                expHeader[k] = v

        if tmp == expHeader:
            print("Header is as expected!")
            return True

        added, removed, modified, same = self.dictCompare(tmp, expHeader)
        
        print("In header but not expected:")
        print(added)
        print("Expected, but not in header:")
        print(removed)
        print("Using wrong comment:")
        print(modified)

        return False 

    # https://stackoverflow.com/questions/4527942/comparing-two-dictionaries-and-checking-how-many-key-value-pairs-are-equal
    def dictCompare(self, d1, d2):
        """
        Returns:
          * whats only in d1
          * what's only in d2
          * what's in both but w/ different values
          * what's identical in both
        """
        d1_keys = set(d1.keys())
        d2_keys = set(d2.keys())
        intersect_keys = d1_keys.intersection(d2_keys)
        added = d1_keys - d2_keys
        removed = d2_keys - d1_keys
        modified = {o : (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
        same = set(o for o in intersect_keys if d1[o] == d2[o])
        return added, removed, modified, same

    def getHdus(self):    
        "Use class attributes to write smoothed data to disk as FITS file"
        # First extension just has our header
        # check to make sure header is complete
        self.checkHeader()

        # now create the FITS PHDU
        header = fits.Header()
        # tranfer data from our header to the FITS PHDU
        for k, v in self.hdr.items():
            if k == "scan_time":
                try:
                    v = (v[0].strftime("%Y-%m-%d %H:%M:%S"), "TBF")
                except:
                    v = ("ERROR", "TBF")    
            if k != 'COMMENT':    
                header[k] = v

        # additional header info
        # if "N" not in self.hdr:
            # header["N"] = self.N
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
