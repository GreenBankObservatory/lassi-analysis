import os
import unittest

import numpy as np

from SmoothedFITS import SmoothedFITS

class TestSmoothedFITS(unittest.TestCase):

    
    def testCompareDicts(self):

        fits = SmoothedFITS()

        d1 = {}
        d2 = {}

        self.assertEquals((set(), set(), {}, set()), 
                           fits.dictCompare(d1, d2))

        d1 = {'a': 1}

        self.assertEquals((set('a'), set(), {}, set()), 
                   fits.dictCompare(d1, d2))

        
        d2 = {'b': 2}

        self.assertEquals((set('a'), set('b'), {}, set()), 
                   fits.dictCompare(d1, d2))

        # add 'c' to both equal
        d1['c'] = 3
        d2['c'] = 3

        self.assertEquals((set('a'), set('b'), {}, set('c')), 
                   fits.dictCompare(d1, d2))
        
        # add 'd' to both NOT equal
        d1['d'] = 4
        d2['d'] = 5

        self.assertEquals((set('a'), set('b'), {'d': (4,5)}, set('c')), 
                   fits.dictCompare(d1, d2))

    def testCheckHeader(self):        

        fits = SmoothedFITS()

        # no header is obviously a problem
        self.assertEquals(False, fits.checkHeader())

        # prepare the right header:
        fits.hdr = {}
        for k, v in fits.expHeader.items():
            fits.hdr[k] = (None, v)

        # and this should be NO problem
        self.assertEquals(True, fits.checkHeader())

        # but change one thing, and it is:
        fits.hdr["SCAN"] = (None, "change comment")
        self.assertEquals(False, fits.checkHeader())

    def testWrite(self):
    
        fits = SmoothedFITS()

        # prepare the expected header:
        hdr = {}
        for k, v in fits.expHeader.items():
            hdr[k] = ("", v)

        x = np.array([1.0])
        y = np.array([2.0])
        z = np.array([3.0])

        N = 512

        scanNum = 1

        # file path
        proj = 'UNIT_TEST_PROJ'
        dataDir = './data'
        filename = 'testWrite'    
        filepath = os.path.join(dataDir, proj, 'LASSI', filename)
        filepath += '.smoothed.fits'

        # make sure this doesn't exist now
        if os.path.isfile(filepath):
            os.remove(filepath)
            self.assertTrue(not os.path.isfile(filepath))

        # finally write all this to disk
        fits.setData(x, y, z, N, hdr, dataDir, proj, filename, scanNum)
        fits.write()

        # make sure it got written
        self.assertTrue(os.path.isfile(filepath))
        
        # Identity Test
        # now verify contents by reading it back in
        reader = SmoothedFITS()

        reader.read(filepath)

        self.assertEquals(x, reader.x)
        self.assertEquals(y, reader.y)
        self.assertEquals(z, reader.z)

        # convert the astropy.io.fits header
        # to our expected dictionary
        header = {}
        for k, v in reader.hdr.items():
            header[k] = (v, reader.hdr.comments[k])
        reader.hdr = header    
        hdrOK = reader.checkHeader(addStandardKeys=True)
        self.assertTrue(hdrOK)