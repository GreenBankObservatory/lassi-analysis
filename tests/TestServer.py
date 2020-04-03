import os
import unittest


from processPTX import getRawXYZ
from server import getMissingFiles, getGpuOutputPaths
from server import processLeicaDataStream
import lassiTestSettings
import settings

class TestServer(unittest.TestCase):

    def setUp(self):
        pass

    def testGetMissingFiles(self):

    	fs = [
    	    "server.py",
    	    "I_dont_exist"
    	]
    	self.assertTrue(os.path.isfile(fs[0]))

    	missing = getMissingFiles(fs)

    	self.assertEquals(len(missing), 1)
    	self.assertEquals(missing,
    		              ["I_dont_exist"])

    def testGetGpuOutputPaths(self):

        paths = ["/path/one/", "/path/two"]
        outfile = "outfile"
        exp = ["/path/one/outfile.1", "/path/two/outfile.2"]
        opaths = getGpuOutputPaths(paths, outfile)
        self.assertEquals(exp, opaths)

    def testProcessLeicaDataStream(self):

        # write locally
        dataDir = './data'
        project = "UNIT_TEST_PROJ"
        filename = "filename"

        # before we prepare, clean up from last time?
        exts = ['.processed.png', '.smoothed.fits']
        expFiles = []
        for ext in exts:
            fn = filename + ext
            path = os.path.join(dataDir, project, 'LASSI', fn)
            expFiles.append(path)
            if os.path.isfile(path):
                # clean up!
                print("Removing old file", path)
                os.remove(path)
        for fn in expFiles:
            self.assertTrue(not os.path.isfile(fn))

        # get raw data from file
        path = settings.UNIT_TEST_PATH
        scan9 = lassiTestSettings.SCAN9
        fn = os.path.join(path, '27mar2019', scan9)
        with open(fn, 'r') as f:
            ls = f.readlines()
        x, y, z, i = getRawXYZ(ls)

        # TBF: remove these if we arent using them
        hdr = {}
        dts = None

        # use processing from settings
        s = lassiTestSettings.SETTINGS_27MARCH2019
        ellipse, rot = lassiTestSettings.getData(s)

        # we can't smooth in tests cause we don't have GPUs,
        # so what canned smooth data to use?
        fn = scan9 + '.csv'
        smoothOutputs = [os.path.join(path, '27mar2019/gpus', fn)]


        processLeicaDataStream(x, y, z, i, dts, hdr,
            ellipse, rot, project, dataDir, filename, 
            plot=True, test=True, smoothOutputs=smoothOutputs)

        # finally, if this function completed, it should have
        # created these files
        for fn in expFiles:
            self.assertTrue(os.path.isfile(fn))
