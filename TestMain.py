import os
import unittest

import numpy as np

from main import smoothGPUs, smoothGPUMulti, smooth, getWeight
from main import importCsv, loadLeicaDataFromGpus
# from lassiTestSettings import DATA_27MARCH2019, SCAN9, SETTINGS_27MARCH2019
import settings

class TestMain(unittest.TestCase):

    def setUp(self):
        pass

    def testSmoothGPUs(self):
        "Make sure system command is configured correctly"
    
        gpuPath = 'gpuPath'
        inFile = 'inFile'
        outFile = 'outFile'
        n = 512
        test = True

        cmd = smoothGPUs(gpuPath, inFile, outFile, n, test=test)

        user = os.getlogin()
        host = settings.GPU_HOST
        exp = "runGpuSmooth gpuPath %s %s inFile outFile 512 0.00100" % (user, host)

        self.assertEqual(exp, cmd)

    def testSmoothGPUMulti(self):
        "Make sure system commands are configured correctly"

        gpuPath = 'gpuPath'
        gpuPaths = [gpuPath]*2
        inFile = 'inFile'
        inFiles = [inFile]*2
        outFile = 'outFile'
        n = 512
        test = True

        cmds = smoothGPUMulti(gpuPaths, inFiles, outFile, n, test=test)

        user = os.getlogin()
        host = settings.GPU_HOST
        host2 = settings.GPU_HOST_2
        hosts = [host, host2]

        exps = []
        for i in range(2):
            exp = [
                'runGpuParts',
                gpuPath,
                user,
                hosts[i],
                inFile,
                outFile,
                str(n),
                str(i+1),
                '2'
            ]
            exps.append(exp)

        self.assertEqual(exps, cmds)

    def testSmooth(self):
        "Test python (dask) implementation of smoothing algorithm"

        # construct some spherical data
        M = 100
        az = np.linspace(-np.pi, np.pi, M)
        el = np.linspace(-np.pi/2, np.pi/2, M)
        azm, elm = np.meshgrid(az, el)
        r = az*el

        # and smooth it to a smaller grid
        N = 10
        sigAz = sigEl = 0.1
        azLoc, elLoc, rs = smooth(az, el, r, N, sigAz=sigAz, sigEl=sigEl)

        self.assertEqual(azLoc.shape, (N,N))
        self.assertEqual(elLoc.shape, (N,N))
        self.assertEqual(rs.shape, (N,N))

        self.assertAlmostEqual(azLoc[0, 0], -3.14159265, 5)
        self.assertAlmostEqual(elLoc[0, 0], -1.57079633, 5)
        self.assertAlmostEqual(rs[0, 0], 4.51131706)

    def testGetWeight(self):
        "Test the function central to our smoothing algorithm"

        # construct some spherical data
        M = 12
        az = np.linspace(-np.pi, np.pi, M)
        el = np.linspace(-np.pi/2, np.pi/2, M)
        azm, elm = np.meshgrid(az, el)

        # construct the smoothing grid
        N = 3
        azRange = np.linspace(min(az), max(az), N)
        elRange = np.linspace(min(el), max(el), N)

        azLoc, elLoc = np.meshgrid(azRange, elRange)

        # get the weights for one point in this grid
        j = k = 1
        ws = getWeight(az, el, azLoc, elLoc, 1.0, 1.0, j, k)

        self.assertAlmostEqual(ws[0], 0.01315931, 5)
        self.assertAlmostEqual(ws[5], 5.97090021, 5)

    def testImportCSV(self):
        "Make sure we can import xyz data from CSV file"

        fn = "Test1_STA14_Bump1_High-02_METERS.ptx.csv"
        path = "/home/scratch/pmargani/LASSI/unitTestData"
        fpath = os.path.join(path, fn)

        x, y, z = importCsv(fpath)

        ln = 5915898
        self.assertEqual(len(x), ln)
        self.assertEqual(len(y), ln)
        self.assertEqual(len(z), ln)

        self.assertEqual(x[0], -1.253032129408059880e+01)
        self.assertEqual(y[0], 2.211914288349020552e+01)
        self.assertEqual(z[0], -4.575297499999999928e+01)

    def testLoadLeicaDataFromGpus(self):

        fn = "data/test"
        x, y, z = loadLeicaDataFromGpus(fn)
        
        self.assertEqual(len(x), 100)
        self.assertEqual(len(y), 100)
        self.assertEqual(len(z), 100)

        # spot check
        self.assertAlmostEqual(x[3], 1.48345411, 7)
