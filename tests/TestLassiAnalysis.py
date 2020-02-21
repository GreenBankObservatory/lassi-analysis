import os
import unittest

import numpy as np

from plotting import scatter3dPlot

from lassiAnalysis import maskLeicaData
from lassiAnalysis import extractZernikesLeicaScanPair
import lassiTestSettings

class TestLassiAnalysis(unittest.TestCase):
    "Test methods in lassiAnalysis that don't call gpu smoothing"


    def setUp(self):

        # plots are good for debugging - run from notebook
        # like this:
        # import unittest
        # from TestLassiAnalysis import TestLassiAnalysis
        # unittest.main(argv=[''], verbosity=2, exit=False)
        self.plots = False


    def testMaskLeicaData(self):
        "Test that we can mask out bumps in a smoothed data"

        # TBF: where to store test data.
        # fn = "data/27mar2019/Clean9.ptx.csv"
        # fn = "Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx.csv"
        fn = lassiTestSettings.SCAN9 + ".csv"
        # path = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing"
        path = lassiTestSettings.DATA_UNIT_TESTS
        fpath = os.path.join(path, '27mar2019/gpus', fn)

        maskGuess=[60., 0., 0., -50., 0., 0.]
        d = maskLeicaData(fpath, radialMask=True, guess=maskGuess)

        origData = d['origData']
        origMaskedData = d['origMasked']
        rotatedData = d['rotated']
        fitResidual = d['fitResidual']
        parabolaFit = d['parabolaFit']
        fitCoeffs = d['parabolaFitCoeffs']

        expCoeffs = [
            6.00000000e+01,
            2.45124145e+00,
            -8.79388591e-01,
            -4.92666730e+01,
            1.94525728e-02,
            -2.86261922e-04
        ]
        for exp, act in zip(expCoeffs, fitCoeffs):
            self.assertAlmostEquals(exp, act, 3)

        N = 512
        orgX = origData[0]
        self.assertEquals(orgX.shape, (N, N))

        orgMX = origMaskedData[0]
        self.assertEquals(orgMX.shape, (N, N))

        mask = orgMX.mask
        self.assertEquals(mask.shape, (N, N))

        fm = mask[mask == False]
        tm = mask[mask == True]
        
        fmLen = 143061
        tmLen = 119083

        # sanity check our expected values
        self.assertEquals(fmLen + tmLen, N*N)

        self.assertEquals(fm.shape, (fmLen,))
        self.assertEquals(tm.shape, (tmLen,))
        self.assertEqual(fitResidual.shape, (N,N))

        mask = fitResidual.mask
        self.assertEquals(mask.shape, (N, N))

        fm = mask[mask == False]
        tm = mask[mask == True]
        
        self.assertEquals(fm.shape, (fmLen,))
        self.assertEquals(tm.shape, (tmLen,))
        
            
    def testExtractZernikesLeicaScanPair(self):

        # fn1 = "Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx.csv"
        # fn2 = "Scan-11_5100x5028_20190327_1155_ReIoNo_ReMxNo_ColorNo_.ptx.csv"
        fn1 = lassiTestSettings.SCAN9 + ".csv"
        fn2 = lassiTestSettings.SCAN11 + ".csv"
        # path = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing"
        path = lassiTestSettings.DATA_UNIT_TESTS
        fpath1 = os.path.join(path, '27mar2019/gpus', fn1)
        fpath2 = os.path.join(path, '27mar2019/gpus', fn2)

        N = 512
        nZern = 36
        x, y, diff, fitlist = extractZernikesLeicaScanPair(fpath1,
                                                           fpath2,
                                                           n=N,
                                                           nZern=nZern)

        # TBF: all the diffs data looks masked. how to check that?
        
        self.assertEquals(diff.shape, (N, N))
        self.assertEquals(len(fitlist), nZern + 1)
        self.assertEquals(fitlist, [0.0]*(nZern + 1))
