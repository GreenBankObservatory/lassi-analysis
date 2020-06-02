import os
import unittest

import numpy as np

from plotting import scatter3dPlot

from lassiAnalysis import maskScan
from lassiAnalysis import maskLeicaData
from lassiAnalysis import extractZernikesLeicaScanPair
import lassiTestSettings
import settings

class TestLassiAnalysis(unittest.TestCase):
    "Test methods in lassiAnalysis that don't call gpu smoothing"


    def setUp(self):

        # plots are good for debugging - run from notebook
        # like this:
        # import unittest
        # from TestLassiAnalysis import TestLassiAnalysis
        # unittest.main(argv=[''], verbosity=2, exit=False)
        self.plots = False

    
    def testMaskScan(self):
        """
        """

        expected = np.array([[False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False,  True, False, False, False, False],
                             [False, False, False,  True,  True, False, False, False],
                             [False, False, False,  True,  True, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False]])

        n = 512
        fn = lassiTestSettings.SCAN9 + ".csv"
        path = settings.UNIT_TEST_PATH
        fpath = os.path.join(path, '27mar2019/gpus', fn)

        xr, yr, zr = maskScan(fpath, n=n, rot=0)

        np.testing.assert_array_equal(zr[n//2-4:n//2+4,n//2-4:n//2+4].mask, expected)
        
            
    def testExtractZernikesLeicaScanPair(self):

        fn1 = lassiTestSettings.SCAN9 + ".csv"
        fn2 = lassiTestSettings.SCAN11 + ".csv"
        path = settings.UNIT_TEST_PATH
        fpath1 = os.path.join(path, '27mar2019/gpus', fn1)
        fpath2 = os.path.join(path, '27mar2019/gpus', fn2)

        expected  = np.array([ 0.00000000e+00,  3.62084426e-06, -2.09326848e-06,  2.00161270e-05,
                              -1.95328477e-03, -7.55799017e-04, -7.26745615e-04,  1.39954039e-04,
                               6.76198828e-06,  3.42735429e-06, -2.37041694e-04,  3.73330598e-05,
                               7.46538852e-04,  2.31084957e-04,  3.07554384e-04, -8.47019848e-06,
                               1.78253932e-04,  3.18025410e-05, -7.75638977e-05, -1.82476190e-06,
                              -6.44858533e-05, -1.98739178e-04,  2.92033128e-05,  3.04666280e-05,
                               6.17212713e-04,  2.04950281e-04,  2.63346578e-04,  1.08807675e-04,
                               2.68525819e-05,  6.83855791e-05, -1.13836213e-04, -7.44452014e-05,
                               1.96018181e-05,  4.85993401e-05,  1.09020282e-05,  3.67054789e-05,
                              -8.91468101e-05])

        N = 512
        nZern = 36
        x, y, diff, fitlist = extractZernikesLeicaScanPair(fpath1,
                                                           fpath2,
                                                           n=N,
                                                           nZern=nZern)

        
        self.assertEquals(diff.shape, (N, N))
        self.assertEquals(len(fitlist), nZern + 1)
        np.testing.assert_allclose(fitlist, expected, rtol=1e-05, atol=2e-5)        
