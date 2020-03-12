import os
import unittest
import numpy as np

from grid import regridXYZ
from plotting import scatter3dPlot

class TestGrid(unittest.TestCase):

    def setUp(self):

        # plots are good for debugging - run from notebook
        # like this:
        # import unittest
        # from TestLassiAnalysis import TestLassiAnalysis
        # unittest.main(argv=[''], verbosity=2, exit=False)
        self.plots = False

    def testRegridXYZ(self):
        "Make sure resampling happens sensibly"

        # construct some cartesian data
        M = 100
        width = 10.
        maxMin = width/2.
        x = np.linspace(-maxMin, maxMin, M)
        y = np.linspace(-maxMin, maxMin, M)
        xm, ym = np.meshgrid(x, y)
        z = xm * ym

        if self.plots:
            scatter3dPlot(xm, ym, z, "org")

        N = 10
        x2, y2, z2 = regridXYZ(xm, ym, z, n=N)

        self.assertEquals(x2.shape, (N, N))
        self.assertEquals(y2.shape, (N, N))
        self.assertEquals(z2.shape, (N, N))

        # check the x, y grid - based of min and maxs
        self.assertEquals(x2[0,0], xm[0,0])
        self.assertEquals(y2[0,0], ym[0,0])

        # spot check the regridded data
        self.assertEquals(z[0, 0], 25.0)

        # make sure the residuals are low
        for i in range(N):
            for j in range(N):
                # print i, j, x2[i, j], y2[i, j], np.abs((x2[i,j] * y2[i,j]) - z2[i, j])
                diff = np.abs((x2[i,j] * y2[i,j]) - z2[i, j])
                self.assertTrue(diff < 1e-2)

        if self.plots:
            scatter3dPlot(x2, y2, z2, "regrid")

        # now, how do these max values work?
        xmin = ymin = -1.0
        x2, y2, z2 = regridXYZ(xm, ym, z, n=N, xmin=xmin, ymin=ymin)

        if self.plots:
            scatter3dPlot(x2, y2, z2, "regrid")

        self.assertEquals(x2.shape, (N, N))
        self.assertEquals(y2.shape, (N, N))
        self.assertEquals(z2.shape, (N, N))

        # check the x, y grid 
        self.assertEquals(x2[0,0], xmin)
        self.assertEquals(y2[0,0], ymin)
