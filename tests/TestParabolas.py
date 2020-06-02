import unittest
from copy import copy

import numpy as np

from parabolas import *

class TestParabolas(unittest.TestCase):


    def setUp(self):

        # base case 
        f = 5.0
        v1x = 0.0
        v1y = 0.0
        v2 = 10.0
        xRot = np.pi/2
        yRot = 0. 
        self.data = [f, v1x, v1y, v2, xRot, yRot]

        # base tolerance
        self.tol = 1e-3


    def testFitLeicaData(self):

        # create a parabola
        focus = 1.
        v1x = 1.
        v1y = 1.
        v2 = 2.
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 10)
        xm, ym = np.meshgrid(x, y)

        z = paraboloid(xm, ym, focus, v1x, v1y, v2)

        # make our guess right on
        guess = [focus, v1x, v1y, v2, 0., 0.]
        r = fitLeicaData(xm, ym, z, guess)
        
        # and the fit should have no work to do
        self.assertEqual(list(r.x), guess)

        # now make our guess a bit off
        # guess = [focus/2., v1x/2., v1y/2., v2/2., 0., 0.]
        guess = [focus, 0., 0., 0., 0., 0.]
        r = fitLeicaData(xm, ym, z, guess)
        
        # how deterministic is this?
        self.assertAlmostEquals(r.x[0], focus, 5)
        self.assertTrue(abs(r.x[1] - v1x) < 1e-2)
        self.assertTrue(abs(r.x[2] - v1y) < 1e-2)
        self.assertAlmostEquals(r.x[3], v2, 5)
        self.assertTrue(abs(r.x[4]) < 1e-5)
        self.assertTrue(abs(r.x[5]) < 1e-5)

