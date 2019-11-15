import unittest
from copy import copy

import numpy as np

# import matplotlib
# matplotlib.use('agg')

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

    #def checkFit(self, answer, guess, fit, diff, tol=None):
    def checkFit(self, diff, tol=None):

        if tol is None:
            tol = self.tol

        for i, d in enumerate(diff):
            #print i, d, tol, d < tol
            if d >= tol:
                print("Fail: ")
                print(i, d, tol, d < tol)
             
            self.assertTrue(d < tol)

    def testFit1(self):
        "base test - guessing the right answer exactly gives us exact answer!" 
        answer, guess, fit, diff = tryFit(self.data)
        #print diff
        self.checkFit(diff, tol=1e-10)

    def testFitBadYRotations(self):
        "base test - but increase y rotation from nothing"
        data = copy(self.data)

        # increase Y rotation from nothing
        data[5] = 0.1
        answer, guess, fit, diff = tryFit(data)
        #print fit, diff
        # everything gets fit perfectly
        self.checkFit(diff[:-1], tol=1e-10)

        # except the y rotation! which is totally off!
        # y-rotation fit to almost nothing
        self.assertTrue(fit[5] < 1e-10)

        # now rotate by Y more
        data[5] = np.pi/10 
        answer, guess, fit, diff = tryFit(data)
        #print fit, diff
        # everything gets fit perfectly
        self.checkFit(diff[:-1], tol=1e-10)

        # except the y rotation! which is totally off!
        # y-rotation fit to almost nothing
        self.assertTrue(fit[5] < 1e-10)

        # and rotate around Y a lot
        # data[5] = np.pi/2 
        # answer, guess, fit, diff = tryFit(data)
        #print fit, diff
        # the fit goes to hell.  the focus isn't even close
        # print fit[0]
        # self.assertTrue(fit[0] > 1e4)
        # print fit
        # print diff
        # self.checkFit(diff[:-1], tol=1e-10)

    def testFitGoodRotations(self):
        "base test - but with small X and Y rotations"

        data = copy(self.data)
        data[4] = 0.1
        data[5] = 0.1
        answer, guess, fit, diff = tryFit(data)
        #print fit, diff
        self.checkFit(diff)

        # increasing the angle still works
        data[4] = np.pi/10 
        data[5] = np.pi/10
        answer, guess, fit, diff = tryFit(data)
        #print fit, diff
        # but we really have to raise our error tolerance
        self.checkFit(diff, tol=0.1)

    def testFitGuesses(self):
        "base test - but don't guess the answer right off the bat"

        # cheat the first time
        data = copy(self.data)
        data[4] = np.pi/10 
        data[5] = np.pi/10
        guess = copy(data)
        answer, guess, fit, diff = tryFit(data, guess)
        #print fit, diff
        # exact fits for non rotations
        self.checkFit(diff[:4], tol=1e-10)
        # and pretty close for the rotations
        self.checkFit(diff[4:], tol=1e-1)

        # Okay, now really guess
        # guess = [1., 0., 0., 0., 0., 0.]
        # answer, guess, fit, diff = tryFit(data, guess)
        #print fit, diff
        # exact fits for non rotations
        # self.checkFit(diff[:4], tol=1e-10)
        # and pretty close for the rotations
        # self.checkFit(diff[4:], tol=1e-1)

    def testFitLeicaData(self):

        # create a parabola
        focus = 1.
        v1x = 1.
        v1y = 1.
        v2 = 2.
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 10)
        xm, ym = np.meshgrid(x, y)

        z = parabola(xm, ym, focus, v1x, v1y, v2)

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

    # TBF: this seems to hang!
    # def testFitLeicaScan(self):

    #     fn = "./data/27mar2019/Clean9.ptx.csv"

    #     diff, x, y = fitLeicaScan(fn,
    #                               numpy=False,
    #                               N=512,
    #                               plot=False)

    #     print diff.shape