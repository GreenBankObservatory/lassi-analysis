import os
import csv
import unittest
from copy import copy

import numpy as np

import matplotlib
matplotlib.use('agg')

#from parabolas import *
from processPTX import * 

class TestProcessPTX(unittest.TestCase):

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
        self.places = 3

    def almostEqualMatrix(self, exp, actual, places=None):

        if places is None:
            places = self.places

        # TBF: better way to do this?
        self.assertEqual(actual.shape, exp.shape)

        w, h = exp.shape
        for i in range(w):
            for j in range(h):
                # TBF: why does this not work?
                #print i, j, exp[i][j], actual[i][j], type(exp[i][j]), type(actual[i][j])
                #self.assertAlmostEquals(exp[i][j], actual[i][j], tol)
                second = exp[i][j]
                first = actual[i][j]
                self.assertTrue(round(abs(second-first), places) == 0)

    def testRotateXYaboutZ_0degrees(self):
        "data not rotated by roateXYaboutZ is equal to itself"

        # data in the xyz direction
        xyz = np.array([(i, 0, 0) for i in range(10)])

        # identity test
        newXyz = rotateXYaboutZ(xyz, 0.)

        self.almostEqualMatrix(xyz, newXyz)

    def testRotateXYaboutZ_90degrees(self):
        "x data rotated 90 degrees by roateXYaboutZ becomes negative y data"

        # data in the xyz direction
        xyz = np.array([(i, 0, 0) for i in range(10)])

        newXyz = rotateXYaboutZ(xyz, 90.)

        exp = np.array([(0., float(i), 0.) for i in range(10)])

        self.almostEqualMatrix(exp, newXyz)

    def testRotateXYaboutZ_180degrees(self):
        "x data rotated 180 degrees by roateXYaboutZ becomes negative x data"

        # data in the xyz direction
        xyz = np.array([(i, 0, 0) for i in range(10)])

        newXyz = rotateXYaboutZ(xyz, 180.)

        exp = np.array([(-float(i), 0., 0.) for i in range(10)])

        self.almostEqualMatrix(exp, newXyz)

    def testFitlerOutRadius(self):
        "make sure our filtering by radius works"

        xyz = [
            (0,0,0),
            (1,1,0),
            (1,2,0),
            (2,2,0)
        ]

        # make the radius big enough to take in everything
        r = filterOutRadius(xyz, searchRadius=4, mysteryX=0)
        self.assertTrue((r == xyz).all()) 

        # now shrink the radius
        exp = xyz[:2]
        r = filterOutRadius(xyz, searchRadius=2, mysteryX=0)
        self.assertTrue((r == exp).all()) 

        # increase radius, but add extension in x
        exp = np.array(xyz[:3])
        r = filterOutRadius(xyz, searchRadius=4, mysteryX=2)
        self.assertTrue((r == exp).all()) 

    def testProcessPTXdata(self):
        "process data with no filters"

        hdr = ['\n']*10
        ls = [
            '0 0 0 0.500000\n',
            '0 0 0 0.500000\n',
            '26.277939 -18.431900 -44.699600 0.031861\n',
            '26.302536 -18.449173 -44.682480 0.252079\n',
        ]

        # don't rotate the data, and don't filter by radius
        rs = processPTXdata(hdr + ls, 0., 1e10)

        # exp answer means the data doesn't change
        exp = np.array([
            [26.277939, -18.4319,   -44.6996],
            [26.302536, -18.449173, -44.68248]
        ])

        self.almostEqualMatrix(exp, rs)

    def testProcessPTXdata2(self):
        "process data with standard filters"

        hdr = ['\n']*10
        ls = [
            '0 0 0 0.500000\n',
            '0 0 0 0.500000\n',
            '26.277939 -18.431900 -44.699600 0.031861\n',
            '26.302536 -18.449173 -44.682480 0.252079\n',
        ]

        # process with values we've seen in the wild 
        rs = processPTXdata(hdr + ls, 150., 1e10) #47)

        # exp answer 
        exp = np.array([
            [-13.54141273, 29.10146314, -44.6996],
            [-13.55407786, 29.1287205, -44.68248]
        ])

        self.almostEqualMatrix(exp, rs)
        
        # we see that when we filter by the usual radius, we filter out all our test data
        rs = processPTXdata(hdr + ls, 150., 47)

        self.assertEqual(len(rs), 0)

        # now add some data that won't get filtered out
        line = '54 0 -44.682480 0.252079\n'
        newLs = hdr + ls + [line]

        rs = processPTXdata(newLs, 150., 47)

        self.assertEqual(len(rs), 1)
        exp = np.array([-46.76537180435968,27.,-44.68248])
        exp.shape = (1,3)
        self.almostEqualMatrix(exp, rs)
 
    def testProcessPTX2(self):
        "test high level processing of simple sample file with standard filters"

        fn = "Test1_STA14_Bump1_High-02_METERS_SAMPLE.ptx"
        fpath = os.path.join("data", fn)

        # make sure input exists
        self.assertTrue(os.path.isfile(fpath))

        opath = fpath + ".csv"

        # make sure the expected output does NOT exist yet
        if os.path.isfile(opath):
            os.remove(opath)
        self.assertTrue(not os.path.isfile(opath))

        # now create the file we just might have cleaned up
        processPTX(fpath, rotationAboutZdegrees=150., searchRadius=47)

        # make sure output exists
        self.assertTrue(os.path.isfile(opath))

        # now parse it
        xyzs = []
        fieldnames = ['x', 'y', 'z']
        with open(opath, 'r') as f:
            reader = csv.DictReader(f, fieldnames=fieldnames)
            for row in reader:
                 xyz = (float(row['x']),
                 float(row['y']),
                 float(row['z']))
                 xyzs.append(xyz)
        
        exp = [
            (-46.76537180435968, 27., -44.68248)
        ]

        self.almostEqualMatrix(np.array(xyzs), np.array(exp))


    def testProcessPTX(self):
        "Test high level processing function against a simple sample and no filters"

        fn = "Test1_STA14_Bump1_High-02_METERS_SAMPLE.ptx"
        fpath = os.path.join("data", fn)

        # make sure input exists
        self.assertTrue(os.path.isfile(fpath))

        opath = fpath + ".csv"

        # make sure the expected output does NOT exist yet
        if os.path.isfile(opath):
            os.remove(opath)
        self.assertTrue(not os.path.isfile(opath))

        # now create the file we just might have cleaned up
        processPTX(fpath, rotationAboutZdegrees=0, searchRadius=1e10)

        # make sure output exists
        self.assertTrue(os.path.isfile(opath))

        # now parse it
        xyzs = []
        fieldnames = ['x', 'y', 'z']
        with open(opath, 'r') as f:
            reader = csv.DictReader(f, fieldnames=fieldnames)
            for row in reader:
                 xyz = (float(row['x']),
                 float(row['y']),
                 float(row['z']))
                 xyzs.append(xyz)
        
        exp = [
            (26.456772, -18.582199, -44.610275),
            (26.486862, -18.603317, -44.60231),
            (26.519547, -18.626236, -44.598648),
            (26.54744, -18.645828, -44.586563),
            (26.573868, -18.664383, -44.572403),
            (26.277939, -18.4319, -44.6996),
            (26.302536, -18.449173, -44.68248),
            (54.0, 0.0, -44.68248)
        ]

        self.almostEqualMatrix(np.array(xyzs), np.array(exp))


    def readCsvFile(self, fpath):
        "read CSV file produced by Mathematica and processPTX"
        xyzs = []
        fieldnames = ['x', 'y', 'z']
        with open(fpath, 'r') as f:
            reader = csv.DictReader(f, fieldnames=fieldnames)
            for row in reader:
                 xyz = (float(row['x']),
                 float(row['y']),
                 float(row['z']))
                 xyzs.append(xyz)
        return xyzs

    def testProcessPTX3(self):
        "tests our processing of 500K lines of input against Mathematica output"

        fn = "Test1_STA14_Bump1_High-02_METERS_SAMPLES_500000.ptx"
        fpath = os.path.join("data", fn)

        # make sure input exists
        self.assertTrue(os.path.isfile(fpath))

        opath = fpath + ".csv"

        # make sure the expected output does NOT exist yet
        if os.path.isfile(opath):
            os.remove(opath)
        self.assertTrue(not os.path.isfile(opath))

        # now create the file we just might have cleaned up
        processPTX(fpath)

        # make sure output exists
        self.assertTrue(os.path.isfile(opath))

        actual = np.array(self.readCsvFile(opath))

        # here's the file actually produced by Mathematica!
        expPath = opath + ".exp"
        exp = np.array(self.readCsvFile(expPath))

        self.almostEqualMatrix(actual, exp)

