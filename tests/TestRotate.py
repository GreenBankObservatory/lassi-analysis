import unittest

import numpy as np

from rotate import shiftRotateXYZ


class TestRotate(unittest.TestCase):


    def setUp(self):

        n = 10
        self.x = np.zeros(n, dtype=np.float)
        self.y = np.zeros(n, dtype=np.float)
        self.z = np.zeros(n, dtype=np.float)
        self.xyz = np.c_[self.x,self.y,self.z]


    def testShiftRotateXYZ0(self):
        """
        No rotation nor shift.
        """
        
        rot = [0,0,0]
        shift = [0,0,0]

        x = self.x+1.
        y = self.y+1.
        z = self.z

        xr, yr, zr = shiftRotateXYZ(x, y, z, shift+rot)

        np.testing.assert_array_equal(xr, x)
        np.testing.assert_array_equal(yr, y)
        np.testing.assert_array_equal(zr, z)


    def testShiftRotate90XYZ(self):
        """
        Rotation along x by 90 degrees.
        """

        rot = [np.deg2rad(90),0,0]
        shift = [0,0,0]

        x = self.x
        y = self.y+1.
        z = self.z

        xr, yr, zr = shiftRotateXYZ(x, y, z, shift+rot)

        np.testing.assert_allclose(xr, x, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(yr, z, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(zr, y, rtol=1e-15, atol=1e-15)


    def testShiftRotateX90YZ(self):
        """
        Rotation along y by -90 degrees.
        """

        rot = [0,np.deg2rad(-90),0]
        shift = [0,0,0]

        x = self.x+1.
        y = self.y
        z = self.z

        xr, yr, zr = shiftRotateXYZ(x, y, z, shift+rot)

        np.testing.assert_allclose(xr, z, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(yr, y, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(zr, x, rtol=1e-15, atol=1e-15)


    def testShiftRotateXY90Z(self):
        """
        Rotation along z by 90 degrees.
        """

        rot = [0,0,np.deg2rad(90)]
        shift = [0,0,0]

        x = self.x+1.
        y = self.y
        z = self.z

        xr, yr, zr = shiftRotateXYZ(x, y, z, shift+rot)

        np.testing.assert_allclose(xr, y, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(yr, x, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(zr, z, rtol=1e-15, atol=1e-15)


    def testShiftRotateXYZ10(self):
        """
        Shift by 10 in all directions.
        """

        rot = [0,0,0]
        shift = [10,10,10]

        x = self.x - 10
        y = self.y - 10
        z = self.z - 10

        xr, yr, zr = shiftRotateXYZ(x, y, z, shift+rot)

        np.testing.assert_allclose(xr, self.x, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(yr, self.y, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(zr, self.z, rtol=1e-15, atol=1e-15)
