"Unit tests for Utils module"

import unittest

import numpy as np

# import matplotlib
# matplotlib.use('agg')

from utils import sph2cart, cart2sph

class TestUtils(unittest.TestCase):
    "unit tests for Utils module"

    # def setUp(self):

    def testCoordinateIdentity(self):
        "test sph2cart -> cart2sph -> original values"

        # create some test data
        xs = np.array(range(10))
        ys = np.array(range(10)) * -1.0
        zs = (xs**2 + ys**2) + 10.

        # transform it - there and back again
        # el == latitude
        # az == longitude
        r, el, az = cart2sph(xs, ys, zs, verbose=False)
        x2, y2, z2 = sph2cart(el, az, r, verbose=False)

        tol = 1e-6

        self.assertTrue(np.all(np.abs(x2-xs) < tol))
        self.assertTrue(np.all(np.abs(y2-ys) < tol))
        self.assertTrue(np.all(np.abs(z2-zs) < tol))
