import unittest

import numpy as np


from zernikies import *

class TestZernikes(unittest.TestCase):
    "Test methods in lassiAnalysis that don't call gpu smoothing"

    def setUp(self):
        pass

    def testZernikePolar(self):
        "Test our equations work like we think they do"

        # TBF: this is a random spot check - seems like there
        # should be a more thourough test
            
        nzs = 37
        zs = np.zeros(nzs)

        # zero
        t = zernikePolar(zs, 0., 0.)
        self.assertEquals(t, 0.)
 
        # demo that the first one does nothing
        zs[0] = 1.
        t = zernikePolar(zs, 0., 0.)
        self.assertEquals(t, 0.)

        # piston
        zs[0] = 0.
        zs[1] = 1.
        t = zernikePolar(zs, 0., 0.)
        self.assertEquals(t, 1.)
        t = zernikePolar(zs, 1., 1.)
        self.assertEquals(t, 1.)

        # horizontal tilt?
        zs[1] = 0.
        zs[2] = 1.
        # should be zero here
        t = zernikePolar(zs, 0., 0.)
        self.assertEquals(t, 0.)

        # but not here
        t = zernikePolar(zs, 1., 1.)
        self.assertAlmostEquals(t, 0.5403023058681398, 5)

        # show that they add together
        zs[1] = 1.
        zs[2] = 1.
        t = zernikePolar(zs, 1., 1.)
        self.assertAlmostEquals(t, 1.5403023058681398, 5)

        # random check
        zs[0] = 0.
        zs[1] = 0.
        zs[19] = 1.
        t = zernikePolar(zs, 0., 0.)
        self.assertEquals(t, 0.)
        t = zernikePolar(zs, 1., 1.)
        self.assertAlmostEquals(t, 1.3817732906760363, 5)



