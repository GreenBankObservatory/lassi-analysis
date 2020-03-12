import unittest
import numpy as np

from zernikies import zernikePolar #, getZernikeCoeffs

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

    # We no longer use opticspy
    # TBD: write a test for the methods we use now.
#    def TestGetZernikeCoeffs(self):
#        "Make sure opticspy zernike surfaces can be identified"
#        
#        # noll index of 11
#        zAmp = 1.0
#        Z = opticspy.zernike.Coefficient(Z11=zAmp)
#
#        surf = Z.zernikematrix()
#
#        order = 36
#        zs = getZernikeCoeffs(surf, order)
#
#        # shows up as expected asAnsi of 13?
#        expIdx = 13
#        tol = 5e-2
#        for i in range(order + 1):
#            if i != expIdx:
#                self.assertTrue(abs(zs[i]) < tol )
#            else:
#                self.assertTrue(abs(zAmp - zs[i]) < 0.1)    
#
#
#        # try noll index 3
#        Z = opticspy.zernike.Coefficient(Z3=zAmp)
#
#        surf = Z.zernikematrix()
#
#        zs = getZernikeCoeffs(surf, order)
#
#        # shows up as expected asAnsi of 3?
#        expIdx = 3
#        tol = 5e-2
#        for i in range(order + 1):
#            if i != expIdx:
#                self.assertTrue(abs(zs[i]) < tol )
#            else:
#                self.assertTrue(abs(zAmp - zs[i]) < 0.1)    
