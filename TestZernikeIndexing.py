import unittest

from zernikeIndexing import printZs, ansi2activeAnsi, noll2ansi
from zernikeIndexing import nollZs, asAnsiZs

class TestZernikes(unittest.TestCase):
    "Test methods for converting between different zernike ordering conventions"
    
    def setUp(self):
        self.maxZ = 36

    def testNoll2Ansi(self):
        "Test that we can convert from noll to ansi"
        noll = list(range(1,self.maxZ + 1))
        ansi = noll2ansi(noll)
        # print("ansi: ")
        # printZs(ansi)
        # print("noll: ")
        # printZs(noll)
        self.assertEqual(ansi, nollZs)


    def testAnsi2ActiveAnsi(self):
        zs = list(range(0, self.maxZ))
        zs2 = ansi2activeAnsi(zs)
        # print("ansi: ")
        # printZs(zs)
        # print("active ansi: ")
        # printZs(zs2)
        # print(len(zs), len(zs2))
        self.assertEqual(zs2, asAnsiZs)
