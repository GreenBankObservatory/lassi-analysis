"Unit tests for analyzeActiveSurface module"

import unittest

from gbtdata.analyzeActiveSurface import analyzeActiveSurfaceScans

class TestAnalyzeActiveSurface(unittest.TestCase):
    "Unit tests for module that examines Active Surface FITS files"

    def testAnalyzeActiveSurfaceScans(self):
        "test highest level function for examing FITS files"

        # here, we're really just making sure we can make the
        # call and nothing blows up - most of the functionality
        # is for making plots that we check by hand.
        scans = [2, 3]
        # path = "/users/pmargani/tmp/lassi-analysis/simdata/TINT_080219/"
        path = "data/TINT_080219/"
        d = analyzeActiveSurfaceScans(path, scans, details=True, test=True)

        self.assertEqual(len(d), len(scans))
