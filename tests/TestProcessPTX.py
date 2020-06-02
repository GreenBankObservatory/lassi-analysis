import os
import unittest
from copy import copy

import numpy as np

from processPTX import processNewPTXData 

from lassiTestSettings import SCAN9, SETTINGS_27MARCH2019
import settings


class TestProcessPTX(unittest.TestCase):


    def testProcessNewPTXDataScan9(self):

        fpath = os.path.join(settings.UNIT_TEST_PATH, '27mar2019', SCAN9)

        with open(fpath, 'r') as f:
            ls = f.readlines()

        # original number of data points (plus header)
        self.assertEqual(len(ls), 12546348)

        xOffset = SETTINGS_27MARCH2019['xOffset']
        yOffset = SETTINGS_27MARCH2019['yOffset']
        radius = SETTINGS_27MARCH2019['radius']
        rot = SETTINGS_27MARCH2019['rot']

        ellipse = [xOffset, yOffset, radius, radius, 0]

        # remove as little data as possible
        xyz, _, _ = processNewPTXData(ls,
                                      ellipse=ellipse,
                                      rot=rot,
                                      plotTest=False,
                                      nFilter=False,
                                      iFilter=False,
                                      rFilter=False,
                                      filterClose=False,
                                      filterParaboloid=False)

        # check 
        self.assertEqual(len(xyz), 10032308)

        # OK, do it again, but with the sensible filters on
        xyz, _, _ = processNewPTXData(ls,
                                      ellipse=ellipse,
                                      rot=rot,
                                      plotTest=False,
                                      nFilter=True,
                                      iFilter=False,
                                      rFilter=True,
                                      filterClose=True,
                                      filterParaboloid=True)

        #self.assertEqual(len(xyz), 9027020)
        # Since we are using a random sample to fit a parabola, 
        # the results can change.
        self.assertAlmostEqual(len(xyz), 9175594, delta=2000)
