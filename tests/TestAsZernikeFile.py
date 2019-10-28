"Unit tests for Utils module"

import unittest

from AsZernikeFile import AsZernikeFile

class TestAsZernikeFile(unittest.TestCase):
    "Unit tests for code that parses AsZernike.conf file"

    def testParse(self):
        "Test that we can parse the .conf file correctly"
        
        asz = AsZernikeFile("data/AsZernike.conf")
        asz.parse()
        #for act in asz.actuatorList:
        #    print act
        ks = sorted(asz.actuators.keys())
        # for k in ks:
        #     print(k, asz.actuators[k].phi)
        # print("Found %d actuators specified" % len(asz.actuatorList))

        # this is how many actuators we have
        self.assertEqual(len(ks), 2209)

        # and they are indicated with a hoop and rib system
        hoops = set([k[0] for k in ks])
        ribs = set([k[1] for k in ks])
 
        self.assertEqual(list(hoops), list(range(1, 46)))
        
        # the ribs are not as predictable
        self.assertEqual(min(list(ribs)), -680)
        self.assertEqual(max(list(ribs)), 680)

        phis = [asz.actuators[k].phi for k in ks]

        self.assertEqual(min(phis), 1.00056)
        self.assertEqual(max(phis), 1.32331)
