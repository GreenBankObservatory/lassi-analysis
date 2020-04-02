import os
import unittest



from server import getMissingFiles
import lassiTestSettings
import settings

class TestServer(unittest.TestCase):

    def setUp(self):
        pass

    def testGetMissingFiles(self):

    	fs = [
    	    "server.py",
    	    "I_dont_exist"
    	]
    	self.assertTrue(os.path.isfile(fs[0]))

    	missing = getMissingFiles(fs)

    	self.assertEquals(len(missing), 1)
    	self.assertEquals(missing,
    		              ["I_dont_exist"])