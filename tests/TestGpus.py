import os
import unittest

import numpy as np

from gpus import smoothGPUs, smoothGPUMulti, loadLeicaDataFromGpus
import lassiTestSettings
import settings

class TestGpus(unittest.TestCase):

    def testSmoothGPUs(self):
        "Make sure system command is configured correctly"
    
        gpuPath = 'gpuPath'
        inFile = 'inFile'
        outFile = 'outFile'
        n = 512
        test = True

        cmd = smoothGPUs(gpuPath, inFile, outFile, n, test=test)

        user = os.getlogin()
        host = settings.GPU_HOST
        exp = "runGpuSmooth gpuPath %s %s inFile outFile 512 0.00100" % (user, host)

        self.assertEqual(exp, cmd)

    def testSmoothGPUMulti(self):
        "Make sure system commands are configured correctly"

        gpuPath = 'gpuPath'
        gpuPaths = [gpuPath]*2
        inFile = 'inFile'
        inFiles = [inFile]*2
        outFile = 'outFile'
        n = 512
        test = True

        cmds = smoothGPUMulti(gpuPaths, inFiles, outFile, n, test=test)

        user = os.getlogin()
        host = settings.GPU_HOST
        host2 = settings.GPU_HOST_2
        hosts = [host, host2]

        exps = []
        for i in range(2):
            exp = [
                'runGpuParts',
                gpuPath,
                user,
                hosts[i],
                inFile,
                outFile,
                str(n),
                str(i+1),
                '2'
            ]
            exps.append(exp)

        self.assertEqual(exps, cmds)

    def testLoadLeicaDataFromGpus(self):

        fn = "data/test"
        x, y, z = loadLeicaDataFromGpus(fn)
        
        self.assertEqual(len(x), 100)
        self.assertEqual(len(y), 100)
        self.assertEqual(len(z), 100)

        # spot check
        self.assertAlmostEqual(x[3], 1.48345411, 7)            
