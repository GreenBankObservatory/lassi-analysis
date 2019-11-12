import os
import unittest

import numpy as np

from gpus import smoothGPUs, smoothGPUMulti, smoothGPUMultiFile, loadLeicaDataFromGpus
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


    def testSmoothGPUMultiFile(self):
        "Make sure system commands are configured correctly for one file"

        numGpus = 8
        gpuPath = 'gpuPath'
        gpuPaths = [gpuPath]*numGpus
        host = settings.GPU_HOST
        # host2 = settings.GPU_HOST_2
        # hosts = [host, host2]
        hosts = [host] * numGpus
        inFile = 'inFile'
        # inFile = "/users/pmargani/tmp/Clean9.ptx.csv"
        # inFiles = [inFile]*2
        # how do we expect this inFile to be used?
        inFiles = [inFile]
        for i in range(2, numGpus + 1):
            inFiles.append(inFile + ('.%d.csv' % i))
        outFile = 'outFile'
        n = 512
        test = True

        cmds = smoothGPUMultiFile(gpuPaths, hosts, inFile, outFile, n, test=test)

        user = os.getlogin()

        exps = []
        for i in range(numGpus):
            exp = [
                'runGpuParts',
                gpuPaths[i],
                user,
                hosts[i],
                inFiles[i],
                outFile,
                str(n),
                str(i + 1),
                str(numGpus)
            ]
            exps.append(exp)

        self.assertEqual(exps, cmds)


    def testSmoothGPUMulti(self):
        "Make sure system commands are configured correctly"

        gpuPath = 'gpuPath'
        gpuPaths = [gpuPath]*2
        host = settings.GPU_HOST
        host2 = settings.GPU_HOST_2
        hosts = [host, host2]
        inFile = 'inFile'
        inFiles = [inFile]*2
        outFile = 'outFile'
        n = 512
        test = True

        cmds = smoothGPUMulti(gpuPaths, hosts, inFiles, outFile, n, test=test)

        user = os.getlogin()

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
