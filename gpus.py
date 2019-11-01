import csv
import os
import subprocess
from copy import copy

import numpy as np

import settings

GPU_PATH = settings.GPU_PATH

def trySmoothGPUMulti(N=10):

    if N is None:
        N = 10

    gpuPath1 = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpuMulti"
    gpuPath2 = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpuMulti2"
    gpuPaths = [gpuPath1, gpuPath2]

    inFile1 = "/home/scratch/pmargani/LASSI/scannerData/Clean9.ptx.csv"
    inFile2 = "/home/scratch/pmargani/LASSI/scannerData/Clean9.ptx.copy.csv"
    inFiles = [inFile1, inFile2]

    outFile = 'outFile'

    smoothGPUMulti(gpuPaths, inFiles, outFile, N)

    # did it work?
    outfiles = []
    xyzs = []
    for i, gpuPath in enumerate(gpuPaths):
        for dim in ['x', 'y', 'z']:
            dimFile = "%s.%d.%s.csv" % (outFile, (i+1), dim)
            dimPath = os.path.join(gpuPath, dimFile)
            outfiles.append(dimPath)
            print(dimPath)
            assert os.path.isfile(dimPath)
            print(("GPUs created file: ", dimPath))
        loadFile = os.path.join(gpuPath, "%s.%d" % (outFile, (i+1)))    
        xyz = loadLeicaDataFromGpus(loadFile)
        xyzs.append(xyz)
    
    x1, y1, z1 = xyzs[0]
    x2, y2, z2 = xyzs[1]

    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    z = np.concatenate((z1, z2))

    scatter3dPlot(x, y, z, "testSmoothGPUMulti")

def smoothGPUMulti(gpuPaths,
                   inFiles,
                   outFile,
                   n,
                   test=False):
    "Smooth the data using multiple GPUs"
    
    gpuMultiScript = "runGpuParts"

    # first just try it with two GPUs
    user = os.getlogin()
    host1 = settings.GPU_HOST
    host2 = settings.GPU_HOST_2
    hosts = [host1, host2]
    parts = [1, 2]
    
    cmds = []
    for i in range(len(parts)):
        part = parts[i]
        host = hosts[i]
        gpuPath = gpuPaths[i]
        inFile = inFiles[i]
        cmd = [gpuMultiScript,
               gpuPath,
               user,
               host,
               inFile,
               outFile,
               "%s" % n,
               "%d" % part,
               "%d" % len(parts)
        ]

        print("cmd: ", cmd)
        cmds.append(cmd)

    if not test:
        # make sure they get fired off
        p1 = subprocess.Popen(cmds[0])  
        print("called first command")
        p2 = subprocess.Popen(cmds[1])
        print("called second command")

        # THEN wait for them to finish
        print("waiting for both to finish ...")
        p1.wait()
        p2.wait()

    print("multiple GPU commands finished")

    return cmds

def smoothGPUs(gpuPath,
               inFile,
               outFile,
               n,
               test=False,
               verbose=False,
               noCos=False,
               spherical=False,
               sigAzEl=None):
    "Ssh's to RH6 machine to run gpu code"

    # catch mutually exclusive options
    if noCos and spherical:
        raise "noCos and spherical are mutally exclusive options"

    # the sigAz and sigEl will always be identical
    if sigAzEl is None:
        sigAzEl = 0.001
    
    # get ssh credentials and target
    user = os.getlogin()
    host = settings.GPU_HOST

    cmd = "runGpuSmooth %s %s %s %s %s %d %1.5f" % (gpuPath, user, host, inFile, outFile, n, sigAzEl)
    # cmd = "runGpuSmooth %s %s %s %d %1.5f" % (gpuPath, inFile, outFile, n, sigAzEl)
    # if sigAz is not None:
    #     cmd += " --sigAz %1.5f " % sigAz
    # if sigEl is not None:
    #     cmd += " --sigEl %1.5f " % sigEl
    if noCos:
        cmd += " --no-cos"

    # spherical option means whether GPUs will be
    # doing spherical coord transform or not
    if spherical:
        cmd += " --no-conv"

    print("system cmd: ", cmd)

    if not test:
        os.system(cmd)

    return cmd

def trySmoothGPUs():

    # gpuPath = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing"
    gpuPath = GPU_PATH
    abspath = os.path.abspath(os.path.curdir)
    inFile = "Test1_STA14_Bump1_High-02_METERS.ptx.csv"
    fpath1 = os.path.join(abspath, "data", inFile)
    n = 100
    assert os.path.isfile(fpath1)
    #outFile1 = os.path.basename(fpath1)
    outFile1 = inFile
    smoothGPUs(gpuPath, fpath1, outFile1, n)

    xOutfile = os.path.join(gpuPath, inFile + ".x.csv")
    assert os.path.isfile(xOutfile)    

def smoothXYZGpu(x, y, z, n, sigXY=None, filename=None):
    "use GPU code to do the simple XYZ smoothing"

    if sigXY is None:
        sigXY = 0.1


    # first get data into file format expected by GPU code:
    # x, y, z per line
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    x = x[[not b for b in np.isnan(x)]]
    y = y[[not b for b in np.isnan(y)]]
    z = z[[not b for b in np.isnan(z)]]

    assert len(x) == len(y)
    assert len(y) == len(z)

    # TBF: how to zip 3 together?
    xyz = []
    for i in range(len(x)):
        xyz.append((x[i], y[i], z[i]))
    xyz = np.array(xyz)

    # where's our input data?
    abspath = os.path.abspath(os.path.curdir)
    if filename is None:
        fn = "test"
    else:
        fn = filename
            
    inFile = os.path.join(abspath, "data", fn)

    np.savetxt(inFile, xyz, delimiter=",") 

    # call the GPU code
    # where is the code we'll be running?
    # gpuPath = "/home/sandboxes/pmargani/LASSI/gpus/versions/gpu_smoothing"
    gpuPath = GPU_PATH
    outFile = fn
    smoothGPUs(gpuPath, inFile, outFile, n, noCos=True, sigAzEl=sigXY)

    # make sure the output is where it should be
    for dim in ['x', 'y', 'z']:
        dimFile = "%s.%s.csv" % (outFile, dim)
        dimPath = os.path.join(gpuPath, dimFile)
        assert os.path.isfile(dimPath)

    # extract the results from the resultant files
    outPath = os.path.join(gpuPath, outFile)
    return loadLeicaDataFromGpus(outPath)    


def loadLeicaDataFromGpus(fn):
    "Crudely loads x, y, z csv files into numpy arrays"

    xyzs = {}
    dims = ['x', 'y', 'z']
    for dim in dims:
        data = []
        fnn = "%s.%s.csv" % (fn, dim)
        with open(fnn, 'r') as f:
            ls = f.readlines()
        for l in ls:
            ll = l[:-1]
            if ll == 'nan':
                data.append(np.nan)
            else:
                data.append(float(ll))
        xyzs[dim] = np.array(data)
    return xyzs['x'], xyzs['y'], xyzs['z']    
