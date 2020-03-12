import csv
import os
import subprocess
import time
from copy import copy

import numpy as np

from plotting import scatter3dPlot
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

    host1 = settings.GPU_HOST
    host2 = settings.GPU_HOST_2
    hosts = [host1, host2]

    outFile = 'outFile'

    smoothGPUMulti(gpuPaths, hosts, inFiles, outFile, N)

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

def smoothGPUParallel(inFile, N, test=False):
    "Highest level for calling smoothing using multiple GPUs"

    print ("smoothGPUParallel: ", N, inFile)
    assert os.path.isfile(inFile)

    # we need to specify the aboslute path of our inputs
    abspath = os.path.abspath(inFile)

    # our output will be same file name, but with appendages
    outFile = os.path.basename(inFile)

    gpuPaths = settings.GPU_MULTI_PATHS
    gpuHosts = settings.GPU_MULTI_HOSTS

    assert len(gpuPaths) == len(gpuHosts)


    cmds = smoothGPUMultiFile(gpuPaths, gpuHosts, abspath, outFile, N, test=test)

    # if this was a test, no files were created on disk for us to read
    if test:
        print ("Tested with commands", cmds)
        return None

    return loadParallelGPUFiles(outFile, gpuPaths)

def loadParallelGPUFiles(outFile, gpuPaths):

    numGpus = len(gpuPaths)

    # now collect the multiple results into one
    outfiles = []
    # xyzs = []
    # initialize what will become our final results
    x = np.array([])
    y = np.array([])
    z = np.array([])

    for i, gpuPath in enumerate(gpuPaths):
        for dim in ['x', 'y', 'z']:
            # there's a little confusion over how to handle
            # this parallel stuff if we just specified one GPU
            if numGpus == 1:
                dimFile = "%s.%s.csv" % (outFile, dim)
            else:    
                dimFile = "%s.%d.%s.csv" % (outFile, (i+1), dim)
            dimPath = os.path.join(gpuPath, dimFile)
            outfiles.append(dimPath)
            print(dimPath)
            assert os.path.isfile(dimPath)
            print("GPUs created file: ", dimPath)
        # now load the data using the base name shared by all dimensions
        if numGpus == 1:    
            loadFile = os.path.join(gpuPath, outFile)    
        else:    
            loadFile = os.path.join(gpuPath, "%s.%d" % (outFile, (i+1)))    
        print("Loading from basename: ", loadFile)
        xyz = loadLeicaDataFromGpus(loadFile)
        # xyzs.append(xyz)
        xn, yn, zn = xyz
        x = np.concatenate((x, xn))
        y = np.concatenate((y, yn))
        z = np.concatenate((z, zn))

    # x1, y1, z1 = xyzs[0]
    # x2, y2, z2 = xyzs[1]

    # x = np.concatenate((x1, x2))
    # y = np.concatenate((y1, y2))
    # z = np.concatenate((z1, z2))

    return x, y, z

def loadGPUFiles(outfiles):
    "Given list of paths to file names w/ out [x,y,z].csv extension, load them"
    # numGpus = len(gpuPaths)
    numGpus = len(outfiles)

    # now collect the multiple results into one
    # outfiles = []
    # xyzs = []
    # initialize what will become our final results
    x = np.array([])
    y = np.array([])
    z = np.array([])

    dimFiles = []

    # for i, gpuPath in enumerate(gpuPaths):
    for i, gpuPath in enumerate(outfiles):
        for dim in ['x', 'y', 'z']:
            # there's a little confusion over how to handle
            # this parallel stuff if we just specified one GPU
            # if numGpus == 1:
            #     dimFile = "%s.%s.csv" % (outFile, dim)
            # else:    
            #     dimFile = "%s.%d.%s.csv" % (outFile, (i+1), dim)
            # if numGpus == 1:
            #     dimFile = "%s.%s.csv" % (gpuPath, dim)
            # else:
            #     dimFile = "%s.%d.%s.csv" % (gpuPath, (i+1), dim)
            dimFile = "%s.%s.csv" % (gpuPath, dim)    
            # dimPath = os.path.join(gpuPath, dimFile)
            # outfiles.append(dimPath)
            dimFiles.append(dimFile)
            print(dimFile)
            assert os.path.isfile(dimFile)
            print("GPUs created file: ", dimFile)
        # now load the data using the base name shared by all dimensions
        # if numGpus == 1:    
        #     loadFile = os.path.join(gpuPath, outFile)    
        # else:    
        #     loadFile = os.path.join(gpuPath, "%s.%d" % (outFile, (i+1)))
        loadFile = gpuPath    
        print("Loading from basename: ", loadFile)
        xyz = loadLeicaDataFromGpus(loadFile)
        # xyzs.append(xyz)
        xn, yn, zn = xyz
        x = np.concatenate((x, xn))
        y = np.concatenate((y, yn))
        z = np.concatenate((z, zn))

    # x1, y1, z1 = xyzs[0]
    # x2, y2, z2 = xyzs[1]

    # x = np.concatenate((x1, x2))
    # y = np.concatenate((y1, y2))
    # z = np.concatenate((z1, z2))

    return x, y, z

def smoothGPUMultiFile(gpuPath, hosts, inFile, outFile, n, test=False):
    "Prepare the input for processing by multiple GPUs"

    # currently it seems that the GPU code can't concurrently read
    # from the same file

    # TBF: we might also want to only read in part of the data of
    # each file

    # for now, make copies of the input file for each gpu
    numInFiles = len(gpuPath)

    # build up the files.  Assume the in put file is a csv
    inFiles = [inFile]
    for i in range(2, numInFiles + 1):
        inFiles.append(inFile + ('.%d.csv' % i))
    print ("infiles: ", inFiles)

    # now copy them; time it!
    s = time.time()

    # if not test:
    ps = []
    for f in inFiles[1:]:
        if test:
            # just do something
            cmd = ["ls", inFile]
        else:
            # shell for copying file!    
            cmd = ["cp", inFile, f]
        print (cmd)
        p = subprocess.Popen(cmd)
        ps.append(p)
    
    # now wait for them all to finish
    for p in ps:
        p.wait()

    # make sure they really got copied
    if not test:
        for f in inFiles:
            assert os.path.isfile(f)

    # how long did the copy take?
    print("Copy files elapsed seconds: ", time.time() - s)

    return smoothGPUMulti(gpuPath, hosts, inFiles, outFile, n, test=test)            

def smoothGPUMulti(gpuPaths,
                   hosts,
                   inFiles,
                   outFile,
                   n,
                   test=False):
    "Smooth the data using multiple GPUs"
    
    # print (len(gpuPaths))
    # print (len(hosts))
    # print (len(inFiles))

    # make sure inputs make sense
    assert len(gpuPaths) == len(hosts)
    assert len(hosts) == len(inFiles)

    numParts = len(gpuPaths)

    scriptDir = os.path.dirname(os.path.realpath(__file__))
    gpuMultiScript = "{0}/runGpuParts".format(scriptDir)

    # first just try it with two GPUs
    user = os.getlogin() if not test else "test"
    # host1 = settings.GPU_HOST
    # host2 = settings.GPU_HOST_2
    # hosts = [host1, host2]
    # parts = [1, 2]
    parts = range(1, numParts + 1)

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
        ps = []
        for cmd in cmds:
            p = subprocess.Popen(cmd)  
            print("called command: ", cmd)
            ps.append(p)

        # THEN wait for them to finish
        print("waiting for all commands to finish ...")
        for p in ps:
            p.wait()

    print("multiple GPU commands finished")

    return cmds

def smoothGPUs(gpuPath,
               inFile,
               outFile,
               n,
               host=None,
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
    user = os.getlogin() if not test else "test"

    if host is None:
        host = settings.GPU_HOST

    scriptDir = os.path.dirname(os.path.realpath(__file__))
    
    cmd = "%s/runGpuSmooth %s %s %s %s %s %d %1.5f" % (scriptDir, gpuPath, user, host, inFile, outFile, n, sigAzEl)
    
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
    # TBF: we have moved to multiple GPUS, but we shouldn't be
    # doing this anymore with GPUs, so this is a temp. kluge
    gpuPath = settings.GPU_MULTI_PATHS[0]
    host = settings.GPU_MULTI_HOSTS[0]
    outFile = fn
    smoothGPUs(gpuPath, inFile, outFile, n, host=host, noCos=True, sigAzEl=sigXY)

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
