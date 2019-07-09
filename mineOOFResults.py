import os
import pickle
import numpy as np

import matplotlib.pylab as plt
import pyfits

def findOOFDirs(path):
    print("Finding OOF dirs in", path)
    dirs = []
    for thisDir, childDirs, filenames in os.walk(path):
        if 'OOF' in childDirs:
            dirs.append(thisDir)
    print("done")        
    return dirs        

def getAllZernikeResults(oofDirs):
    zs = []
    for i, oofDir in enumerate(oofDirs):
        print("%d of %d oofDirs" % (i, len(oofDirs)))
        z = getZernikeResultsFromOOFDir(oofDir)
        if len(z) > 0:
            zs.extend(z)
    return zs        

def getZernikeResultsFromOOFDir(oofDir):

    # import ipdb; ipdb.set_trace()

    if 'OOF' not in oofDir:
        oofDir = os.path.join(oofDir, 'OOF')

    zs = []
    solDirs = os.listdir(oofDir)
    for solDir in solDirs:
        solDirPath = os.path.join(oofDir, solDir)
        z = getZernikeResults(solDirPath)
        if z is not None:
            zs.append(z)
    return zs    

def getZernikeResults(solutionDirPath):

    order = 5
    expFile = "Solutions/z%d/tzernike.fits" % order    
    expPath = os.path.join(solutionDirPath, expFile)

    if not os.path.isfile(expPath):
        print("No results in ", expPath)
        return None

    # read the file like GFM does
    hdulist = pyfits.open(expPath)
    extData = hdulist[1].data
    extColumnNames = extData.names
    extColumnDefs  = extData.formats
    zdict = makeZdict(extData)
    hdulist.close()

    return zdict

def makeZdict(extData):
    tzernikes =[] 
    zdict = {}
    for zernike in extData:
        tzernikes.append(eval(str(zernike)))
    for zernike in tzernikes:
        zdict[zernike[0]] = zernike[1]

    return zdict

def saveObj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadObj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)               

def analyzeZ(name, zs):

    meanV = np.mean(zs)
    maxV = np.max(zs)
    minV = np.min(zs)
    stdV = np.std(zs)
    rng = (meanV - (2*stdV), meanV + (2*stdV))

    print("%10s: len=%5d, min=% 8.2f, max=% 8.2f, mean=% 8.2f, std=% 8.2f, rng=(% 8.2f, %8.2f)" % (name, len(zs), minV, maxV, meanV, stdV, rng[0], rng[1]))

    # make some histograms - but get rid of outliers
    outHi = meanV + (2*stdV)
    outLo = meanV - (2*stdV)
    zMid = [z for z in zs if z > outLo and z < outHi]
    zLog = [np.log(np.abs(z)) for z in zs]

    f = plt.figure()
    ax = f.gca()
    ax.hist(zs)
    plt.title(name + " all")

    f = plt.figure()
    ax = f.gca()
    ax.hist(zLog)
    plt.title(name + " all log")

    f = plt.figure()
    ax = f.gca()
    ax.hist(zMid, bins=10)
    plt.title(name + " inner 2 sigma")

def findOofs():

    oofDirs = findOOFDirs("/home/gbtdata")
    #getZernikeResultsFromOOFDir(oofDirs[0])
    #zss = getZernikeResultsFromOOFDir("/home/gbtdata/AGBT17B_151_28")
    #oofDirs = ["/home/gbtdata/AGBT18A_014_04", "/home/gbtdata/AGBT17B_151_28"]
    zss = getAllZernikeResults(oofDirs)
    #for zs in zss:
    #    print zs

    # organize values by keys
    values = {}
    for zs in zss:
        for k, v in zs.items():
            if k not in values:
                values[k] = []
            values[k].append(v)
            
    # save to a pickle file for later analysis
    saveObj(values, "oofValues")

    # compute stats
    print("Stats from %d results" % (len(zss)))
    keys = sorted(values.keys())
    for k in keys:
        v = values[k]
        v = np.array(v)
        meanV = np.mean(v)
        maxV = np.max(v)
        minV = np.min(v)
        stdV = np.std(v)
        rng = (meanV - stdV, meanV + stdV)
        print("%10s: len=%5d, min=% 8.2f, max=% 8.2f, mean=% 8.2f, std=% 8.2f, rng=(% 8.2f, %8.2f)" % (k, len(v), minV, maxV, meanV, stdV, rng[0], rng[1]))


if __name__ == "__main__":
    findOofs()
    #x = {'a': [1., 2.], 'b': [3., 4.]}
    #saveObj(x, "test")
    #x2 = loadObj("test")
    #print 'we got: ', x2
