from astropy.io import fits
try:
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D
except:
    print "No Plotting for you!"

import numpy as np

def plotData(data, scanNumber, fieldName):
    fig = plt.figure()
    ax = Axes3D(fig)
    hoops = data.field('HOOP')
    ribs = data.field('RIB')
    plt.xlabel('hoop #')
    plt.ylabel('rib #')
    plt.title("%s:%s" % (scanNumber, fieldName))
    ax.scatter(hoops, ribs, data.field(fieldName))
    
def plotIndicated(data, scanNumber):
    "There's a wierd point we want to filter out"
    fig = plt.figure()
    ax = Axes3D(fig)
    hoops = data.field('HOOP')
    ribs = data.field('RIB')
    plt.xlabel('hoop #')
    plt.ylabel('rib #')
    fieldName = 'INDICATED'

    plt.title("%s:Filtered_%s" % (scanNumber, fieldName))
    
    ind = data.field(fieldName)
    print "total indicated: ", len(ind)
    #ind = ind[ind > -1e-4]
    # get rid of this one weird outlier
    tol = np.min(ind)
    wh = np.where(ind <= tol)
    ind = np.delete(ind, wh[0])
    hoops = np.delete(hoops, wh[0])
    ribs = np.delete(ribs, wh[0])
    print "filtered indicated: ", len(ind)
    ax.scatter(hoops, ribs, ind)
    
def plotZernikes(ext, title):
    names = ext.data.field('NAME')
    values = ext.data.field('VALUE')
    fig = plt.figure()
    ax = fig.gca()
    plt.title(title)
    ax.plot(range(len(values)), values)
    
def plotFile(fn):
    hs = fits.open(fn)
    hdr = hs[0].header
    
    scan = hdr['SCAN']
    print "Zeros enabled:", hdr['ZERO']
    print "FEM enabled:", hdr['FEM']
    print "Random enabled:", hdr['RANDOM']
    print "Zernikes enabled:", hdr['ZERNIKE']
    print "Thermal Zernikes enabled", hdr['THRMZERN']
    
    if len(hs) < 2:
        print "Active Surface FITS file only has Primary Header"
        return

    if len(hs) > 1 and hs[1].name == "ZERNIKE":
        plotZernikes(hs[1], hs[1].name)
    if len(hs) > 2 and hs[2].name == "THRMZERN":
        plotZernikes(hs[2], hs[2].name)
    
    try:
        data = hs['SURFACE'].data
    except KeyError:
        print "Does not contain SURFACE extension"
        return

    #plotData(data, scan, 'INDICATED')
    plotIndicated(data, scan)
    plotData(data, scan, 'ABSOLUTE')


def surfacePlot(data, z, title):
    fig = plt.figure()
    ax = Axes3D(fig)
    hoops = data.field('HOOP')
    ribs = data.field('RIB')
    plt.xlabel('hoop #')
    plt.ylabel('rib #')

    plt.title(title)

    print "z: ", z
    ax.scatter(hoops, ribs, z)
    

def plotDiffs(fn1, fn2):

    plotFile(fn1)
    plotFile(fn2)

    diff, abs1, abs2 = diffField(fn1, fn2, "SURFACE", "ABSOLUTE")

    print type(diff), diff
    print "ABSOLUTE Diff between scans: min=%f, max=%f, mean=%f " % (np.min(diff), np.max(diff), np.mean(diff))
    # just need the hoops and ribs, which are always the same
    hs = fits.open(fn1)
    data = hs["SURFACE"].data

    surfacePlot(data, diff, "ABSOLUTE Difference")

    return

    fig = plt.figure()
    ax = Axes3D(fig)
    hoops = data.field('HOOP')
    ribs = data.field('RIB')
    plt.xlabel('hoop #')
    plt.ylabel('rib #')

    plt.title("diff")

    scatter(hoops, ribs, diff)

def diffField(fn1, fn2, tableName, fieldName):

    hs1 = fits.open(fn1)
    data1 = hs1[tableName].data.field(fieldName)
    hs2 = fits.open(fn2)
    data2 = hs2[tableName].data.field(fieldName)

    return data1 - data2, data1, data2


def main():
    fn  = 'data/ActiveSurfaceMgr/2019_02_07_17:33:29.fits'
    fn2 = 'data/ActiveSurfaceMgr/2019_02_07_18:38:01.fits'
    #plotFile(fn)
    print diffField(fn, fn2, "SURFACE", "ABSOLUTE")

if __name__ == '__main__':
    main()
