import os

from astropy.io import fits
try:
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D
except:
    print "No Plotting for you!"

import numpy as np

from ProjectScanLog import ProjectScanLog

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

def readActSrfTxt(fn):

    f = open(fn, 'r')
    ls = f.readlines()

    x = []
    y = []
    z = []

    # skip the comments in the first two lines:
    print ls[:2]
    for l in ls[2:]:
        ps = l.split(' ')
        x.append(float(ps[0]))
        y.append(float(ps[1]))
        z.append(int(ps[2][:-1]))

    return np.array(x), np.array(y), np.array(z)

def smoothSlow(az, el, r, n, sigEl=None, sigAz=None):
    "smooth our data"

    azRange = np.linspace(min(az), max(az), n)
    elRange = np.linspace(min(el), max(el), n)

    azLoc, elLoc = np.meshgrid(azRange, elRange)

    if sigEl is None:
        sigEl=0.001
    if sigAz is None:    
        sigAz=0.001;

    # init our smoothing result
    rSm = np.ndarray(shape=(n,n))
    rSms = []
    for j in range(n):
        # print "J:", j
        for k in range(n):
            w=2*np.pi*np.exp( (- (az - azLoc[j,k])**2 /( 2.*sigAz**2 )-(el-elLoc[j,k])**2 /(2.*sigEl**2 )))
            norm=sum(w)
            if norm==0:
                norm=1
                rSm[j,k]=np.nan #0 #min( r )
            else:
                w = w / norm
                rSm[j,k] = sum(r * w)

    return (azLoc, elLoc, rSm)

def plotActSrfTxt(fn):

    x, y, z = readActSrfTxt(fn)

    f = plt.figure()
    ax = Axes3D(f)
    ax.scatter(x, y, z)
    plt.title("Act Srf Txt")

    # now regrid the data
    N = 100
    sigX = sigY = .1
    xLoc, yLoc, zSmooth = smoothSlow(x, y, z, N, sigEl=sigX, sigAz=sigY)

    print "Smoothed data using %s x %d size grid, sigs: %f, %f" % (N, N, sigX, sigY)

    f = plt.figure()
    ax = Axes3D(f)
    ax.plot_surface(xLoc, yLoc, zSmooth)
    plt.title("Act Srf Txt Smoothed")

    f = plt.figure()
    ax = f.gca()
    cax = ax.imshow(zSmooth)
    cbar = f.colorbar(cax)
    plt.title("Act Srf Txt Smoothed")
    plt.show()

def analyzeActiveSurfaceScan(path, fn, scanNum):

    print "Scan: ", scanNum
    fitsPath = os.path.join(path, fn)

    print "FITS: ", fitsPath
    plotFile(fitsPath)

    txtFile = "asdata.%s.txt" % scanNum
    txtPath = os.path.join(path, txtFile)

    print "Txt:", txtPath
    if os.path.isfile(txtPath):    
        plotActSrfTxt(txtPath)
    else:
        print "no asdata.*.txt: ", txtPath

def analyzeActiveSurfaceScans(scanLogPath, scanNums):

    device = "ActiveSurfaceMgr"
    p = ProjectScanLog(scanLogPath)
    p.open()
    
    for scanNum in scanNums:
        print ""
        f = p.getDeviceFilename(device, scanNum)
        path = os.path.join(scanLogPath, device)
        analyzeActiveSurfaceScan(path, f, scanNum)

def parseAsZernikeConf(fn):

    f = open(fn, 'r')

    # skipp comments
    ls = f.readlines()
    for l in ls[7:]:
        ps = l.split(' ')
        # actuator[rib][hoop] x y rho theta phi rho_y theta_y phi_y

def main():
    #fn  = 'data/ActiveSurfaceMgr/2019_02_07_17:33:29.fits'
    #fn2 = 'data/ActiveSurfaceMgr/2019_02_07_18:38:01.fits'
    #plotFile(fn)
    #print diffField(fn, fn2, "SURFACE", "ABSOLUTE")
    path = "/users/pmargani/tmp/simdata/TINT_080219/"
    analyzeActiveSurfaceScans(path, [1, 2])

if __name__ == '__main__':
    main()
