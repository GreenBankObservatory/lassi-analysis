import os

from astropy.io import fits
try:
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D
except:
    print("No Plotting for you!")

import numpy as np

from ProjectScanLog import ProjectScanLog
from AsZernikeFile import AsZernikeFile

def plotData(data,
             scanNumber,
             fieldName,
             xlim=None,
             ylim=None,
             zlim=None,
             filterDisabled=True,
             xy=False,
             dividePhi=False,
             filterMin=False,
             test=False):

    hoops = data.field('HOOP')
    ribs = data.field('RIB')
    z = data.field(fieldName)



    if filterMin:
        orgLen = len(z)
        tol = np.min(z)
        wh = np.where(z <= tol)
        z = np.delete(z, wh[0])
        hoops = np.delete(hoops, wh[0])
        ribs = np.delete(ribs, wh[0])
        print("Filtered min values from data, len from %d to %d" % (orgLen, len(z)))

    title = "%s:%s" % (scanNumber, fieldName)

    if dividePhi:
        phis = []
        fn = "/home/gbt/etc/config/AsZernike.conf"
        asz = AsZernikeFile(fn)
        asz.parse()
        for i, h in enumerate(hoops):
            r = ribs[i]
            act = asz.actuators[(h,r)]
            phis.append(act.phi)
        phis = np.array(phis)
        print("Dividing z axis by phis.")
        z = z / phis
        title += ":phis"

    if xy:
        print("Plotting in x and y.")
        # convert to x and y
        xlabel = 'x'
        ylabel = 'y'
        # TBF: make this a singleton
        fn = "/home/gbt/etc/config/AsZernike.conf"
        asz = AsZernikeFile(fn)
        asz.parse()
        x = []
        y = []
        for i, h in enumerate(hoops):
            r = ribs[i]
            act = asz.actuators[(h,r)]
            x.append(act.x)
            y.append(act.y)
        x = np.array(x)    
        y = np.array(y)    
    else:
        # stick with hoops and ribs
        xlabel = 'hoop #'
        ylabel = 'rib #'
        x = hoops
        y = ribs

    # Use masked arrays.
    x = np.ma.masked_invalid(x)
    y = np.ma.masked_invalid(y)
    z = np.ma.masked_invalid(z)

    if filterDisabled:
        enabled= data.field('ENABLED')
        print(enabled)
        mask = enabled == True
        print("mask: ", mask)
        orgLen = len(x)
        #x = x[mask]
        #y = y[mask]
        #z = z[mask]
        x.mask = ~mask
        y.mask = ~mask
        z.mask = ~mask
        print("Removing %d disabled actuators" % (orgLen - len(z.compressed())))

    # now that we have the data the way we want it,
    # plot it
    if not test:
        fig = plt.figure()
        ax = Axes3D(fig)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if zlim is not None:
            ax.set_zlim(zlim)
        ax.scatter(x.compressed(), y.compressed(), z.compressed())
    
    return x, y, z

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
    print("total indicated: ", len(ind))
    #ind = ind[ind > -1e-4]
    # get rid of this one weird outlier
    tol = np.min(ind)
    wh = np.where(ind <= tol)
    ind = np.delete(ind, wh[0])
    hoops = np.delete(hoops, wh[0])
    ribs = np.delete(ribs, wh[0])
    print("filtered indicated: ", len(ind))
    ax.scatter(hoops, ribs, ind)
    
    return hoops, ribs, ind

def plotZernikes(ext, title):
    names = ext.data.field('NAME')
    values = ext.data.field('VALUE')
    fig = plt.figure()
    ax = fig.gca()
    plt.title(title)
    ax.plot(range(len(values)), values)

    # print non-zero values
    nzValues = [(i, v) for i, v in enumerate(values) if v > 0.0 ]
    if len(nzValues) > 0:
        print("Non-zero zernikies for", title)
        for i, v in nzValues:
            print("%d: %5.2f" % (i, v))

def plotFile(fn, filterMin=False, filterDisabled=True, test=False):
    hs = fits.open(fn)
    hdr = hs[0].header
    
    scan = hdr['SCAN']
    print("Zeros enabled:", hdr['ZERO'])
    print("FEM enabled:", hdr['FEM'])
    print("Random enabled:", hdr['RANDOM'])
    print("Zernikes enabled:", hdr['ZERNIKE'])
    print("Thermal Zernikes enabled", hdr['THRMZERN'])
    
    if len(hs) < 2:
        print("Active Surface FITS file only has Primary Header")
        return

    #if len(hs) > 1 and hs[1].name == "ZERNIKE":
    #    plotZernikes(hs[1], hs[1].name)
    #if len(hs) > 2 and hs[2].name == "THRMZERN":
    #    plotZernikes(hs[2], hs[2].name)
    
    try:
        hdu = hs['ZERNIKE']
        plotZernikes(hdu, 'ZERNIKE')
    except KeyError:
        print("Does not contain ZERNIKE extension")

    try:
        hdu = hs['THRMZERN']
        plotZernikes(hdu, 'THRMZERN')
    except KeyError:
        print("Does not contain ZERNIKE extension")

    try:
        data = hs['SURFACE'].data
    except KeyError:
        print("Does not contain SURFACE extension")
        return

    #plotData(data, scan, 'INDICATED')
    print("The Indicated column in FITS is *actually*")
    print("the Indicated actuator values minus their zero points")
    # indData = plotIndicated(data, scan)
    indData = plotData(data,
                       scan,
                       'INDICATED',
                       filterDisabled=filterDisabled,
                       filterMin=filterMin,
                       test=test)
    h, r, ind = indData
    zlim = (np.min(ind), np.max(ind))

    print("Plot Indicated again, but taking into account phi")
    h, r, indPhi = plotData(data,
                 scan,
                 'INDICATED',
                 zlim=zlim,
                 filterMin=filterMin,
                 filterDisabled=filterDisabled,
                 dividePhi=True,
                 test=test)

    print("Plot Indicated again, but taking into account phi, and in x, y")
    x, y, indPhi = plotData(data,
                 scan,
                 'INDICATED',
                 zlim=zlim,
                 xy=True,
                 filterMin=filterMin,
                 filterDisabled=filterDisabled,
                 dividePhi=True,
                 test=test)

    print("The Absolute column in FITS is *actually*")
    print("the Indicated (readback from hardware) actuator values.")
    h, r, absd = plotData(data,
                          scan,
                          'ABSOLUTE',
                          filterDisabled=filterDisabled,
                          test=test)

    print("The Delta column in FITS is *actually*")
    print("the difference between the commanded and indicated (actual) positions")
    print("this will be all zeros in the simulator")
    delData = plotData(data, scan, 'DELTA', test=test)

    return h, r, x, y, ind, indPhi, absd

def extractSurface(filename, fieldName, dividePhi=True, xy=True, filterDisabled=False, verbose=False):
    """
    """

    hdu = fits.open(filename)
    hdr = hdu[0].header

    scan = hdr['SCAN']
    
    if verbose:
        print("Zeros enabled:", hdr['ZERO'])
        print("FEM enabled:", hdr['FEM'])
        print("Random enabled:", hdr['RANDOM'])
        print("Zernikes enabled:", hdr['ZERNIKE'])
        print("Thermal Zernikes enabled", hdr['THRMZERN'])

    if len(hdu) < 2:
        print("Active Surface FITS file only has Primary Header")
        return

    try:
        data = hdu['SURFACE'].data
    except KeyError:
        print("Does not contain SURFACE extension")
        return

    hoops = data.field('HOOP')
    ribs = data.field('RIB')
    z = data.field(fieldName)

    if dividePhi:
        phis = []
        fn = "/home/gbt/etc/config/AsZernike.conf"
        asz = AsZernikeFile(fn)
        asz.parse()
        for i, h in enumerate(hoops):
            r = ribs[i]
            act = asz.actuators[(h,r)]
            phis.append(act.phi)
        phis = np.array(phis)
        if verbose:
            print("Dividing z axis by phis.")
        z = z / phis

    if xy:
        fn = "/home/gbt/etc/config/AsZernike.conf"
        asz = AsZernikeFile(fn)
        asz.parse()
        x = []
        y = []
        for i, h in enumerate(hoops):
            r = ribs[i]
            act = asz.actuators[(h,r)]
            x.append(act.x)
            y.append(act.y)
        x = np.array(x)
        y = np.array(y)
    else:
        x = hoops
        y = ribs

    # Use masked arrays.
    x = np.ma.masked_invalid(x)
    y = np.ma.masked_invalid(y)
    z = np.ma.masked_invalid(z)

    if filterDisabled:
        enabled = data.field('ENABLED')
        mask = (enabled == True)
        x.mask = ~mask
        y.mask = ~mask
        z.mask = ~mask
        if verbose:
            print("Removing %d disabled actuators" % (len(z) - len(z.compressed())))
   
    return x, y, z 

def surfacePlot(data, z, title):
    fig = plt.figure()
    ax = Axes3D(fig)
    hoops = data.field('HOOP')
    ribs = data.field('RIB')
    plt.xlabel('hoop #')
    plt.ylabel('rib #')

    plt.title(title)

    print("z: ", z)
    ax.scatter(hoops, ribs, z)
    

def plotDiffs(fn1, fn2):

    plotFile(fn1)
    plotFile(fn2)

    diff, abs1, abs2 = diffField(fn1, fn2, "SURFACE", "ABSOLUTE")

    print(type(diff), diff)
    print("ABSOLUTE Diff between scans: min=%f, max=%f, mean=%f " % (np.min(diff), np.max(diff), np.mean(diff)))
    # just need the hoops and ribs, which are always the same
    hs = fits.open(fn1)
    data = hs["SURFACE"].data

    surfacePlot(data, diff, "ABSOLUTE Difference")

    return diff

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # hoops = data.field('HOOP')
    # ribs = data.field('RIB')
    # plt.xlabel('hoop #')
    # plt.ylabel('rib #')

    # plt.title("diff")

    # scatter(hoops, ribs, diff)

def diffField(fn1, fn2, tableName, fieldName):

    hs1 = fits.open(fn1)
    data1 = hs1[tableName].data.field(fieldName)
    hs2 = fits.open(fn2)
    data2 = hs2[tableName].data.field(fieldName)

    return data1 - data2, data1, data2

def readActSrfTxt(fn):
    "Parses ASCII file created by Act. Surf. Mgr."

    f = open(fn, 'r')
    ls = f.readlines()

    x = []
    y = []
    z = []

    # skip the comments in the first two lines:
    print(ls[:2])
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

def plotActSrfTxt(fn, test=False):
    "Read in, smooth, and plot the ASCII file generated by Act. Surf."

    x, y, z = readActSrfTxt(fn)

    f = plt.figure()
    ax = Axes3D(f)
    ax.scatter(x, y, z)
    plt.title("Act Srf Txt")

    # now regrid the data
    N = 100
    sigX = sigY = .1
    xLoc, yLoc, zSmooth = smoothSlow(x, y, z, N, sigEl=sigX, sigAz=sigY)

    print("Smoothed data using %s x %d size grid, sigs: %f, %f" % (N, N, sigX, sigY))

    if not test:
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

    return (x, y, z), (xLoc, yLoc, zSmooth)

def analyzeActiveSurfaceScan(path, fn, scanNum, test=False):

    print("Scan: ", scanNum)
    fitsPath = os.path.join(path, fn)

    print("FITS: ", fitsPath)
    fileData = plotFile(fitsPath, test=test)

    txtFile = "asdata.%s.txt" % scanNum
    txtPath = os.path.join(path, txtFile)

    print("Txt:", txtPath)
    if os.path.isfile(txtPath):    
        actTxtData = plotActSrfTxt(txtPath, test=test)
    else:
        print("no asdata.*.txt: ", txtPath)
        actTxtData = None

    return fileData, actTxtData
        
def analyzeActiveSurfaceScans(scanLogPath, scanNums, details=False, test=False):

    device = "ActiveSurfaceMgr"
    p = ProjectScanLog(scanLogPath)
    p.open()

    data = []    
    if details:
        for scanNum in scanNums:
            print("")
            f = p.getDeviceFilename(device, scanNum)
            path = os.path.join(scanLogPath, device)
            d = analyzeActiveSurfaceScan(path, f, scanNum, test=test)
            data.append(d)

    plotTxtFilesOnePlot(scanLogPath, scanNums)

    return data
    
def plotTxtFilesOnePlot(path, scanNums):

    path = os.path.join(path, "ActiveSurfaceMgr")

    fig = plt.figure()
    n = len(scanNums)
    for i, scanNum in enumerate(scanNums):
        plt.subplot(n, 3, i+1)
        fn = "asdata.%s.txt" % scanNum
        fn = os.path.join(path, fn)
        x, y, z = readActSrfTxt(fn)
        # now regrid the data
        N = 100
        sigX = sigY = .1
        xLoc, yLoc, zSmooth = smoothSlow(x, y, z, N, sigEl=sigX, sigAz=sigY)
        plt.title(str(scanNum))
        plt.imshow(zSmooth)

        #ax = Axes3D(fig)
        #ax.scatter(x, y, z)
        #plt.title("Act Srf Txt")
        
def parseAct(actStr):

    assert "act" in actStr

    i1_1 = actStr.index('[')
    i1_2 = actStr.index(']')

    rib = int(actStr[i1_1+1:i1_2])

    i2_1 = actStr.index('[', i1_2+1)
    i2_2 = actStr.index(']', i1_2+1)

    hoop = int(actStr[i2_1+1:i2_2])
 
    return rib, hoop

def parseAsZernikeConf(fn):

    f = open(fn, 'r')

    data = []

    # skipp comments
    ls = f.readlines()
    for l in ls[7:]:
        print(l)
        ps = l.split(' ')
        # act[rib][hoop] x y rho theta phi rho_y theta_y phi_y
        actuatorStr = ps[0]
        rib, hoop = parseAct(actuatorStr)
        x = float(ps[1])
        y = float(ps[2])
        rho = float(ps[3])
        theta = float(ps[4])
        phi = float(ps[5])
        data.append((actuatorStr, rib, hoop, x, y, rho, theta, phi))

    return data    

def main():
    #fn  = 'data/ActiveSurfaceMgr/2019_02_07_17:33:29.fits'
    #fn2 = 'data/ActiveSurfaceMgr/2019_02_07_18:38:01.fits'
    #plotFile(fn)
    #print diffField(fn, fn2, "SURFACE", "ABSOLUTE")
    path = "/users/pmargani/tmp/lassi-analysis/simdata/TINT_080219/"
    analyzeActiveSurfaceScans(path, [2, 3], details=True, test=True)

if __name__ == '__main__':
    main()
