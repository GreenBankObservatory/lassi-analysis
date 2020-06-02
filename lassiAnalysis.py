"""
This is a very high level module for integrating other modules and 
analyzing a single scan of Leica data once it's available.
"""

import os
import time
import numpy as np
from copy import copy
from shutil import copyfile

from astropy.stats import sigma_clip

import settings
import lassiTestSettings as usettings

from grid import regridXYZ, regridXYZMasked
from rotate import shiftRotateXYZ, align2Paraboloid
from SmoothedFITS import SmoothedFITS
from zernikies import getZernikeCoeffs, getZernikeCoeffsCycle
from plotting import sampleXYZData, scatter3dPlot, surfacePlot
from utils.utils import midPoint, gridLimits, polyFitRANSAC
from processPTX import processNewPTX, processNewPTXData, getRawXYZ
from parabolas import loadLeicaData, paraboloid, fitLeicaData, subtractParaboloid, paraboloidFit
from gpus import smoothGPUs, smoothXYZGpu, loadLeicaDataFromGpus,  smoothGPUParallel, loadParallelGPUFiles

# where is the code we'll be running?
GPU_PATH = settings.GPU_PATH
# where should we save the results from the GPU smoothing?
GPU_OUTPUT_PATH = settings.GPU_OUTPUT_PATH


def tryProcessLeicaDataStream():

    s = usettings.SETTINGS_27MARCH2019

    test = True
    if test:
        # read data from previous scans
        path = s['dataPath']
        fn = usettings.SCAN9

        fpath = os.path.join(path, fn)
        with open(fpath, 'r') as f:
            ls = f.readlines()

        # finally, substitue the data!
        x, y, z, i = getRawXYZ(ls)

        # fake the datetimes
        dts = np.zeros(len(x))
    
    # TBF: in production this might come from config file?        
    ellipse, rot = usettings.getData(s)

    # TBF: we'll get these from the manager
    proj = "TEST"
    dataDir = "/home/sandboxes/pmargani/LASSI/data"
    filename = "test2"

    hdr = {'test': 1}
    processLeicaDataStream(x,
                           y,
                           z,
                           i,
                           dts,
                           hdr,
                           ellipse,
                           rot,
                           proj,
                           dataDir,
                           filename)        


def processLeicaDataStream(x,
                           y,
                           z,
                           i,
                           dts,
                           hdr,
                           ellipse,
                           rot,
                           project,
                           dataDir,
                           filename,
                           plot=True):
    """
    x, y, z: data streamed from scanner
    i: intensity values
    dts: datetime values
    hdr: header dictionary from scanner stream
    ellipse: used for filtering data (dependent on scanner position)
    dataDir: where data files get written to (eg, /home/gbtdata)
    project: same sense as GBT project (eg, TGBT19A_500_01)
    filename: basename of file products (eg, 2019_09_26_01:35:43)
    """

    # do basic processing first: remove bad data, etc.
    # TBF: backwards compatible (bad) interface
    lines = None
    xyzi = (x, y, z, i)
    xyz, dts = processNewPTXData(lines,
                                 xyzi=xyzi,
                                 rot=rot,
                                 ellipse=ellipse,
                                 rFilter=True,
                                 iFilter=False)


    # TBF: refactor the GPU interface to not use files.
    # for now we need to create this file as input to GPUs
    fpathBase = os.path.join(dataDir, project, filename)
    processedPath = "%s.csv" % fpathBase
    print ("writing filtered data to CSV file: ", processedPath)
    np.savetxt(processedPath, xyz, delimiter=",")

    # We can lower this for testing purposes
    N = 512
    # N = 100

    x, y, z = smoothGPUParallel(processedPath, N)

    # save this off for later use
    fitsio = SmoothedFITS()
    fitsio.setData(x, y, z, N, hdr, dataDir, project, filename)
    fitsio.write()

    smoothedFitsFilename = fitsio.getFilePath()

    if plot:
        # how should we save the image of our processed data?
        ext = "smoothed.fits"
        fn = smoothedFitsFilename[:-len(ext)] + "processed.png"
        print("Processing smoothed data, imaging to:", fn)
        # process a little more and create a surface plot for diagnosis
        imageSmoothedData(x, y, z, N, filename=fn)

    return fitsio.getFilePath()

    # TBF: we need to create a visual of the scan,
    # but the fitting routine below for this
    # only hangs now, even though unit tests pass
    # create a visual for validation purposes
    # diff, x, y = fitLeicaScan(None, 
    #                           xyz=(x, y, z),
    #                           N=N,
    #                           rFilter=False,
    #                           weights=None)

   
    # xr, yr, zr = regridXYZ(x, y, diff, N)

    # and save it off for use in GFM later


def extractZernikesLeicaScanPair(refScanFile, sigScanFile, n=512, nZern=36):
    """
    Takes two smoothed Leica scans and extracts Zernike coefficients from their difference.

    Parameters
    ----------
    refScanFile : str
        File with the smoothed reference scan. <scan_name>.ptx.csv
    sigScanFile : str
        File with the smoothed signal scan. <scan_name>.ptx.csv
    n : int, optional
        Number of elements per side in the smoothed data.
    nZern : int, optional
        Number of Zernike polynomials to include in the fit. Can be up to 120.
    """

    print('Masking file: {}'.format(refScanFile))
    xra, yra, zra = maskScan(refScanFile, n=n)
    
    print('Masking file: {}'.format(sigScanFile))
    xsa, ysa, zsa = maskScan(sigScanFile, n=n)    

    
    # Find the grid limits for the reference and signal scans.
    xmin, xmax = gridLimits(xsa, xra)
    ymin, ymax = gridLimits(ysa, yra)
    print("Regridding scans.")
    xrag, yrag, zrag = regridXYZMasked(xra, yra, zra, n=n, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    xsag, ysag, zsag = regridXYZMasked(xsa, ysa, zsa, n=n, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)    

    # Surface deformation map: signal - difference
    diff = np.ma.masked_invalid(zsag - zrag)
    diff = sigma_clip(diff, 5)
    diff = np.ma.masked_invalid(diff)

    # Find Zernike coefficients on the surface deformation map.
    fitlist, diff_sub = getZernikeCoeffsCycle(xrag, yrag, diff, nZern=nZern, fill=0)

    return xrag, yrag, diff_sub, fitlist


def smooth(fpath, N=512, spherical=False):

    assert os.path.isfile(fpath)

    # we need to specify the aboslute path of our inputs
    abspath = os.path.abspath(fpath)

    # our output will be same file name, but with appendages
    outfile = os.path.basename(fpath)

    smoothGPUs(GPU_PATH,
               abspath,
               outfile,
               N,
               spherical=spherical)

    # make sure the output is where it should be
    outfiles = []
    for dim in ['x', 'y', 'z']:
        dimFile = "%s.%s.csv" % (outfile, dim)
        dimPath = os.path.join(GPU_PATH, dimFile)
        outfiles.append(dimPath)
        assert os.path.isfile(dimPath)
        print("GPUs created file: ", dimPath)

    return outfiles


def writeGPUoutput(gpuPath, fn, dim, data, cp2ext=None):

    fn = "%s.%s.csv" % (fn, dim)
    fpath = os.path.join(gpuPath, fn)
    # here we make sure each element gets its own line
    # with as much precision as what I think the GPU is doing
    print("writeGPUoutput to ", fpath)
    if cp2ext is not None and os.path.isfile(fpath):
        # avoid overwriting this file by copying to a location with 
        # this extension
        copyfile(fpath, fpath + '.' + cp2ext)

    np.savetxt(fpath, data, fmt="%.18f")
    return fpath


def processLeicaScan(fpath,
                     N=512,
                     rot=None,
                     ellipse=[-8., 50., 49., 49., 0.],
                     sampleSize=None,
                     plot=True):
    """
    High level function for processing leica data:
       * Processes PTX file: removes bad data and isolates the primary reflector.
       * Smooths it by calling GPU code.
    The output is a smoothed scan which is evenly sampled in spherical coordinates.
    Smoothed scan has to be masked and reggrided to Cartesian coordinates.
    
    Parameters
    ----------
    fpath : str
        Path to the .ptx file containing the scan data.
    N : int, optional
        Number of samples in the azimuth and elevation directions
        to use when smoothing.
    rot : float, optional
        Rotate the scan by this angle, in degrees, along the z axis after
        isolating the primary reflector.
    ellipse : list, optional
        Parameters of the ellipse used for the initial dish segmentation.
        The parameters are given as [center in x, center in y, major axis, 
        minor axis, rotation angle]
    sampleSize : int, optional
        Number of lines to use from the PTX file. A random sample of this 
        length will be selected. Defaults to the the whole PTX file.
    plot : bool, optional
        Display plots?
    """
    
    assert os.path.isfile(fpath)

    fileBasename = os.path.basename(fpath)

    # we'll provide primitive timing reports
    s = time.time()

    # removes headers, does basic rotations, etc.
    print("Processing PTX file ...")
    if rot is None:
        rot = 0.

    processedPath = fpath + ".csv"
    
    xyz, dts, intensity = processNewPTX(fpath,
                                        rot=rot,
                                        ellipse=ellipse,
                                        rFilter=True,
                                        iFilter=False,
                                        sampleSize=sampleSize)

    e = time.time()
    print("Elapsed minutes: %5.2f" % ((e - s) / 60.))

    # reduces our data via GPUs
    s = time.time()
    print("Smoothing data ...")

    x, y, z = smoothGPUParallel(processedPath, N)
    print("Smoothed data to shape: ", x.shape, y.shape, z.shape)

    smoothedFiles = ["{}.csv.{}.csv".format(os.path.join(GPU_OUTPUT_PATH, fileBasename), coo) for coo in ['x', 'y', 'z']]
    for smoothedFile,vector in zip(smoothedFiles, [x,y,z]):
        print("Saving smoothed data to: {}".format(smoothedFile))
        np.savetxt(smoothedFile, vector)

    e = time.time()
    print("Elapsed minutes: %5.2f" % ((e - s) / 60.))
    
    if plot:
        imageSmoothedData(x, y, z, N)


def loadProcessedData(filename):
    "Loads the results of processLeicaScan from file"
    d = np.load(filename)
    return d["xs"], d["ys"], d["diffs"], d["retroMask"]


def imageSmoothedData(x, y, z, N, filename=None):
    """
    Prepares the data to produce the diagnostic plots shown by GFM.

    :param x: x coordinates of the point cloud.
    :param y: y coordinates of the point cloud.
    :param z: z coordinates of the point cloud.
    :param N: number of points per side in the smoothed data.
    :param filename: save the plot to this location.
    """

    # Reshape the coordinates before masking.
    x.shape = (N,N)
    y.shape = (N,N)
    z.shape = (N,N)

    # Mask deviant points in the z coordinate.
    masked = maskXYZ(x, y, z, n=N, guess=[60., 0., 0., -49., 0., 0.], 
                     bounds=None, radialMask=True, maskRadius=49.)
    
    # Regrid to a uniformly sampled grid in cartesian coordinates, keeping the mask.
    xx, yy, zz = regridXYZMasked(x, y, masked['fitResidual'], n=N)

    # Plot the surface.
    surfacePlot(xx,
                yy,
                np.log10(abs(np.diff(zz))),
                title="Column by column difference",
                vMin=-5,
                vMax=-3,
                colorbarLabel="Log10[m]",
                filename=filename)

    print("RMS on parabola subtracted scan: {} m".format(np.ma.std(zz)))


def maskXYZ(x, y, z, n=512, guess=[60., 0., 0., 0., 0., 0.], bounds=None, radialMask=True, maskRadius=40., poly_order=2, **kwargs):
    """
    """
    
    xf = x[~np.isnan(z)]
    yf = y[~np.isnan(z)]
    zf = z[~np.isnan(z)]

    # Use masked arrays on the input data to avoid warnings.
    x = np.ma.masked_invalid(x)
    y = np.ma.masked_invalid(y)
    z = np.ma.masked_invalid(z)

    # Fit a parabola to the data.
    fitresult = fitLeicaData(xf, yf, zf, guess, bounds=bounds, weights=None)

    # Subtract the fitted parabola from the data.
    # The difference should be flat.
    c = fitresult.x
    zp = paraboloid(x, y, c[0])
    cor = np.hstack((-1*c[1:4],c[4:6],0))
    xrr, yrr, zrr = shiftRotateXYZ(x, y, z, cor)
    diff = zrr - zp

    if radialMask:
        xc = midPoint(xrr)
        yc = midPoint(yrr)
        mask = (((xrr - xc)**2. + (yrr - yc)**2.) < maskRadius**2.)
        mdiff = np.ma.masked_where(~mask, diff)
    else:
        mdiff = diff

    # Mask any pixels which deviate from the noise.
    # This should mask out retroreflectors, misaligned panels and sub-scans where the TLS moved due to wind.
    mcdiff = sigma_clip(mdiff)

    # Apply the mask to the original data and repeat once more.
    # In the end also fit a parabola to each row in the data, and mask outliers.
    # This allows to find more subtle features in the data.
    xf = np.ma.masked_where(mcdiff.mask, x)
    yf = np.ma.masked_where(mcdiff.mask, y)
    zf = np.ma.masked_where(mcdiff.mask, z)

    masked_fitresult = fitLeicaData(xf.compressed(), yf.compressed(), zf.compressed(),
                                    guess, bounds=bounds, weights=None)

    c = masked_fitresult.x
    zp = paraboloid(x, y, c[0])
    cor = np.hstack((-1*c[1:4],c[4:6],0))
    xrrm, yrrm, zrrm = shiftRotateXYZ(x, y, z, cor)
    masked_diff = zrrm - zp


    mcdiff2 = masked_diff

    # Final mask.
    map_mask = np.zeros((n,n), dtype=bool)

    xl = np.linspace(0,n,n)

    # Loop over rows fitting a parabola and masking any pixels that deviate from noise.
    for i in range(0,n):

        yl = mcdiff2[i]

        if len(xl[~yl.mask]) > 3:

            poly_c = np.polyfit(xl[~yl.mask], yl[~yl.mask], poly_order)
            poly_f = np.poly1d(poly_c)

            res = np.ma.masked_invalid(yl - poly_f(xl))
            res_sc = sigma_clip(res, **kwargs)

            map_mask[i] = res_sc.mask

        else:

            map_mask[i] = True

    # Prepare output.
    origData = (x, y, z)

    origMaskedData = (np.ma.masked_where(map_mask, x),
                      np.ma.masked_where(map_mask, y),
                      np.ma.masked_where(map_mask, z))

    rotatedData = (xrrm,
                   yrrm,
                   np.ma.masked_where(map_mask, zrrm))

    fitResidual = np.ma.masked_where(map_mask, zrrm - zp)

    parabolaFit = (xrrm, yrrm, zp)

    outData = {'origData': origData,
               'origMasked': origMaskedData,
               'rotated': rotatedData,
               'fitResidual': fitResidual,
               'parabolaFit': parabolaFit,
               'parabolaFitCoeffs': c}

    return outData


def maskLeicaData(filename, n=512, guess=[60., 0., 0., 0., 0., 0.], bounds=None, radialMask=True, maskRadius=40., **kwargs):
    """
    Given a GPU smoothed file, it will try and find bumps in the surface and mask them.

    :param filename: file with the GPU smoothed data.
    :param n: number of samples in the GPU smoothed data. Default n=512.
    :param guess: Initial guess for the parabola fit. [focus, x0, y0, z0, thetaX, thetaY].
    :param kwargs: keyword arguments passed to astropy.stats.sigma_clip.
    """

    orgData, cleanData = loadLeicaData(filename, n=n, numpy=False)

    xf = orgData[0]
    yf = orgData[1]
    zf = orgData[2]

    outData = maskXYZ(xf, yf, zf, n=n, guess=guess, bounds=bounds, radialMask=radialMask, maskRadius=maskRadius, **kwargs)

    return outData 


def maskRow(x, y, order=3, threshold=5):
    """
    Removes a polynomial from the 2D data and 
    applies sigma clipping to the residuals.

    
    """

    y_m = sigma_clip(y)
    yp = polyFitRANSAC(x, y_m, order)
    res = y - yp
    resm = sigma_clip(res, sigma=threshold)
    
    return resm


def maskDiff(diff, n=512, threshold=3):
    """
    Mask the first n rows of diff.
    Each row is masked using `maskRow`.

    Parameters
    ----------
    
    """
    
    x = np.linspace(0,n,n)

    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    for i in range(n):
        if diff[i].mask.sum() < n-10:
            diff[i] = maskRow(x, diff[i], order=3, threshold=threshold)
                
    return diff


def maskScanXYZ(x, y, z, n=512, outMaskThreshold=4, inMaskThreshold=3, inMaskClip=5):
    """
    """

    xg, yg, zg = regridXYZ(x, y, z, n=n)

    # Subtract the paraboloid from the primary reflector.
    diff, _ = subtractParaboloid(xg, yg, zg)

    # Mask points outside of the primary reflector.
    diff_m = sigma_clip(diff, outMaskThreshold)

    # Mask points on the primary reflector.
    diff_m = maskDiff(diff_m, n, threshold=inMaskThreshold)
    diff_m = sigma_clip(diff_m, inMaskClip)
    
    # Apply the masks to the original data.
    zg_4_pfit = np.ma.masked_where(diff_m.mask, zg)

    # Fit a paraboloid to the masked data.
    fit = paraboloidFit(xg, yg, zg_4_pfit)
    xga, yga, zga = align2Paraboloid(xg, yg, zg, fit.x)

    # Apply mask to the registered scan.
    zga = np.ma.masked_where(zg_4_pfit.mask, zga)

    return xga, yga, zga


def maskScan(fn, n=512, rot=0):
    """
    """

    orgData, cleanData = loadLeicaData(fn, n=None, numpy=False)
    x = orgData[0]
    y = orgData[1]
    z = orgData[2]
    xr, yr, zr = shiftRotateXYZ(x, y, z, [0, 0, 0, 0, 0, np.deg2rad(rot)])

    xga, yga, zga = maskScanXYZ(xr, yr, zr, n=512, outMaskThreshold=4, inMaskThreshold=3, inMaskClip=5)

    return xga, yga, zga
