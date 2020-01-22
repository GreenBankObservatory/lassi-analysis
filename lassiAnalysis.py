"""
This is a very high level module for integrating other modules and 
analyzing a single scan of Leica data once it's available.
"""

import os
import time
import numpy as np
from copy import copy
from shutil import copyfile
from scipy.interpolate import griddata
# import matplotlib
# matplotlib.use('agg')

from astropy.stats import sigma_clip

from zernikies import getZernikeCoeffs
from processPTX import processPTX, processNewPTX, aggregateXYZ
from gpus import smoothGPUs, smoothXYZGpu, loadLeicaDataFromGpus,  smoothGPUParallel
from parabolas import fitLeicaScan, imagePlot, surface3dPlot, radialReplace, loadLeicaData, fitLeicaData, \
                      newParabola, rotateData, parabola
from plotting import sampleXYZData, scatter3dPlot
from utils.utils import sph2cart, cart2sph, log, difflog, midPoint, gridLimits, splitXYZ
import plotting
import settings

# where is the code we'll be running?
GPU_PATH = settings.GPU_PATH
# where should we save the results from the GPU smoothing?
GPU_OUTPUT_PATH = settings.GPU_OUTPUT_PATH


def extractZernikesLeicaScanPair(refScanFile, sigScanFile, n=512, nZern=36, pFitGuess=[60., 0., 0., -50., 0., 0.], rMaskRadius=49.):
    """
    Takes two smoothed Leica scans and extracts Zernike coefficients from their difference, reference - signal.
    :param refScanFile: File with the smoothed reference scan. <scan_name>.ptx.csv
    :param sigScanFile: File with the smoothed signal scan. <scan_name>.ptx.csv
    :param n: Number of elements per side in the smoothed data.
    :param nZern: Number of Zernike polynomials to include in the fit. Can be up to 120.
    :param pFitGuess: Initial guess for ther parabola fit.
    :param rMaskRadius: Radius of the radial mask.
    """

    if rMaskRadius <= 0.:
        radialMask = False
    else:
        radialMask = True

    print('Masking file: {}'.format(refScanFile))
    ref_data = maskLeicaData(refScanFile, n=n, guess=pFitGuess, radialMask=radialMask, maskRadius=rMaskRadius)

    # Extract the data we will use.
    xrr, yrr, zrr = ref_data['rotated']
    cr = ref_data['parabolaFitCoeffs']

    print('Masking file: {}'.format(sigScanFile))
    sig_data = maskLeicaData(sigScanFile, n=n, guess=pFitGuess, radialMask=radialMask, maskRadius=rMaskRadius)
    
    xs, ys, zs = sig_data['origMasked']
     
    # Rotate the signal scan.
    xsr, ysr, zsr = rotateData(xs, ys, zs, cr[4], cr[5])
    xsr.shape = ysr.shape = zsr.shape = (n,n)
    zsr = np.ma.masked_where(zs.mask, zsr)

    # The data has been rotated, but we haven't applied the shifts.
    xrr = xrr - cr[1]
    yrr = yrr - cr[2]
    zrr = zrr - cr[3]
    xsr = xsr - cr[1]
    ysr = ysr - cr[2]
    zsr = zsr - cr[3]

    # Find the grid limits for the reference and signal scans.
    xmin, xmax = gridLimits(xrr, xsr)
    ymin, ymax = gridLimits(yrr, ysr)

    xrrg, yrrg, zrrg = regridXYZMasked(xrr, yrr, zrr, n=n, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    xsrg, ysrg, zsrg = regridXYZMasked(xsr, ysr, zsr, n=n, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    # Surface deformation map: reference - signal
    diff = zrrg - zsrg
    diff = diff[::-1] # Flip the signal scan. Required to conform with the way we handle the data.
                      # This should not be required if we handled the data consistently throughout.

    # Find Zernike coefficients on the surface deformation map.
    fitlist = getZernikeCoeffs(diff.filled(0), nZern, barChart=False, norm='active-surface')

    return xsrg, ysrg, diff, fitlist


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
                     addOffset=False,
                     plot=True):
    """
    High level function for processing leica data:
       * processes PTX file
       * smoothes it by calling gpu code
       * fits parabolas to data
       * regrids final data
    Final processed scan is ready for difference between
    this and a ref or signal scan.   
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
    
    xyz, dts = processNewPTX(fpath,
                             rot=rot,
                             ellipse=ellipse,
                             rFilter=True,
                             iFilter=False,
                             sampleSize=sampleSize,
                             addOffset=addOffset)

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
        x.shape = (N,N)
        y.shape = (N,N)
        z.shape = (N,N)
        masked = maskXYZ(x, y, z, n=N, guess=[60., 0., 0., -49., 0., 0.], bounds=None, radialMask=True, maskRadius=49.)
        xx, yy, zz = regridXYZ(x, y, masked['fitResidual'], n=N)
        plotting.surfacePlot(xx, yy, np.log10(abs(np.diff(zz.T))), title="Pixel by pixel difference", vMin=-5, vMax=0, colorbarLabel="Log10[m]")
        print("RMS on parabola subtracted scan: {} m".format(np.ma.std(zz)))

def loadProcessedData(filename):
    "Loads the results of processLeicaScan from file"
    d = np.load(filename)
    return d["xs"], d["ys"], d["diffs"], d["retroMask"]


def maskXYZ(x, y, z, n=512, guess=[60., 0., 0., 0., 0., 0.], bounds=None, radialMask=True, maskRadius=40., **kwargs):
    """
    """
    
    xf = x[~np.isnan(x)]
    yf = y[~np.isnan(x)]
    zf = z[~np.isnan(x)]

    # Fit a parabola to the data.
    fitresult = fitLeicaData(xf, yf, zf, guess, bounds=bounds, weights=None)

    # Subtract the fitted parabola from the data.
    # The difference should be flat.
    c = fitresult.x
    newX, newY, newZ = newParabola(x, y, z, c[0], c[1], c[2], c[3], c[4], c[5])
    newX.shape = newY.shape = newZ.shape = (n, n)
    xrr, yrr, zrr = rotateData(x, y, z, c[4], c[5])
    xrr.shape = yrr.shape = zrr.shape = (n, n)
    diff = zrr - newZ

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
    newXm, newYm, newZm = newParabola(x, y, z, c[0], c[1], c[2], c[3], c[4], c[5])
    newXm.shape = newYm.shape = newZm.shape = (n, n)
    xrrm, yrrm, zrrm = rotateData(x, y, z, c[4], c[5])
    xrrm.shape = yrrm.shape = zrrm.shape = (n, n)
    masked_diff = zrrm - newZm

    if radialMask:
        xc = midPoint(xrrm)
        yc = midPoint(yrrm)
        mask = (((xrrm - xc)**2. + (yrrm - yc)**2.) < maskRadius**2.)
        masked_diff = np.ma.masked_where(~mask, masked_diff)
    else:
        masked_diff = np.ma.masked_invalid(masked_diff)

    mcdiff2 = masked_diff

    # Final mask.
    map_mask = np.zeros((n,n), dtype=bool)

    xl = np.linspace(0,n,n)

    # Loop over rows fitting a parabola and masking any pixels that deviate from noise.
    for i in range(0,n):

        yl = mcdiff2[i]

        if len(xl[~yl.mask]) > 3:

            poly_c = np.polyfit(xl[~yl.mask], yl[~yl.mask], 2)
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

    fitResidual = np.ma.masked_where(map_mask, zrrm - newZm)

    parabolaFit = (newXm, newYm, newZm)

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


def regridXYZ(x, y, z, n=512., verbose=False, xmin=False, xmax=False, ymin=False, ymax=False, method='linear'):
    """
    Regrids the XYZ data to a regularly sampled grid.

    :param x: vector with the x coordinates.
    :param y: vector with the y coordinates.
    :param z: vector with the z coordinates.
    :param n: number of samples in the grid.
    :param verbose: verbose output?
    """
    
    # Set the grid limits.
    if not xmin:
        xmin = np.nanmin(x)
    if not xmax:
        xmax = np.nanmax(x)
    if not ymin:
        ymin = np.nanmin(y)
    if not ymax:
        ymax = np.nanmax(y)

    if verbose:
        print("Limits: ", xmin, xmax, ymin, ymax)

    # Set the grid spacing.
    dx = (xmax - xmin)/n
    dy = (ymax - ymin)/n
    
    # Make the grid.
    grid_xy = np.mgrid[xmin:xmax:dx,
                       ymin:ymax:dy]
    if verbose:
        print("New grid shape: ", grid_xy[0].shape)

    # Regrid the data.
    reg_z = griddata(np.array([x[~np.isnan(z)].flatten(),y[~np.isnan(z)].flatten()]).T, 
                     z[~np.isnan(z)].flatten(), 
                     (grid_xy[0], grid_xy[1]), method=method, fill_value=np.nan)

    # We need to flip the reggrided data in the abscisa axis 
    # so that it has the same orientation as the input.
    return grid_xy[0], grid_xy[1], reg_z.T


def regridXYZMasked(x, y, z, n=512, verbose=False, xmin=False, xmax=False, ymin=False, ymax=False):
    """
    """

    outMask = np.ma.masked_invalid(x).mask

    xReg,yReg,zReg = regridXYZ(x[~outMask],
                               y[~outMask],
                               z[~outMask],
                               n=n, verbose=verbose,
                               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    _,_,retroMask = regridXYZ(x[~outMask],
                              y[~outMask],
                              z.mask.astype(float)[~outMask],
                              n=n, verbose=verbose,
                              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    zRegMasked = np.ma.masked_where(retroMask, zReg)

    return xReg, yReg, zRegMasked

