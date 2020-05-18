"""
Functions to grid the data.
"""

import numpy as np

from scipy.interpolate import griddata


def regridXYZ(x, y, z, n=512., verbose=False, xmin=False, xmax=False, ymin=False, ymax=False, method='linear'):
    """
    Regrids the XYZ data to a regularly sampled cartesian grid.

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

    x_ = x[~np.isnan(z)].flatten()
    y_ = y[~np.isnan(z)].flatten()
    points = np.vstack((x_,y_)).T
    z_ = z[~np.isnan(z)].flatten()

    # Regrid the data.
    reg_z = griddata(points, z_, (grid_xy[0], grid_xy[1]), 
                     method=method, fill_value=np.nan)

    return grid_xy[0], grid_xy[1], reg_z


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
