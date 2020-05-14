"""
Functions to grid the data.
"""

import numpy as np

from scipy.interpolate import griddata


def regridXYZ(x, y, z, n=512., verbose=False, xmin=False, xmax=False, ymin=False, ymax=False, method='linear'):
    """
    Regrids the XYZ data to a regularly sampled cartesian grid.

    :param x: Vector with the x coordinates.
    :param y: Vector with the y coordinates.
    :param z: Vector with the z coordinates.
    :param n: Number of samples in the grid.
    :param verbose: Verbose output?
    :param xmin: Minimum x value of the grid.
    :param xmax: Maximum x value of the grid.
    :param ymin: Minimum y value of the grid.
    :param ymax: Maximum y value of the grid.
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
    Regrids masked XYZ data to a regularly sampled cartesian grid.

    :param x: Vector with the x coordinates.
    :param y: Vector with the y coordinates.
    :param z: Vector with the z coordinates.
    :param n: Number of samples in the grid.
    :param verbose: Verbose output?
    :param xmin: Minimum x value of the grid.
    :param xmax: Maximum x value of the grid.
    :param ymin: Minimum y value of the grid.
    :param ymax: Maximum y value of the grid.
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
