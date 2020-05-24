"""
Functions to grid the data.
"""

import numpy as np

from scipy.interpolate import griddata


def regridXYZ(x, y, z, n=512., verbose=False, xmin=False, xmax=False, ymin=False, ymax=False, method='linear'):
    """
    Regrids a scan to a regularly sampled square cartesian grid.
    
    Parameters
    ----------
    x : array
        Array with the x coordinates.
    y : array
        Array with the y coordinates.
    z : array
        Array with the z coordinates.
    n : int, optional
        Number of pixels per side of the grid.
        The grid will have `n`x`n` pixels.
    verbose : bool, optional
        Verbose output?
    xmin : float, optional
        Minimum x value of the grid. 
        Defaults to the minimum valid value of `x`.
    xmax : float, optional
        Maximum x value of the grid. 
        Defaults to the maximum valid value of `x`.
    ymin : float, optional
        Minimum y value of the grid. 
        Defaults to the minimum valid value of `y`.
    ymax : float, optional
        Maximum y value of the grid. 
        Defaults to the maximum valid value of `y`.
    method : {‘linear’, ‘nearest’, ‘cubic’}, optional
        See `scipy.interpolate.griddata` for details.

    Returns
    -------
    tuple
        Tuple with the x coordinates of the grid, 
        the y coordinates and the z values.
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


def regridXYZMasked(x, y, z, n=512, verbose=False, xmin=False, xmax=False, ymin=False, ymax=False, method='linear'):
    """
    Regrids a scan to a regularly sampled square cartesian grid.
    It also regrid the mask to the new grid.
    It calls `regridXYZ` twice, once for the data, and once for the mask.
    
    Parameters
    ----------
    x : array
        Array with the x coordinates.
    y : array
        Array with the y coordinates.
    z : array
        Array with the z coordinates.
    n : int, optional
        Number of pixels per side of the grid.
        The grid will have `n`x`n` pixels.
    verbose : bool, optional
        Verbose output?
    xmin : float, optional
        Minimum x value of the grid. 
        Defaults to the minimum valid value of `x`.
    xmax : float, optional
        Maximum x value of the grid. 
        Defaults to the maximum valid value of `x`.
    ymin : float, optional
        Minimum y value of the grid. 
        Defaults to the minimum valid value of `y`.
    ymax : float, optional
        Maximum y value of the grid. 
        Defaults to the maximum valid value of `y`.
    method : {‘linear’, ‘nearest’, ‘cubic’}, optional
        See `scipy.interpolate.griddata` for details.

    Returns
    -------
    tuple
        Tuple with the x coordinates of the grid, 
        the y coordinates and the z values.
    """

    outMask = np.ma.masked_invalid(x).mask

    xReg,yReg,zReg = regridXYZ(x[~outMask],
                               y[~outMask],
                               z[~outMask],
                               n=n, verbose=verbose,
                               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                               method=method)

    _,_,retroMask = regridXYZ(x[~outMask],
                              y[~outMask],
                              z.mask.astype(float)[~outMask],
                              n=n, verbose=verbose,
                              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                              method=method)

    zRegMasked = np.ma.masked_where(retroMask, zReg)

    return xReg, yReg, zRegMasked
