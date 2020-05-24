import matplotlib
#matplotlib.use('agg')

import random
import numpy as np
import matplotlib.pylab as plt

from copy import copy
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from grid import regridXYZ
from simulateSignal import zernikePoly
from utils.utils import midPoint, sampleXYZData


class MidpointNormalize(matplotlib.colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work 
    their way either side from a prescribed midpoint value.


    Examples
    --------

    >>> import pylab as plt
    >>> array = [[50,100,50],[-100,-50,-100],[50,100,50]]
    >>> im = plt.imshow(array, norm=MidpointNormalize(midpoint=0., vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)


    def __call__(self, value, clip=None):

        if clip is None:
            clip = self.clip

        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]

        result, is_scalar = self.process_value(value)

        result = np.ma.array(np.interp(value, x, y), mask=result.mask, fill_value=np.nan, copy=False)

        if is_scalar:
            result = result[0]

        return result 


def surfacePlot(x, y, z, title=None, midPoint=0, vMin=-5e-3, vMax=5e-3, colorbarLabel=None, filename=None):
    """
    Plot a surface using `imshow`. The data should be a 2-D ndarray with square pixels.

    Parameters
    ----------
    x : 2-D ndarray
        Array with x coordinates of the surface.
        Used to set the labels of the x ticks.
        It assumes the units are meters.
    y : 2-D ndarray
        Array with y coordinates of the surface.
        Used to set the labels of the y ticks.
        It assumes the units are meters.
    z : 2-D ndarray
        Array with z coordinates of the surface.
    title : string, optional
        Text to display as plot title.
    midPoint : float, optional
        Value used as middle of the colorbar.
    vMin : float, optional
        Value used as minimum of the colorbar.
    vMax : float, optional
        Value used as maximum of the colorbar.
    colorbarLabel : string, optional
        Text for the colorbar.
    filename : string, optional
        Save the plot to this file.
    """

    cmap = plt.matplotlib.cm.coolwarm
    cmap.set_bad(color='white')

    extent = [np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)]

    fig = plt.figure(figsize=(6, 5), dpi=150, frameon=False)
    ax = fig.add_subplot(111)

    norm = MidpointNormalize(midpoint=midPoint, vmin=vMin, vmax=vMax)
    im = ax.imshow(z, extent=extent, vmin=vMin, vmax=vMax, norm=norm, cmap=cmap, origin='lower')

    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    if colorbarLabel is not None:
        cb.ax.set_ylabel(colorbarLabel)

    ax.minorticks_on()
    ax.tick_params('both', which='both', direction='in', top=True, right=True, bottom=True, left=True)

    ax.set_xlabel('x axis (m)')
    ax.set_ylabel('y axis (m)')
    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.06)
        

def barChartPlot(index, coefficients, expected=[]):
    """
    Plots a bar chart with Zernike coefficients, `:math: C_{i}`.
    
    Parameters
    ----------
    index : list of ints
        Indices of the polynomial coefficients.
    coefficients : array
        Coefficients of the Zernike polynomials.
    expected : array, optional
        Expected values of the coefficients.
    """

    fig = plt.figure(figsize=(9, 6), dpi=80)
    xticklist = []
    width = 0.6
    # Prepare the labels of the  x ticks.
    for i in index:
        xticklist.append('Z'+str(i))
    barfigure = plt.bar(index, coefficients, width, color='#2E9AFE', edgecolor='#2E9AFE', label='Measured')
    if len(expected) > 0:
        plt.bar(index-width/3, expected, width, color='#882255', edgecolor='#882255', label='Input', alpha=0.5)

    # Add a legend.
    plt.legend(loc=0, fancybox=True)

    # Set the ticks and their labels.
    plt.xticks(index+width//2, xticklist, rotation=90)
    plt.xlabel('Zernike polynomials', fontsize=18)
    plt.ylabel('Coefficient', fontsize=18)
    plt.title('Fitted Zernike polynomial coefficients', fontsize=18)

    plt.gca().minorticks_on()
    plt.gca().tick_params('both', direction='in', top=True, right=True)
    plt.gca().tick_params('y', which='minor', direction='in', left=True, right=True)
    plt.gca().tick_params('x', which='minor', bottom=False)


def zernikeResiduals2DPlot(xx, yy, zz):
    """
    Plots the residuals of a Zernike fit.
    """

    fig = plt.figure(figsize=(9, 6), dpi=80)
    ax = fig.gca()
    im = plt.pcolormesh(xx, yy, zz, cmap=plt.get_cmap('RdYlGn'))
    plt.colorbar()
    plt.title('Remaining Aberration', fontsize=18)
    ax.set_aspect('equal', 'datalim')


#def linePlot(y, title):
#    fig = plt.figure()
#    ax = fig.gca()
#    ax.plot(range(len(y)), y)
#    plt.title(title)


def imagePlot(z, title):
    """
    """
    
    fig = plt.figure()
    ax = fig.gca()
    cax = ax.imshow(z)
    fig.colorbar(cax)
    plt.title(title)


def surface3dPlot(x, y, z, title=None, xlim=None, ylim=None, sample=None):
    """
    Plot a surface in 3D using a mesh.

    Parameters
    ----------
    x : 2-D array
    y : 2-D array
    z : 2-D array
    sample : float between 0 and 1, optional
        Percentage of the data to be displayed.
    """

    fig = plt.figure()
    ax = Axes3D(fig)

    # plot all the data, or just some?
    if sample is not None:
        print("Plotting %5.2f percent of data" % sample)
        x, y, z = sampleXYZData(x, y, z, sample)

    ax.plot_surface(x, y, z)
    plt.xlabel("x")
    plt.ylabel("y")
    
    if title is not None:
        plt.title(title)

    if xlim is not None:
        plt.xlim(xlim)
    
    if ylim is not None:
        plt.ylim(ylim)


def scatter3dPlot(x, y, z, title=None, xlim=None, ylim=None, sample=None, fig=None, axes=None, color=None):
    """
    Plot a surface in 3D using points.

    Parameters
    ----------
    x : 2-D array
    y : 2-D array
    z : 2-D array
    xlim : tuple, optional
        Limits for the display in the x coordinate.
    ylim L tuple, optional
        Limits for the display in the y coordinate.
    sample : float between 0 and 1, optional
        Percentage of the data to be displayed.
    fig : figure object, optional
        Display the points in this Figure.
    axes : axes object, optional
        Display the points in this Axes.
    color : color, sequence, or sequence of colors, optional

    Returns
    -------
    tuple
        Tuple with the Figure and Axes objects used for the plot.
    """

    # plot all the data, or just some?
    if sample is not None:
        print("Plotting %5.2f percent of data" % sample)
        x, y, z = sampleXYZData(x, y, z, sample)
        print("Now length of data is %d" % len(x))

    if fig is None:
        fig = plt.figure()
    
    if axes is None:
        axes = Axes3D(fig)
    if color is None:
        color = 'b'

    axes.scatter(x, y, z, c=color)
    
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    
    if title is not None:
        plt.title(title)
    
    if xlim is not None:
        axes.set_xlim(xlim)
    
    if ylim is not None:
        axes.set_ylim(ylim)    
    
    return fig, axes


def scatterPlot(x, y, z=None, title=None, xlim=None, ylim=None, sample=None):
    """
    Display a surface using a scatter plot in 2D.

    Parameters
    ----------
    
    """
    if z is None:
        z = np.ones(x.shape, dtype=np.int)

    # plot all the data, or just some?
    if sample is not None:
        print("Plotting %5.2f percent of data" % sample)
        x, y, z = sampleXYZData(x, y, z, sample)
        print("Now length of data is %d" % len(x))

    fig = plt.figure()
    
    ax = fig.gca()
    
    sc = ax.scatter(x, y, c=z)
    plt.colorbar(sc)
    
    plt.xlabel("x")
    plt.ylabel("y")

    if title is not None:
        plt.title(title)
    
    if xlim is not None:
        plt.xlim(xlim)
    
    if ylim is not None:
        plt.ylim(ylim)


def sampleXYZData(x, y, z, samplePercentage, seed=None):
    """
    Return a percentage of the data using a random sample.

    Parameters
    ----------
    x : 1-D array
    y : 1-D array
    x : 1-D array
    samplePercentage : float between 0 and 1
        Percentage of the data to return.
    seed : int, optional
        Initialize the random number generator.
        Use a fixed value for reproducible results.
    
    Returns
    -------
    tuple
        Tuple with a random sample of x, y and z.
    """

    assert len(x) == len(y)
    assert len(y) == len(z)

    if seed is not None:
        random.seed(seed)

    lenx = len(x)

    sampleSize = int((lenx * samplePercentage) / 100.)

    idx = random.sample(range(lenx), sampleSize)

    return copy(x[idx]), copy(y[idx]), copy(z[idx])


def plotZernikes(x, y, zernCoeffs, n=512, title=None, filename=None):
    """
    Plot Zernike polynomials with coefficients `zernCoeffs` over a square grid.

    Parameters
    ----------
    x : array
        Array with x coordinates.
        Used to evaluate the Zernike polynomials.
    y : array
        Array with y coordinates.
        Used to evaluate the Zernike polynomials.
    zernCoeffs : array
        Array with the coefficients of a Zernike polynomial.
    n : int, optional
        Number of pixels per side of the square grid used to
        display the Zernike polynomial.
    title : string, optional
    filename : string, optional
        Save the plot to this disk location.
    """

    # Create the linear combination of the Zernike polynomials.
    zPoly = zernikePoly(x, y, midPoint(x), midPoint(y), zernCoeffs)
    
    # Grid it to a regularly sampled cartesian grid.
    xx, yy, zz = regridXYZ(x, y, zPoly, n=n)

    # Make it look like the dish of the GBT by selecting a circular area.
    mask = (((xx - midPoint(xx))**2. + (yy - midPoint(yy))**2.) < 49.**2.)
    zz[~mask] = np.nan

    # To get meaningful plot tick labels.
    extent = [np.nanmin(xx), np.nanmax(xx), np.nanmin(yy), np.nanmax(yy)]

    fig = plt.figure(figsize=(12, 8), dpi=80)
    ax = fig.gca()
    im = ax.imshow(zz.T, cmap=cm.RdYlGn, extent=extent)
    plt.colorbar(im)

    # Add minor ticks and make them point inwards.
    ax.minorticks_on()
    ax.tick_params('both', which='both', direction='in', top=True, right=True, bottom=True, left=True)
    
    # Set a title.
    if title is not None:
        plt.title(title)

    # Set axis label names.
    ax.set_xlabel('x axis (m)')
    ax.set_ylabel('y axis (m)')

    # Save the figure to disk.
    if filename is not None:
        plt.savefig(filename)


def nicePlotDates(labels, i0=0):
    """
    Given a list of text labels representing dates with the format %m/%d %H:%M
    it will update the list to keep only the month and day portion when it
    changes.

    Parameters
    ----------
    labels : list 
        List with the text labels formatted as %m/%d %H:%M.
    i0 : int, optional
        Index of the first text label to keep in full.
        Useful when the index 0 text label does not appear in the plot
    
    Examples
    --------

    >>> labels = ['06/11 15:00', '06/11 18:00']
    >>> new_labels = nicePlotDates(labels)
    >>> new_labels
    ['06/11 15:00', '18:00']
    """

    new_labels = []
    month_ = None
    day_ = None
    for i,label in enumerate(labels):
        date, time = label.split(' ')
        month, day = date.split('/')
        if i == i0:
            month_ = month
            day_ = day
            new_labels.append(label)
        if i != i0:
            if month != month_:
                month_ = month
                new_labels.append(label)
            elif day != day_:
                day_ = day
                new_labels.append(label)
            else:
                new_labels.append(time)

    return new_labels
