from copy import copy
import random

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

def barChartPlot(index, fitlist):
    """
    Plots a bar chart with Zernike coefficients.
    This is used in zernikies.getZernikeCoeffs.

    :param index: List with the index of the Zernike polynomials to show.
    :param fitlist: List with the coefficients of the Zernike polynomials.
    """

    fig = plt.figure(figsize=(9, 6), dpi=80)
    xticklist = []
    width = 0.6
    for i in index:
        xticklist.append('Z'+str(i))
    barfigure = plt.bar(index, fitlist, width, color='#2E9AFE', edgecolor='#2E9AFE')
    plt.xticks(index+width//2, xticklist, rotation=90)
    plt.xlabel('Zernike polynomials', fontsize=18)
    plt.ylabel('Coefficient', fontsize=18)
    plt.title('Fitted Zernike polynomial coefficients', fontsize=18)

def zernikeResiduals2DPlot(xx, yy, zz):
    """
    Plots the residuals of a Zernike fit.
    This is used in zernikies.getZernikeCoeffs.
    """

    fig = plt.figure(figsize=(9, 6), dpi=80)
    ax = fig.gca()
    im = plt.pcolormesh(xx, yy, zz, cmap=plt.get_cmap('RdYlGn'))
    plt.colorbar()
    plt.title('Remaining Aberration', fontsize=18)
    ax.set_aspect('equal', 'datalim')

def imagePlot(z, title):
    fig = plt.figure()
    ax = fig.gca()
    #cax = ax.imshow(np.log(np.abs(np.diff(zrr - newZ))))
    cax = ax.imshow(z)
    fig.colorbar(cax)
    plt.title(title)

def surface3dPlot(x, y, z, title, xlim=None, ylim=None, sample=None):
    fig = plt.figure()
    ax = Axes3D(fig)

    # plot all the data, or just some?
    if sample is not None:
        print("Plotting %5.2f percent of data" % sample)
        x, y, z = sampleXYZData(x, y, z, sample)

    ax.plot_surface(x, y, z)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

def scatter3dPlot(x, y, z, title, xlim=None, ylim=None, sample=None):

    # plot all the data, or just some?
    if sample is not None:
        print("Plotting %5.2f percent of data" % sample)
        x, y, z = sampleXYZData(x, y, z, sample)
        print("Now length of data is %d" % len(x))


    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)    


def sampleXYZData(x, y, z, samplePercentage, seed=None):
    "Return a random percentage of the data"

    assert len(x) == len(y)
    assert len(y) == len(z)

    if seed is not None:
        random.seed(seed)

    lenx = len(x)

    sampleSize = int((lenx * samplePercentage) / 100.)

    idx = random.sample(range(lenx), sampleSize)

    return copy(x[idx]), copy(y[idx]), copy(z[idx])
