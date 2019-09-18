from copy import copy

import numpy as np
#import opticspy

from plotting import barChartPlot
from zernikeIndexing import noll2asAnsi, printZs

nMax = 36
zernikeNorm = { 1: 1.,
                2: 4.,
                3: 4.,
                4: 6.,
                5: 3.,
                6: 6.,
                7: 8.,
                8: 8.,
                9: 8.,
               10: 8.,
               11: 10.,
               12: 10.,
               13: 5.,
               14: 10.,
               15: 10.,
               16: 12.,
               17: 12.,
               18: 12.,
               19: 12.,
               20: 12.,
               21: 12.,
               22: 14.,
               23: 14.,
               24: 14.,
               25: 7.,
               26: 14.,
               27: 14.,
               28: 14.,
               29: 4.,
               30: 4.,
               31: 4.,
               32: 4.,
               33: 4.,
               34: 4.,
               35: 4.,
               36: 4.}


def _single_term_cos(power, rho, theta):
    """
    Utility function to compute Zernike polynomials.
    """

    return np.power(rho, power) * np.cos(power * theta)

def _single_term_sin(power, rho, theta):
    """
    Utility function to compute Zernike polynomials.
    """

    return np.power(rho, power) * np.sin(power * theta)

def _multi_term_cos(coef, power, multiplier, rho, theta):
    """
    Utility function to compute Zernike polynomials.
    """
    
    power2 = (coef + 1.) * -1.
    return ((coef * np.power(rho, power2) + (power * np.power(rho, power))) * np.cos(multiplier * theta))

def _multi_term_sin(coef, power, multiplier, rho, theta):
    """
    Utility function to compute Zernike polynomials.
    """
    
    power2 = (coef + 1.) * -1.
    return ((coef * np.power(rho, power2) + (power * np.power(rho, power))) * np.sin(multiplier * theta))

def zernikePolar(coefficients, rho, theta):
    """
    Zernike polynomials in polar coordinates following the Active Surface indexing.

    :param coefficients: Coefficients for the Zernike polynomials.
    :param rho: Radial coordinate.
    :param theta: Azimutal coordinate.
    :return: 


    """
    
    z = coefficients

    z1  = z[1+0]  * 1. # Piston
    z2  = z[1+1]  * _single_term_cos(1., rho, theta) # 
    z3  = z[1+2]  * _single_term_sin(1., rho, theta)
    z4  = z[1+3]  * _single_term_cos(2., rho, theta)
    z5  = z[1+4]  * (2.*rho**2. - 1.)
    z6  = z[1+5]  * _single_term_sin(2., rho, theta)
    z7  = z[1+6]  * _single_term_cos(3., rho, theta)
    z8  = z[1+7]  * _multi_term_cos(-2., 3., 1., rho, theta)
    z9  = z[1+8]  * _multi_term_sin(-2., 3., 1., rho, theta)
    z10 = z[1+9]  * _single_term_sin(3., rho, theta)
    z11 = z[1+10] * _single_term_cos(4., rho, theta)
    z12 = z[1+11] * _multi_term_cos(-3., 4., 2., rho, theta)
    z13 = z[1+12] * (1. - (6.*np.power(rho, 2.))  + (6.*np.power(rho, 4.)))
    z14 = z[1+13] * _multi_term_sin(-3., 4., 2., rho, theta)
    z15 = z[1+14] * _single_term_sin(4., rho, theta)
    z16 = z[1+15] * _single_term_cos(5., rho, theta)
    z17 = z[1+16] * _multi_term_cos(-4., 5., 3., rho, theta)
    z18 = z[1+17] * (3.*rho - 12.*np.power(rho, 3.) + 10.*np.power(rho, 5.)) * np.cos(theta)
    z19 = z[1+18] * (3.*rho - 12.*np.power(rho, 3.) + 10.*np.power(rho, 5.)) * np.sin(theta)
    z20 = z[1+19] * _multi_term_sin(-4., 5., 3., rho, theta)
    z21 = z[1+20] * _single_term_sin(5., rho, theta)
    z22 = z[1+21] * _single_term_cos(6., rho, theta)
    z23 = z[1+22] * _multi_term_cos(-5., 6., 4., rho, theta)
    z24 = z[1+23] * (((6.*np.power(rho, 2.)) - (20.*np.power(rho, 4.)) + (15.*np.power(rho, 6.))) * np.cos(2.*theta))
    z25 = z[1+24] * (-1. + (12.*np.power(rho, 2.)) - (30.*np.power(rho, 4.)) + (20.*np.power(rho, 6.)))
    z26 = z[1+25] * (((6.*np.power(rho, 2.)) - (20.*np.power(rho, 4.)) + (15.*np.power(rho, 6.))) * np.sin(2.*theta))
    z27 = z[1+26] * _multi_term_sin(-5., 6., 4., rho, theta)
    z28 = z[1+27] * _single_term_sin(6., rho, theta)
    z29 = z[1+28] * _single_term_cos(7., rho, theta)
    z30 = z[1+29] * _multi_term_cos(-6., 7., 5., rho, theta)
    z31 = z[1+30] * (((10.*np.power(rho, 3.)) - (30.*np.power(rho, 5.)) + (21.*np.power(rho, 7.))) * np.cos(3.*theta))
    z32 = z[1+31] * (((-4.*np.power(rho, 1.)) + (30.*np.power(rho, 3.)) - (60.*np.power(rho, 5.)) + (35.*np.power(rho, 7.)))*np.cos(theta))
    z33 = z[1+32] * (((-4.*np.power(rho, 1.)) + (30.*np.power(rho, 3.)) - (60.*np.power(rho, 5.)) + (35.*np.power(rho, 7.)))*np.sin(theta))
    z34 = z[1+33] * (((10.*np.power(rho, 3.)) - (30.*np.power(rho, 5.)) + (21.*np.power(rho, 7.))) * np.sin(3.*theta))
    z35 = z[1+34] * _multi_term_sin(-6., 7., 5., rho, theta)
    z36 = z[1+35] * _single_term_sin(7., rho, theta)

    z_tot = z1 + z2 + z3+  z4 + z5 + z6 + z7 + z8 + z9 + \
            z10+ z11+ z12+ z13+ z14+ z15+ z16+ z17+ z18+ z19+ \
            z20+ z21+ z22+ z23+ z24+ z25+ z26+ z27+ z28+ z29+ \
            z30+ z31+ z32+ z33+ z34+ z35+ z36

    return z_tot

def getZernikeCoeffs(surface, order, plot2D=False, barChart=False, printReport=True):
    """

    Determines the coefficients of Zernike polynomials that best describe a surface.

    :param surface: The surface where Zernike coefficients will be determined.
    :param order: How many order of Zernike polynomials you want to incorporate. Less than 37.
    :param printReport: Print the determined Zernike coefficients?
    :param barChart: Plot a bar chart with the Zernike coefficients?
    :param plot2D: Plot the residuals of the surface and the best fit Zernike polynomials?
    :return: n-th Zernike coefficients describing a surface.
    """

    if order > nMax:
        raise ValueError('order must be less than {}.'.format(nMax+1))

    coeffs = []
    # The active surface starts counting from Z1.
    coeffs.append(0)

    # Make the support to evaluate the Zernike polynomials.
    l = len(surface)
    x = np.linspace(-1., 1., l)
    y = np.linspace(-1., 1., l)
    [xx,yy] = np.meshgrid(x, y)
    r = np.sqrt(xx**2. + yy**2.)
    u = np.arctan2(yy, xx)

    # Loop over Zernike polynomials and use their orthogonality to determine their coefficients.
    for i in range(1,order+1):
        c = [0]*i + [1] + [0]*(37-i-1)
        zf = zernikePolar(c, r, u)
        mask = (r > 1)
        zf[mask] = 0
        a = np.sum(surface*zf)*2.*2./l/l/np.pi*zernikeNorm[i]
        coeffs.append(a)

    if plot3D or plot2D:
        # Compute the residuals.
        z_new = surface - zernikePolar(coeffs, r, u)
        z_new[mask] = 0

    # Plot bar chart of Zernike coefficients.
    if barChart == True:
        fitlist = coeffs[1:order+1]
        index = np.arange(1,order+1)
        barChartPlot(index, fitlist)

    if plot2D:
        zernikeResiduals2DPlot(xx, yy, z_new)

    if printReport:
        zernikePrint(coeffs)

    return coeffs

def zernikePrint(z):
    """
    Print a table with the coefficients of Zernike polynomials.
    """

    print("                            Zernike coefficients list (microns)                   ")
    print("----------------------------------------------------------------------------------")
    print("|   Z1  |   Z2  |   Z3  |   Z4  |   Z5  |   Z6  |   Z7  |   Z8  |   Z9  |  Z10  |")
    print("----------------------------------------------------------------------------------")
    print("|{0:^7.3f}|{1:^7.3f}|{2:^7.3f}|{3:^7.3f}|{4:^7.3f}|{5:^7.3f}|{6:^7.3f}|{7:^7.3f}|{8:^7.3f}|{9:^7.3f}|".format\
            (z[1],z[2],z[3],z[4],z[5],z[6],z[7],z[8],z[9],z[10]))
    print("----------------------------------------------------------------------------------")
    print("|  Z11  |  Z12  |  Z13  |  Z14  |  Z15  |  Z16  |  Z17  |  Z18  |  Z19  |  Z20  |")
    print("----------------------------------------------------------------------------------")
    print("|{0:^7.3f}|{1:^7.3f}|{2:^7.3f}|{3:^7.3f}|{4:^7.3f}|{5:^7.3f}|{6:^7.3f}|{7:^7.3f}|{8:^7.3f}|{9:^7.3f}|".format\
            (z[11],z[12],z[13],z[14],z[15],z[16],z[17],z[18],z[19],z[20]))
    print("----------------------------------------------------------------------------------")
    print("|  Z21  |  Z22  |  Z23  |  Z24  |  Z25  |  Z26  |  Z27  |  Z28  |  Z29  |  Z30  |")
    print("----------------------------------------------------------------------------------")
    print("|{0:^7.3f}|{1:^7.3f}|{2:^7.3f}|{3:^7.3f}|{4:^7.3f}|{5:^7.3f}|{6:^7.3f}|{7:^7.3f}|{8:^7.3f}|{9:^7.3f}|".format\
            (z[21],z[22],z[23],z[24],z[25],z[26],z[27],z[28],z[29],z[30]))
    print("----------------------------------------------------------------------------------")
    print("|  Z31  |  Z32  |  Z33  |  Z34  |  Z35  |  Z36                                  |")
    print("----------------------------------------------------------------------------------")
    print("|{0:^7.3f}|{1:^7.3f}|{2:^7.3f}|{3:^7.3f}|{4:^7.3f}|{4:^7.3f}                    |".format\
            (z[31],z[32],z[33],z[34],z[35],z[36]))
    print("----------------------------------------------------------------------------------")
