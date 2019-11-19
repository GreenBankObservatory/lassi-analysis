from copy import copy

import numpy as np
#import opticspy

from plotting import barChartPlot, zernikeResiduals2DPlot
from zernikeIndexing import noll2asAnsi, printZs

nMax = 55
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
               29: 16.,
               30: 16.,
               31: 16.,
               32: 16.,
               33: 16.,
               34: 16.,
               35: 16.,
               36: 16.,
               37: 18.,
               38: 18.,
               39: 18.,
               40: 18.,
               41: 9.,
               42: 18.,
               43: 18.,
               44: 18.,
               45: 18.,
               46: 20.,
               47: 20.,
               48: 20.,
               49: 20.,
               50: 20.,
               51: 20.,
               52: 20.,
               53: 20.,
               54: 20.,
               55: 20.,
                   }

def z0(rho, theta):
    """
    """
    return np.zeros_like(rho)

def z1(rho, theta):
    """
    Piston.
    """
    return np.ones_like(rho)

def z2(rho, theta):
    """
    
    """
    return _single_term_cos(1., rho, theta)

def z3(rho, theta):
    """

    """
    return _single_term_sin(1., rho, theta)

def z4(rho, theta):
    """
    """
    return _single_term_cos(2., rho, theta)

def z5(rho, theta):
    """
    Defocus.
    """
    return (2.*rho**2. - 1.)

def z6(rho, theta):
    """

    """
    return _single_term_sin(2., rho, theta)

def z7(rho, theta):
    """
    Oblique trefoil.
    """
    return _single_term_cos(3., rho, theta)

def z8(rho, theta):
    """

    """
    return _multi_term_cos(-2., 3., 1., rho, theta)

def z9(rho, theta):
    """

    """
    return _multi_term_sin(-2., 3., 1., rho, theta)

def z10(rho, theta):
    """

    """
    return _single_term_sin(3., rho, theta)

def z11(rho, theta):
    """

    """
    return _single_term_cos(4., rho, theta)

def z12(rho, theta):
    """

    """
    return _multi_term_cos(-3., 4., 2., rho, theta)

def z13(rho, theta):
    """
    Primary spherical.
    """
    return (1. - (6.*np.power(rho, 2.))  + (6.*np.power(rho, 4.)))

def z14(rho, theta):
    """

    """
    return _multi_term_sin(-3., 4., 2., rho, theta)

def z15(rho, theta):
    """
    Oblique quadrafoil.
    """
    return _single_term_sin(4., rho, theta)

def z16(rho, theta):
    """

    """
    return _single_term_cos(5., rho, theta)

def z17(rho, theta):
    """

    """
    return _multi_term_cos(-4., 5., 3., rho, theta)

def z18(rho, theta):
    """

    """
    return (3.*rho - 12.*np.power(rho, 3.) + 10.*np.power(rho, 5.)) * np.cos(theta)

def z19(rho, theta):
    """

    """
    return (3.*rho - 12.*np.power(rho, 3.) + 10.*np.power(rho, 5.)) * np.sin(theta)

def z20(rho, theta):
    """

    """
    return _multi_term_sin(-4., 5., 3., rho, theta)

def z21(rho, theta):
    """

    """
    return _single_term_sin(5., rho, theta)

def z22(rho, theta):
    """

    """
    return _single_term_cos(6., rho, theta)

def z23(rho, theta):
    """

    """
    return _multi_term_cos(-5., 6., 4., rho, theta)

def z24(rho, theta):
    """

    """
    return (((6.*np.power(rho, 2.)) - (20.*np.power(rho, 4.)) + (15.*np.power(rho, 6.))) * np.cos(2.*theta))

def z25(rho, theta):
    """

    """
    return (-1. + (12.*np.power(rho, 2.)) - (30.*np.power(rho, 4.)) + (20.*np.power(rho, 6.)))

def z26(rho, theta):
    """

    """
    return (((6.*np.power(rho, 2.)) - (20.*np.power(rho, 4.)) + (15.*np.power(rho, 6.))) * np.sin(2.*theta))

def z27(rho, theta):
    """

    """
    return _multi_term_sin(-5., 6., 4., rho, theta)

def z28(rho, theta):
    """

    """
    return _single_term_sin(6., rho, theta)

def z29(rho, theta):
    """

    """
    return _single_term_cos(7., rho, theta)

def z30(rho, theta):
    """

    """
    return _multi_term_cos(-6., 7., 5., rho, theta)

def z31(rho, theta):
    """

    """
    return (((10.*np.power(rho, 3.)) - (30.*np.power(rho, 5.)) + (21.*np.power(rho, 7.))) * np.cos(3.*theta))

def z32(rho, theta):
    """

    """
    return (((-4.*np.power(rho, 1.)) + (30.*np.power(rho, 3.)) - (60.*np.power(rho, 5.)) + (35.*np.power(rho, 7.)))*np.cos(theta))

def z33(rho, theta):
    """

    """
    return (((-4.*np.power(rho, 1.)) + (30.*np.power(rho, 3.)) - (60.*np.power(rho, 5.)) + (35.*np.power(rho, 7.)))*np.sin(theta))

def z34(rho, theta):
    """
    
    """
    return (((10.*np.power(rho, 3.)) - (30.*np.power(rho, 5.)) + (21.*np.power(rho, 7.))) * np.sin(3.*theta))

def z35(rho, theta):
    """

    """
    return _multi_term_sin(-6., 7., 5., rho, theta)

def z36(rho, theta):
    """

    """
    return _single_term_sin(7., rho, theta)

def z37(rho, theta):
    """
    """
    return _single_term_cos(8., rho, theta)

def z38(rho, theta):
    """
    """
    return _multi_term_cos(-7., 8., 6., rho, theta)

def z39(rho, theta):
    """
    """
    return (((15.*np.power(rho, 4.)) - (42.*np.power(rho, 6.)) + (28.*np.power(rho, 8.))) * np.cos(4.*theta))

def z40(rho, theta):
    """
    """
    return (((-10.*np.power(rho, 2.)) + (60.*np.power(rho, 4.)) - (105.*np.power(rho, 6.)) + (56.*np.power(rho, 8.))) * np.cos(2.*theta))

def z41(rho, theta):
    """
    """
    return (1. + (-20.*np.power(rho, 2.)) + (90.*np.power(rho, 4.)) - (140.*np.power(rho, 6.)) + (70.*np.power(rho, 8.)))

def z42(rho, theta):
    """
    """
    return (((-10.*np.power(rho, 2.)) + (60.*np.power(rho, 4.)) - (105.*np.power(rho, 6.)) + (56.*np.power(rho, 8.))) * np.sin(2.*theta))

def z43(rho, theta):
    """
    """
    return (((15.*np.power(rho, 4.)) - (42.*np.power(rho, 6.)) + (28.*np.power(rho, 8.))) * np.sin(4.*theta))

def z44(rho, theta):
    """
    """
    return _multi_term_sin(-7., 8., 6., rho, theta)

def z45(rho, theta):
    """
    """
    return _single_term_sin(8., rho, theta)

def z46(rho, theta):
    """
    """
    return _single_term_cos(9., rho, theta)

def z47(rho, theta):
    """
    """
    return _multi_term_cos(-8., 9., 7., rho, theta)

def z48(rho, theta):
    """
    """
    return (((21.*np.power(rho, 5.)) - (56.*np.power(rho, 7.)) + (36.*np.power(rho, 9.))) * np.cos(5.*theta))

def z49(rho, theta):
    """
    """
    return (((-20.*np.power(rho, 3.)) + (105.*np.power(rho, 5.)) - (168.*np.power(rho, 7.)) + (84.*np.power(rho, 9.))) * np.cos(3.*theta))

def z50(rho, theta):
    """
    """
    return (((5.*np.power(rho, 1.)) + (-60.*np.power(rho, 3.)) + (210.*np.power(rho, 5.)) - (280.*np.power(rho, 7.)) + (126.*np.power(rho, 9.))) * np.cos(1.*theta))

def z51(rho, theta):
    """
    """
    return (((5.*np.power(rho, 1.)) + (-60.*np.power(rho, 3.)) + (210.*np.power(rho, 5.)) - (280.*np.power(rho, 7.)) + (126.*np.power(rho, 9.))) * np.sin(1.*theta))

def z52(rho, theta):
    """
    """
    return (((-20.*np.power(rho, 3.)) + (105.*np.power(rho, 5.)) - (168.*np.power(rho, 7.)) + (84.*np.power(rho, 9.))) * np.sin(3.*theta))

def z53(rho, theta):
    """
    """
    return (((21.*np.power(rho, 5.)) - (56.*np.power(rho, 7.)) + (36.*np.power(rho, 9.))) * np.sin(5.*theta))

def z54(rho, theta):
    """
    """
    return _multi_term_sin(-8., 9., 7., rho, theta)

def z55(rho, theta):
    """
    """
    return _single_term_sin(9., rho, theta)

zernikes = [z0,
            z1,
            z2,
            z3,
            z4,
            z5,
            z6,
            z7,
            z8,
            z9,
            z10,
            z11,
            z12,
            z13,
            z14,
            z15,
            z16,
            z17,
            z18,
            z19,
            z20,
            z21,
            z22,
            z23,
            z24,
            z25,
            z26,
            z27,
            z28,
            z29,
            z30,
            z31,
            z32,
            z33,
            z34,
            z35,
            z36,
            z37,
            z38,
            z39,
            z40,
            z41,
            z42,
            z43,
            z44,
            z45,
            z46,
            z47,
            z48,
            z49,
            z50,
            z51,
            z52,
            z53,
            z54,
            z55,
]

def _single_term_cos(power, rho, theta):
    """
    Utility function to compute Zernike polynomials.
    This captures Zernike polynomials :math:`Z^{m}_{n}` with n-m=0.
    """

    return np.power(rho, power) * np.cos(power * theta)

def _single_term_sin(power, rho, theta):
    """
    Utility function to compute Zernike polynomials.
    This captures Zernike polynomials :math:`Z^{-m}_{n}` with n-m=0.
    """

    return np.power(rho, power) * np.sin(power * theta)

def _multi_term_cos(coef, power, multiplier, rho, theta):
    """
    Utility function to compute Zernike polynomials.
    This captures Zernike polynomials :math:`Z^{m}_{n}` with n-m=2.
    """
    
    power2 = (coef + 1.) * -1.
    return ((coef * np.power(rho, power2) + (power * np.power(rho, power))) * np.cos(multiplier * theta))

def _multi_term_sin(coef, power, multiplier, rho, theta):
    """
    Utility function to compute Zernike polynomials.
    This captures Zernike polynomials :math:`Z^{-m}_{n}` with n-m=2.
    """
    
    power2 = (coef + 1.) * -1.
    return ((coef * np.power(rho, power2) + (power * np.power(rho, power))) * np.sin(multiplier * theta))

def zernikePolar(coefficients, rho, theta):
    """
    """

    order = len(coefficients)
    zf = np.zeros_like(rho)

    for i in range(1,order):
        func = zernikes[i]
        zf += coefficients[i]*func(rho, theta)

    return zf

def getZernikeCoeffs(surface, order, plot2D=False, barChart=False, printReport=False, norm='sqrt'):
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
    ny = surface.shape[0]
    nx = surface.shape[1]
    x = np.linspace(-1., 1., nx)
    y = np.linspace(-1., 1., ny)
    [xx,yy] = np.meshgrid(x, y)
    r = np.sqrt(xx**2. + yy**2.)
    u = np.arctan2(yy, xx)

    # Loop over Zernike polynomials and use their orthogonality to determine their coefficients.
    for i in range(1,order+1):
        func = zernikes[i]
        zf = func(r, u)
        mask = (r > 1)
        zf[mask] = 0

        # Define the normalization factor.
        if norm == 'sqrt':
            zn = np.sqrt(zernikeNorm[i])
        elif norm == 'active-surface':
            zn = zernikeNorm[i]
        elif norm == 'one':
            zn = 1.

        # Get the coefficients like in a Fourier series.
        a = np.sum(surface*zf)*2.*2./nx/ny/np.pi*zn
        coeffs.append(a)

    # Plot bar chart of Zernike coefficients.
    if barChart:
        fitlist = coeffs[1:order+1]
        index = np.arange(1,order+1)
        barChartPlot(index, fitlist)

    # Plot the residuals.
    if plot2D:
        # Compute the residuals.
        z_new = surface - zernikePolar(coeffs, r, u)
        z_new[mask] = 0
        zernikeResiduals2DPlot(xx, yy, z_new)

    # Print a table with the coefficients.
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
