from copy import copy

import numpy as np

from scipy.sparse import diags
from scipy.optimize import least_squares

from utils.utils import midPoint, getRollingStat, radialMask

nMax = 120
zernikeNorm = { 1:  1.,
                2:  4.,
                3:  4.,
                4:  6.,
                5:  3.,
                6:  6.,
                7:  8.,
                8:  8.,
                9:  8.,
               10:  8.,
               11:  10.,
               12:  10.,
               13:  5.,
               14:  10.,
               15:  10.,
               16:  12.,
               17:  12.,
               18:  12.,
               19:  12.,
               20:  12.,
               21:  12.,
               22:  14.,
               23:  14.,
               24:  14.,
               25:  7.,
               26:  14.,
               27:  14.,
               28:  14.,
               29:  16.,
               30:  16.,
               31:  16.,
               32:  16.,
               33:  16.,
               34:  16.,
               35:  16.,
               36:  16.,
               37:  18.,
               38:  18.,
               39:  18.,
               40:  18.,
               41:  9.,
               42:  18.,
               43:  18.,
               44:  18.,
               45:  18.,
               46:  20.,
               47:  20.,
               48:  20.,
               49:  20.,
               50:  20.,
               51:  20.,
               52:  20.,
               53:  20.,
               54:  20.,
               55:  20.,
               56:  22.,
               57:  22.,
               58:  22.,
               59:  22.,
               60:  22.,
               61:  11.,
               62:  22.,
               63:  22.,
               64:  22.,
               65:  22.,
               66:  22.,
               67:  24.,
               68:  24.,
               69:  24.,
               70:  24.,
               71:  24.,
               72:  24.,
               73:  24.,
               74:  24.,
               75:  24.,
               76:  24.,
               77:  24.,
               78:  24.,
               79:  26.,
               80:  26.,
               81:  26.,
               82:  26.,
               83:  26.,
               84:  26.,
               85:  13.,
               86:  26.,
               87:  26.,
               88:  26.,
               89:  26.,
               90:  26.,
               91:  26.,
               92:  28.,
               93:  28.,
               94:  28.,
               95:  28.,
               96:  28.,
               97:  28.,
               98:  28.,
               99:  28.,
               100: 28.,
               101: 28.,
               102: 28.,
               103: 28.,
               104: 28.,
               105: 28.,
               106: 30.,
               107: 30.,
               108: 30.,
               109: 30.,
               110: 30.,
               111: 30.,
               112: 30.,
               113: 15.,
               114: 30.,
               115: 30.,
               116: 30.,
               117: 30.,
               118: 30.,
               119: 30.,
               120: 30.,
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
    return (1. - (6.*np.power(rho, 2.)) + (6.*np.power(rho, 4.)))

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

def z56(rho, theta):
    """
    """
    return _single_term_cos(10., rho, theta)

def z57(rho, theta):
    """
    """
    return _multi_term_cos(-9., 10., 8., rho, theta)

def z58(rho, theta):
    """
    """
    return (((28.*np.power(rho, 6.)) - (72.*np.power(rho, 8.)) + (45.*np.power(rho, 10.))) * np.cos(6.*theta))

def z59(rho, theta):
    """
    """
    return (((-35.*np.power(rho, 4.)) + (168.*np.power(rho, 6.)) - (252.*np.power(rho, 8.)) + (120.*np.power(rho, 10.))) * np.cos(4.*theta))

def z60(rho, theta):
    """
    """
    return (((15.*np.power(rho, 2.)) - (140.*np.power(rho, 4.)) + (420.*np.power(rho, 6.)) - (504.*np.power(rho, 8.)) + (210.*np.power(rho, 10.))) * np.cos(2.*theta))

def z61(rho, theta):
    """
    """
    return ((-1. + (30.*np.power(rho, 2.)) - (210.*np.power(rho, 4.)) + (560.*np.power(rho, 6.)) - (630.*np.power(rho, 8.)) + (252.*np.power(rho, 10.))))

def z62(rho, theta):
    """
    """
    return (((15.*np.power(rho, 2.)) - (140.*np.power(rho, 4.)) + (420.*np.power(rho, 6.)) - (504.*np.power(rho, 8.)) + (210.*np.power(rho, 10.))) * np.sin(2.*theta))

def z63(rho, theta):
    """
    """
    return (((-35.*np.power(rho, 4.)) + (168.*np.power(rho, 6.)) - (252.*np.power(rho, 8.)) + (120.*np.power(rho, 10.))) * np.sin(4.*theta))

def z64(rho, theta):
    """
    """
    return (((28.*np.power(rho, 6.)) - (72.*np.power(rho, 8.)) + (45.*np.power(rho, 10.))) * np.sin(6.*theta))

def z65(rho, theta):
    """
    """
    return _multi_term_sin(-9., 10., 8., rho, theta)

def z66(rho, theta):
    """
    """
    return _single_term_sin(10., rho, theta)

def z67(rho, theta):
    """
    """
    return _single_term_cos(11., rho, theta)

def z68(rho, theta):
    """
    """
    return _multi_term_cos(-10., 11., 9., rho, theta)

def z69(rho, theta):
    """
    """
    return (((36.*np.power(rho, 7.)) - (90.*np.power(rho, 9.)) + (55.*np.power(rho, 11.))) * np.cos(7.*theta))

def z70(rho, theta):
    """
    """
    return (((-56.*np.power(rho, 5.)) + (252.*np.power(rho, 7.)) - (360.*np.power(rho, 9.)) + (165.*np.power(rho, 11.))) * np.cos(5.*theta))

def z71(rho, theta):
    """
    """
    return (((35.*np.power(rho, 3.)) - (280.*np.power(rho, 5.)) + (756.*np.power(rho, 7.)) - (840.*np.power(rho, 9.)) + (330.*np.power(rho, 11.))) * np.cos(3.*theta))

def z72(rho, theta):
    """
    """
    return (((-6.*np.power(rho, 1.)) + (105.*np.power(rho, 3.)) - (560.*np.power(rho, 5.)) + (1260.*np.power(rho, 7.)) - (1260.*np.power(rho, 9.)) + (462.*np.power(rho, 11.))) * np.cos(1.*theta))

def z73(rho, theta):
    """
    """
    return (((-6.*np.power(rho, 1.)) + (105.*np.power(rho, 3.)) - (560.*np.power(rho, 5.)) + (1260.*np.power(rho, 7.)) - (1260.*np.power(rho, 9.)) + (462.*np.power(rho, 11.))) * np.sin(1.*theta))

def z74(rho, theta):
    """
    """
    return (((35.*np.power(rho, 3.)) - (280.*np.power(rho, 5.)) + (756.*np.power(rho, 7.)) - (840.*np.power(rho, 9.)) + (330.*np.power(rho, 11.))) * np.sin(3.*theta))

def z75(rho, theta):
    """
    """
    return (((-56.*np.power(rho, 5.)) + (252.*np.power(rho, 7.)) - (360.*np.power(rho, 9.)) + (165.*np.power(rho, 11.))) * np.sin(5.*theta))

def z76(rho, theta):
    """
    """
    return (((36.*np.power(rho, 7.)) - (90.*np.power(rho, 9.)) + (55.*np.power(rho, 11.))) * np.sin(7.*theta))

def z77(rho, theta):
    """
    """
    return _multi_term_sin(-10., 11., 9., rho, theta)

def z78(rho, theta):
    """
    """
    return _single_term_sin(11., rho, theta)

def z79(rho, theta):
    """
    """
    return _single_term_cos(12., rho, theta)

def z80(rho, theta):
    """
    """
    return _multi_term_cos(-11., 12., 10., rho, theta)

def z81(rho, theta):
    """
    """
    return (((45.*np.power(rho, 8.)) - (110.*np.power(rho, 10.)) + (66.*np.power(rho, 12.))) * np.cos(8.*theta))

def z82(rho, theta):
    """
    """
    return (((-84.*np.power(rho, 6.)) + (360.*np.power(rho, 8.)) - (495.*np.power(rho, 10.)) + (220.*np.power(rho, 12.))) * np.cos(6.*theta))

def z83(rho, theta):
    """
    """
    return (((70.*np.power(rho, 4.)) - (504.*np.power(rho, 6.)) + (1260.*np.power(rho, 8.)) - (1320.*np.power(rho, 10.)) + (495.*np.power(rho, 12.))) * np.cos(4.*theta))

def z84(rho, theta):
    """
    """
    return (((-21.*np.power(rho, 2.)) + (280.*np.power(rho, 4.)) - (1260.*np.power(rho, 6.)) + (2520.*np.power(rho, 8.)) - (2310.*np.power(rho, 10.)) + (792.*np.power(rho, 12.))) * np.cos(2.*theta))

def z85(rho, theta):
    """
    """
    return ((1. - (42.*np.power(rho, 2.)) + (420.*np.power(rho, 4.)) - (1680.*np.power(rho, 6.)) + (3150.*np.power(rho, 8.)) - (2772.*np.power(rho, 10.)) + (924.*np.power(rho, 12.))))

def z86(rho, theta):
    """
    """
    return (((-21.*np.power(rho, 2.)) + (280.*np.power(rho, 4.)) - (1260.*np.power(rho, 6.)) + (2520.*np.power(rho, 8.)) - (2310.*np.power(rho, 10.)) + (792.*np.power(rho, 12.))) * np.sin(2.*theta))

def z87(rho, theta):
    """
    Z_{12}^{-4}
    """
    return (((70.*np.power(rho, 4.)) - (504.*np.power(rho, 6.)) + (1260.*np.power(rho, 8.)) - (1320.*np.power(rho, 10.)) + (495.*np.power(rho, 12.))) * np.sin(4.*theta))

def z88(rho, theta):
    """
    Z_{12}^{-6}
    """
    return (((-84.*np.power(rho, 6.)) + (360.*np.power(rho, 8.)) - (495.*np.power(rho, 10.)) + (220.*np.power(rho, 12.))) * np.sin(6.*theta))

def z89(rho, theta):
    """
    Z_{12}^{-8}
    """
    return (((45.*np.power(rho, 8.)) - (110.*np.power(rho, 10.)) + (66.*np.power(rho, 12.))) * np.sin(8.*theta))

def z90(rho, theta):
    """
    Z_{12}^{-10}
    """
    return _multi_term_sin(-11., 12., 10., rho, theta)

def z91(rho, theta):
    """
    Z_{12}^{-12}
    """
    return _single_term_sin(12., rho, theta)

def z92(rho, theta):
    """
    Z_{13}^{13}
    """
    return _single_term_cos(13., rho, theta)

def z93(rho, theta):
    """
    Z_{13}^{11}
    """
    return _multi_term_cos(-12., 13., 11., rho, theta)

def z94(rho, theta):
    """
    Z_{13}^{9}
    """
    return (((55.*np.power(rho, 9.)) - (132.*np.power(rho, 11.)) + (78.*np.power(rho, 13.))) * np.cos(9.*theta))

def z95(rho, theta):
    """
    Z_{13}^{7}
    """
    return (((-120.*np.power(rho, 7.)) + (495.*np.power(rho, 9.)) - (660.*np.power(rho, 11.)) + (286.*np.power(rho, 13.))) * np.cos(7.*theta))

def z96(rho, theta):
    """
    Z_{13}^{5}
    """
    return (((126.*np.power(rho, 5.)) - (840.*np.power(rho, 7.)) + (1980.*np.power(rho, 9.)) - (1980.*np.power(rho, 11.)) + (715.*np.power(rho, 13.))) * np.cos(5.*theta))

def z97(rho, theta):
    """
    Z_{13}^{3}
    """
    return (((-56.*np.power(rho, 3.)) + (630.*np.power(rho, 5.)) - (2520.*np.power(rho, 7.)) + (4620.*np.power(rho, 9.)) - (3960.*np.power(rho, 11.)) + (1287.*np.power(rho, 13.))) * np.cos(3.*theta))

def z98(rho, theta):
    """
    Z_{13}^{1}
    """
    return (((7.*np.power(rho, 1.)) - (168.*np.power(rho, 3.)) + (1260.*np.power(rho, 5.)) - (4200.*np.power(rho, 7.)) + (6930.*np.power(rho, 9.)) - (5544.*np.power(rho, 11.)) + (1716.*np.power(rho, 13.))) * np.cos(1.*theta))

def z99(rho, theta):
    """
    Z_{13}^{-1}
    """
    return (((7.*np.power(rho, 1.)) - (168.*np.power(rho, 3.)) + (1260.*np.power(rho, 5.)) - (4200.*np.power(rho, 7.)) + (6930.*np.power(rho, 9.)) - (5544.*np.power(rho, 11.)) + (1716.*np.power(rho, 13.))) * np.sin(1.*theta))

def z100(rho, theta):
    """
    Z_{13}^{-3}
    """
    return (((-56.*np.power(rho, 3.)) + (630.*np.power(rho, 5.)) - (2520.*np.power(rho, 7.)) + (4620.*np.power(rho, 9.)) - (3960.*np.power(rho, 11.)) + (1287.*np.power(rho, 13.))) * np.sin(3.*theta))

def z101(rho, theta):
    """
    Z_{13}^{-5}
    """
    return (((126.*np.power(rho, 5.)) - (840.*np.power(rho, 7.)) + (1980.*np.power(rho, 9.)) - (1980.*np.power(rho, 11.)) + (715.*np.power(rho, 13.))) * np.sin(5.*theta))

def z102(rho, theta):
    """
    Z_{13}^{-7}
    """
    return (((-120.*np.power(rho, 7.)) + (495.*np.power(rho, 9.)) - (660.*np.power(rho, 11.)) + (286.*np.power(rho, 13.))) * np.sin(7.*theta))

def z103(rho, theta):
    """
    Z_{13}^{-9}
    """
    return (((55.*np.power(rho, 9.)) - (132.*np.power(rho, 11.)) + (78.*np.power(rho, 13.))) * np.sin(9.*theta))

def z104(rho, theta):
    """
    Z_{13}^{-11}
    """
    return _multi_term_sin(-12., 13., 11., rho, theta)

def z105(rho, theta):
    """
    Z_{13}^{-13}
    """
    return _single_term_sin(13., rho, theta)

def z106(rho, theta):
    """
    Z_{14}^{14}
    """
    return _single_term_cos(14., rho, theta)

def z107(rho, theta):
    """
    Z_{14}^{12}
    """
    return _multi_term_cos(-13., 14., 12., rho, theta)

def z108(rho, theta):
    """
    Z_{14}^{10}
    """
    return (((66.*np.power(rho, 10.)) - (156.*np.power(rho, 12.)) + (91.*np.power(rho, 14.))) * np.cos(10.*theta))

def z109(rho, theta):
    """
    Z_{14}^{8}
    """
    return (((-165.*np.power(rho, 8.)) + (660.*np.power(rho, 10.)) - (858.*np.power(rho, 12.)) + (364.*np.power(rho, 14.))) * np.cos(8.*theta))

def z110(rho, theta):
    """
    Z_{14}^{6}
    """
    return (((210.*np.power(rho, 6.)) - (1320.*np.power(rho, 8.)) + (2970.*np.power(rho, 10.)) - (2860.*np.power(rho, 12.)) + (1001.*np.power(rho, 14.))) * np.cos(6.*theta))

def z111(rho, theta):
    """
    Z_{14}^{4}
    """
    return (((-126.*np.power(rho, 4.)) + (1260.*np.power(rho, 6.)) - (4620.*np.power(rho, 8.)) + (7920.*np.power(rho, 10.)) - (6435.*np.power(rho, 12.)) + (2002.*np.power(rho, 14.))) * np.cos(4.*theta))

def z112(rho, theta):
    """
    Z_{14}^{2}
    """
    return (((28.*np.power(rho, 2.)) - (504.*np.power(rho, 4.)) + (3150.*np.power(rho, 6.)) - (9240.*np.power(rho, 8.)) + (13860.*np.power(rho, 10.)) - (10296.*np.power(rho, 12.)) + (3003.*np.power(rho, 14.))) * np.cos(2.*theta))

def z113(rho, theta):
    """
    Z_{14}^{0}
    """
    return ((-1 + (56.*np.power(rho, 2.)) - (756.*np.power(rho, 4.)) + (4200.*np.power(rho, 6.)) - (11550.*np.power(rho, 8.)) + (16632.*np.power(rho, 10.)) - (12012.*np.power(rho, 12.)) + (3432.*np.power(rho, 14.))))

def z114(rho, theta):
    """
    Z_{14}^{-2}
    """
    return (((28.*np.power(rho, 2.)) - (504.*np.power(rho, 4.)) + (3150.*np.power(rho, 6.)) - (9240.*np.power(rho, 8.)) + (13860.*np.power(rho, 10.)) - (10296.*np.power(rho, 12.)) + (3003.*np.power(rho, 14.))) * np.sin(2.*theta))

def z115(rho, theta):
    """
    Z_{14}^{-4}
    """
    return (((-126.*np.power(rho, 4.)) + (1260.*np.power(rho, 6.)) - (4620.*np.power(rho, 8.)) + (7920.*np.power(rho, 10.)) - (6435.*np.power(rho, 12.)) + (2002.*np.power(rho, 14.))) * np.sin(4.*theta))

def z116(rho, theta):
    """
    Z_{14}^{-6}
    """
    return (((210.*np.power(rho, 6.)) - (1320.*np.power(rho, 8.)) + (2970.*np.power(rho, 10.)) - (2860.*np.power(rho, 12.)) + (1001.*np.power(rho, 14.))) * np.sin(6.*theta))

def z117(rho, theta):
    """
    Z_{14}^{-8}
    """
    return (((-165.*np.power(rho, 8.)) + (660.*np.power(rho, 10.)) - (858.*np.power(rho, 12.)) + (364.*np.power(rho, 14.))) * np.sin(8.*theta))

def z118(rho, theta):
    """
    Z_{14}^{-10}
    """
    return (((66.*np.power(rho, 10.)) - (156.*np.power(rho, 12.)) + (91.*np.power(rho, 14.))) * np.sin(10.*theta))

def z119(rho, theta):
    """
    Z_{14}^{-12}
    """
    return _multi_term_sin(-13., 14., 12., rho, theta)

def z120(rho, theta):
    """
    Z_{14}^{-14}
    """
    return _single_term_sin(14., rho, theta)

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
            z56,
            z57,
            z58,
            z59,
            z60,
            z61,
            z62,
            z63,
            z64,
            z65,
            z66,
            z67,
            z68,
            z69,
            z70,
            z71,
            z72,
            z73,
            z74,
            z75,
            z76,
            z77,
            z78,
            z79,
            z80,
            z81,
            z82,
            z83,
            z84,
            z85,
            z86,
            z87,
            z88,
            z89,
            z90,
            z91,
            z92,
            z93,
            z94,
            z95,
            z96,
            z97,
            z98,
            z99,
            z100,
            z101,
            z102,
            z103,
            z104,
            z105,
            z106,
            z107,
            z108,
            z109,
            z110,
            z111,
            z112,
            z113,
            z114,
            z115,
            z116,
            z117,
            z118,
            z119,
            z120,
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


def getZernikeCoeffs(surface, order, printReport=False, norm='sqrt', radius=1):
    """
    Determines the coefficients of Zernike polynomials that best describe a surface.

    Parameters
    ----------
    surface : 2-D array
        The surface where Zernike coefficients will be determined.
    order : int
        How many order of Zernike polynomials you want to incorporate. Less than 120.
    printReport : bool, optional
        Print the determined Zernike coefficients?
    Returns
    -------
    array
        First `order` coefficients for the Zernike polynomials that describe a deformed surface.
    """

    if order > nMax:
        raise ValueError('order must be less than {}.'.format(nMax+1))

    coeffs = np.zeros(order+1, dtype=np.float)

    # Make the support to evaluate the Zernike polynomials.
    ny = surface.shape[0]
    nx = surface.shape[1]
    x = np.linspace(-1., 1., nx)
    y = np.linspace(-1., 1., ny)
    [xx,yy] = np.meshgrid(x, y)
    r = np.sqrt(xx**2. + yy**2.)
    u = np.arctan2(yy, xx)

    # Loop over Zernike polynomials and use their orthogonality to determine their coefficients.
    # The active surface starts counting from Z1.
    for i in range(1,order+1):
        func = zernikes[i]
        zf = func(r, u)
        mask = (r > radius)
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
        coeffs[i] = a

    # Print a table with the coefficients.
    if printReport:
        zernikePrint(coeffs)

    return coeffs


def getZernikeCoeffsOLS(x, y, z, nZern, xOffset=0, yOffset=0, xMax=None, yMax=None, weights=None):
    """
    Determines the coefficients of the Zernike polynomials in z using weighted least squares.
    """
    
    if xMax is None:
        xMax = np.nanmax(x)
    if yMax is None:
        yMax = np.nanmax(y)
    if weights is None:
        w = np.ma.ones(x.shape, dtype=np.int)
    else:
        w = weights
    
    # Remove NaN values.
    z = np.ma.masked_where(x.mask | y.mask | z.mask | w.mask, z)
    x = np.ma.masked_where(x.mask | y.mask | z.mask | w.mask, x)
    y = np.ma.masked_where(x.mask | y.mask | z.mask | w.mask, y)
    w = np.ma.masked_where(x.mask | y.mask | z.mask | w.mask, w)
    x = x.compressed()
    y = y.compressed()
    w = w.compressed()
    z = z.compressed()
    
    # Transform the coordinates to the unit circle.
    xcn = (x - xOffset)/xMax
    ycn = (y - yOffset)/yMax

    # We defined the Zernike polynomials in polar coordinates.
    rcn = np.sqrt(xcn**2. + ycn**2.)
    ucn = np.arctan2(ycn, xcn)
    
    # Build the matrix with the Zernike polynomials.
    zMat = np.zeros((np.prod(rcn.shape), nZern), dtype=np.float)
    for i in range(0,nZern):
        zMat[:,i] = zernikes[i+1](rcn.flatten(), ucn.flatten())

    zMat = np.matrix(zMat)
    sMat = np.matrix(z)
    wMat = diags(w)
    coefs = np.linalg.lstsq(wMat*zMat, wMat*sMat.T, rcond=None)
    
    return np.hstack((0, np.asarray(coefs[0])[:,0]))


def zernikeWLS(x, y, z, nZern, weights=None):
    """
    Use weighted least-squares to compute Zernike coefficients.
    """

    # Use WLS to determine the Zernike coefficients.
    if weights is None:
        weights = np.ones(z.shape)

    weights = np.ma.masked_invalid(weights)
    
    x_ = np.ma.masked_invalid(x)
    x_ -= midPoint(x)
    y_ = np.ma.masked_invalid(y)
    y_ -= midPoint(y)
    
    fl_wls = getZernikeCoeffsOLS(x_, y_, z, nZern, weights=weights)
    
    return fl_wls


def zernikeCost(coeffs, x, y, z, w):
    """
    Cost funtion for the least squares problem.
    """

    zpoly = zernikePoly(x, y, midPoint(x), midPoint(y), coeffs)

    return w*(z.flatten() - zpoly.flatten())


def zernikeJac(coeffs, x, y, z):
    """
    Jacobian of the cost function for the least squares problem.
    This does not work at the moment.
    """

    coeffs = np.ones_like(coeffs)
    coeffs[:2] = 0

    jac = np.zeros((len(z),len(coeffs)))

    for i in range(2,len(coeffs)):
        cis = np.zeros(36)
        cis[i] = 1.
        jac[i] = zernikePoly(x, y, midPoint(x), midPoint(y), cis)

    return jac


def zernikeFitLS(x, y, z, guess, weights=None, max_nfev=10000, loss="soft_l1", f_scale=1e-3, ftol=1e-10, xtol=1e-10):
    """
    Determine the coefficients of the Zernike polynomials that describe a deformed surface using least-squares.
    """

    if weights is None:
        weights = np.ones_like(z)

    x_scale = np.ones_like(guess)*1e-5

    args = (x, y, z, weights)
    r = least_squares(zernikeCost,
                      guess,
                      args=args,
                      max_nfev=10000,
                      loss=loss,
                      f_scale=f_scale,
                      ftol=ftol,
                      xtol=xtol,
                      x_scale=x_scale)

    return r


def zernikeResidualSurfaceError(x, y, z_coef, r=50, eps=230e-6, lmbd=2.6e-3, verbose=False):
    """
    Computes the effect of the residual Zernike on the surface error.


    """

    # Simulate thermal deformation.
    # The Zernike polynomials is at the center of the map.
    x_mid = midPoint(x)
    y_mid = midPoint(y)
    z_sim = zernikePoly(x, y, x_mid, y_mid, coefficients=z_coef)
    z_sim[~radialMask(x,y,r)] = np.nan
    eps_z = np.nanstd(z_sim)
    if verbose:
        print("Residual surface error: {} microns".format(eps_z*1e6))

    # Compute surface error.
    eps_tot = np.sqrt(eps_z**2. + eps**2.)
    if verbose:
        print("Total surface error: {} microns".format(eps_tot*1e6))

    # Compute aperture efficiency.
    eta = lambda lmbd, eps: 0.71*np.exp(-np.power(4.*np.pi*eps/lmbd, 2.))
    eta_tot = eta(lmbd, eps_tot)
    if verbose:
        print("Aperture efficiency: {}".format(eta_tot))
        print("Change in aperture efficiency: {} %".format((eta(lmbd, eps) - eta_tot)/eta(lmbd, eps)*100.))

    return eps_tot, eta_tot, eta


def zernikePoly(x, y, xOffset, yOffset, coefficients, xMax=-1e22, yMax=-1e22, verbose=False):
    """
    Evaluate a linear combination of Zernike polynomials with the given coefficients.
    The Zernike polynomails will be evaluated over the unit circle by normalizing the
    x and y coordinates.

    Parameters
    ----------
    x : array
        x coordinates where to evaluate the Zernike polynomial.
    y : array
        y coordinates where to evaluate the Zernike polynomial.
    xOffset : float
        Center of the Zernike polynomial in the x coordinate.
    yOffset : float
        Center of the Zernike polynomial in the y coordinate.
    coefficients : array
        Coefficients for the Zernike polynomials.
        The first element is the coefficient for the order 0 polynomial,
        the second element for the order 1 polynomial and so on.
    xMax : float, optional
        Value used to normalize the x coordinates.
        It defaults to the maximum valid value of `x`.
    yMax : float, optional
        Value used to normalize the y coordinates.
        It defaults to the maximum valid value of `y`.
    verbose : bool, optional
        Verbose output?

    Returns
    -------
    array
        Array of the same shape as x and y with the z values of the Zernike polynomial.
    """

    if len(coefficients) > nMax + 1:
        raise ValueError('coefficients must have less than {} items.'.format(zernikies.nMax+1))

    if xMax == -1e22:
        xMax = np.nanmax(x - xOffset)
    if yMax == -1e22:
        yMax = np.nanmax(y - yOffset)

    xcn = (x - xOffset)/xMax
    ycn = (y - yOffset)/yMax

    rcn = np.sqrt(xcn**2. + ycn**2.)
    ucn = np.arctan2(ycn, xcn)

    z = zernikePolar(coefficients, rcn, ucn)

    if verbose:
        print("Zernike polynomials with coefficients", coefficients)
        print("Their linear combination has mean: {0:.2e}, min: {1:.2e}, max: {2:.2e}".format(np.mean(z), np.nanmin(z), np.nanmax(z)))

    return z


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
