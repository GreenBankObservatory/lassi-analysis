import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import opticspy
from test_surface2 import *
from test_surface import spherical_surf

def getRampSurfaceX():
    return np.array([[x for x in range(200)] for y in range(200)])

def getRampSurfaceY():
    return np.array([[y for x in range(200)] for y in range(200)])

def getFlatSurface(amp):
    return np.array([[amp for x in range(200)] for y in range(200)])

def getHalfCircle():

    # Create a 1/2 sphere
    l1 = 200
    r = 1.
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phis = np.linspace(0, pi/2.0, l1)
    thetas = np.linspace(0, 2.0*pi, l1)
    phi, theta = np.meshgrid(phis, thetas)
    print "phi.shape: ", phi.shape
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)

    # zero out parts outside range
    # not necessary because x and y never go out of range
    #for i in range(len(z)):
    #    for j in range(len(z)):
    #        if x[i][j]**2+y[i][j]**2>r:
    #            z[i][j]=0

    #import ipdb; ipdb.set_trace()

    #Set colours and render
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    # print "x.shape: ", x.shape
    ax = Axes3D(fig)

    ax.plot_surface(
        #x, y, z,  rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)     
        x, y, z) #  rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)     

    plt.show()

    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    # print "x.shape: ", x.shape
    ax = Axes3D(fig)

    ax.scatter(
        #x, y, z,  rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)     
        x, y, z) 
    return z

def main():

    # create different test surfaces:

    # Luckily, this produces all zero's for coefficients!
    # Z = getFlatSurface(0)

    # TBF: This may reveal a bug?
    # The barchart shows Z1 (piston) as huge (== our amplitude!), while everything else is negligable.
    # but the later printouts show that Z1 is zero?  And give ssmall coeff's for Z4
    # and Z11.
    #Z = getFlatSurface(1000)

    # This produces a hemisphere that gives us sensible zernikes.  Ex:
    # Z4 =  -0.229 Z20 Defocus
    # Z11 =  -0.042 Z40 Primary Spherical
    # Note that bar chart gives a Z1 close to the overall radius ...
    Z = spherical_surf(200)

    # sticking in an offset produces very interesting results!
    # Note that since our offset is symetrical, many of the Z's
    # are also symetrical!
    # Z2 =  -0.191 Z11 x Tilt
    # Z3 =  -0.191 Z11 y Tilt
    # Z4 =  -0.212 Z20 Defocus
    # Z5 =  0.024 Z22 Primary Astigmatism at 45
    # Z7 =  0.011 Z31 Primary y Coma
    # Z8 =  0.011 Z31 Primary x Coma
    # Z9 =  0.018 Z33 y Trefoil
    # Z10 =  -0.018 Z33 x Trefoil
    # Z11 =  0.02 Z40 Primary Spherical    
    #Z = spherical_surf(200, offset=0.25)


    # Look how these coefficient's change with the radius (previously one)
    # Z4 =  -0.114 Z20 Defocus
    # Z11 =  0.09 Z40 Primary Spherical
    # Z = spherical_surf(200, radius=0.5)

    # changing the x y scale without changing the radius amounts to 
    # a very small relative hemisphere that doesn't seem to get fit 
    # very well by these:
    # Z4 =  -0.011 Z20 Defocus
    # Z11 =  0.014 Z40 Primary Spherical
    #Z = spherical_surf(200, scale=10.)

    # Making sure we scale the radius by the same factor gives us the 
    # same coefficients as the default case, but scaled by 10!
    # Z4 =  -2.287 Z20 Defocus
    # Z11 =  -0.423 Z40 Primary Spherical
    # Z = spherical_surf(200, scale=10., radius=10.)

    # This ramp goes in just the x direction, so the domination of Z2
    # makes sense.  Can we just ignore the higher coefficients?
    # Z2 =  49.132 Z11 x Tilt
    # Z4 =  -0.211 Z20 Defocus
    # Z8 =  -0.172 Z31 Primary x Coma
    # Z10 =  -0.08 Z33 x Trefoil
    # Z11 =  -0.272 Z40 Primary Spherical
    #Z = getRampSurfaceX()

    # going in the Y direction gives a similar result
    # Z3 =  49.132 Z11 y Tilt
    # Z4 =  -0.211 Z20 Defocus
    # Z7 =  -0.172 Z31 Primary y Coma
    # Z9 =  0.08 Z33 y Trefoil
    #Z = getRampSurfaceY()

    # TBF: need to understand this surface better
    #Z = testsurface2()

    # our half circle data may not work with zernike fitting, even 
    # though the surface plot looks identical to spherical_surface, because
    # the spacing of the data is different - it's x, y is not regular, since
    # it was derived from regular theta and phi
    #Z = getHalfCircle()

    # Find the Zernike Coefficients:
    #fitlist,C1 = opticspy.zernike.fitting(Z,12,remain2D=1,remain3D=1,barchart=1,interferogram=1)
    fitlist,C1 = opticspy.zernike.fitting(Z,12,remain2D=1,barchart=1)
    print "fitlist: ", fitlist
    C1.listcoefficient()
    C1.zernikemap()


    print "NOTE: this is using the polynomial ordering found at"
    print "https://www.telescope-optics.net/images/zernike_noll.PNG"
    print "The ordering used by Mathematica is:"
    print "https://en.wikipedia.org/wiki/Zernike_polynomials#/media/File:Zernike_polynomials2.png"


if __name__ == "__main__":
    main()
