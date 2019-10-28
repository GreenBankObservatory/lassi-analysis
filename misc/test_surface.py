import numpy as __np__
from numpy import sqrt as __sqrt__
from numpy import cos as __cos__
from numpy import sin as __sin__
import matplotlib.pyplot as __plt__
from matplotlib import cm as __cm__
from matplotlib.ticker import LinearLocator as __LinearLocator__
from matplotlib.ticker import FormatStrFormatter as __FormatStrFormatter__
from mpl_toolkits.mplot3d import axes3d, Axes3D
#generate test surface figure
def spherical_surf(l1, radius=None, offset=None, scale=None):
    #R = 1.02
    if radius is None:
        R = 1.00
    else:
        R = radius
    if offset is None:
        offset = 0.

    if scale is None:
        scale = 1.0

    print("Radius: ", R)
    print("Offset: ", offset)
    print("Scale: ", scale)

    l1 = l1  #surface matrix length
    # theta = __np__.linspace(0, 2*__np__.pi, l1)
    # rho = __np__.linspace(0, 1, l1)
    # #rho = __np__.linspace(0, __np__.pi, l1)
    # [u,r] = __np__.meshgrid(theta,rho)
    # X = r*__cos__(u)
    # Y = r*__sin__(u)
    # Z = __sqrt__(R**2-r**2)-__sqrt__(R**2-1)
    # v_1 = max(abs(Z.max()),abs(Z.min()))

    # noise = (__np__.random.rand(len(Z),len(Z))*2-1)*0.05*v_1
    # #Z = Z+noise
    # fig = __plt__.figure(figsize=(12, 8), dpi=80)
    # #ax = fig.gca(projection='3d')
    # ax = Axes3D(fig)
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=__cm__.RdYlGn,\
				# 				linewidth=0, antialiased=False, alpha = 0.6)
    # v = max(abs(Z.max()),abs(Z.min()))
    # ax.set_zlim(-1, 2)
    # ax.zaxis.set_major_locator(__LinearLocator__(10))
    # ax.zaxis.set_major_formatter(__FormatStrFormatter__('%.02f'))
    # cset = ax.contourf(X, Y, Z, zdir='z', offset=-1, cmap=__cm__.RdYlGn)
    # fig.colorbar(surf, shrink=1, aspect=30)
    # __plt__.title('Test Surface: Spherical surface with some noise',fontsize=16)
    # __plt__.show()

    # now we really generate the test data
    #Generate test surface matrix from a detector
    x = __np__.linspace(-1*scale, 1*scale, l1)
    y = __np__.linspace(-1*scale, 1*scale, l1)
    #x = __np__.linspace(0, 1*scale, l1)
    #y = __np__.linspace(0, 1*scale, l1)
    [X,Y] = __np__.meshgrid(x,y)

    # Add noise?
    #Z = __sqrt__(R**2-(X**2+Y**2))-__sqrt__(R**2-1)+noise

    # what is the purpose of the added sqrt term?
    #Z = __sqrt__(R**2-(X**2+Y**2))-__sqrt__(R**2-1)

    # This will produce sqrt(neg. numbers) == nan for x and y's outside
    # the radius
    Z = __sqrt__(R**2-((X+offset)**2+(Y+offset)**2))
    #import ipdb; ipdb.set_trace()

    # print "Any Nan's in data before fixing?", __np__.isnan(Z).any()

    # convert those nan's outside the radius to zeros
    for i in range(len(Z)):
        for j in range(len(Z)):
            if (x[i]+offset)**2+(y[j]+offset)**2 > (R**2):
                Z[i][j]=0

    # print "Any Nan's in data after fixing?", __np__.isnan(Z).any()

    fig = __plt__.figure(figsize=(12, 8), dpi=80)
    #ax = fig.gca(projection='3d')
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='c', alpha=0.6)
    __plt__.show()


    return Z
