import numpy as np

def Rx(theta):

    x1 = [1., 0., 0.]
    x2 = [0., np.cos(theta), -np.sin(theta)]
    x3 = [0., np.sin(theta), np.cos(theta)]

    return np.array([x1, x2, x3])

def Ry(theta):

    x1 = [np.cos(theta), 0., np.sin(theta)]
    x2 = [0., 1., 0.]
    x3 = [-np.sin(theta), 0., np.cos(theta)]

    return np.array([x1, x2, x3])

def Rz(theta):
    # we won't be doing any rotating around z
    return np.array([[1., 0., 0.],[0., 1., 0.], [0., 0., 1.]])

def Prime(pry):
    #return Rx(pry[0])*Ry(pry[1])*Rz(pry[2])
    #return np.dot(Rx(pry[0]), np.dot(Ry(pry[1]), Rz(pry[2])))
    return np.dot(Rz(pry[2]), np.dot(Ry(pry[1]), Rx(pry[0])))

def PrimeX(pry, v):
    #return np.array([1., 0., 0.]) * Prime(pry) * v
    return np.dot(np.array([1., 0., 0.]), np.dot(Prime(pry), v))

def PrimeY(pry, v):
    #return np.array([0., 1., 0.]) * Prime(pry) * v
    return np.dot(np.array([0., 1., 0.]), np.dot(Prime(pry), v))

def PrimeZ(pry, v):
    cnst = np.array([0., 0., 1.])
    #print "cnst: ", cnst, cnst.shape
    ppry = Prime(pry)
    #print "pry", pry, pry.shape
    #print "Prime(pry)", ppry, ppry.shape
    #print "v:", v, v.shape
    #return np.array([0., 0., 1.]) * Prime(pry) * v
    return np.dot(cnst, np.dot(ppry, v))

def rotateXY(x, y, z, xRads, yRads):

    # assume x y z all have same shape
    orgShape = z.shape

    L = np.array([x.flatten(), y.flatten(), z.flatten()])
    # only rotate in x, y
    pry = (xRads, yRads, 0.)
    xr = PrimeX(pry, L)
    yr = PrimeY(pry, L)
    zr = PrimeZ(pry, L)

    xr.shape = yr.shape = zr.shape = orgShape

    return xr, yr, zr
