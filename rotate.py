import numpy as np
from scipy.optimize import least_squares


def align2Paraboloid(x, y, z, paraFit):
    """
    Applies the shifts and rotations that align 
    (x,y,z) with the best fit paraboloid.
    
    Parameters
    ----------
    x : array
        1-d array with the x coordinates.
    y : array
        1-d array with the y coordinates.
    z : array
        1-d array with the z coordinates.
    paraFit : array
        Shifts and rotations derived from `parabolas.paraboloidFit`.
    
    Returns
    -------
    tuple
        Shifted and rotated (x,y,z).

    Examples
    --------
    >>> from parabolas import paraboloidFit
    >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    >>> z = np.array([0.1, 0.5, 0.9, 0.1, -0.8, -1.0])
    >>> fitresult = paraboloidFit(x, y, z, [60, 0, 0, -50, 0, 0])
    >>> xr, yr, zr = align2Paraboloid(x, y, z, fitresult.x)
    """

    cor = np.hstack((-1*paraFit[1:4], paraFit[4:6], 0))
    xr, yr, zr = shiftRotateXYZ(x, y, z, cor)

    return xr, yr, zr


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
    
    x1 = [np.cos(theta), -np.sin(theta), 0.]
    x2 = [np.sin(theta), np.cos(theta), 0.]
    x3 = [0., 0., 1.]

    return np.array([x1, x2, x3])


def invRx(theta):
    """
    """

    x1 = [1., 0., 0.]
    x2 = [0., np.cos(theta), np.sin(theta)]
    x3 = [0., -np.sin(theta), np.cos(theta)]

    return np.array([x1, x2, x3])


def invRy(theta):

    x1 = [np.cos(theta), 0., -np.sin(theta)]
    x2 = [0., 1., 0.]
    x3 = [np.sin(theta), 0., np.cos(theta)]

    return np.array([x1, x2, x3])


def invRz(theta):

    x1 = [np.cos(theta), np.sin(theta), 0.]
    x2 = [-np.sin(theta), np.cos(theta), 0.]
    x3 = [0., 0., 1.]

    return np.array([x1, x2, x3])


def Prime(pry):
    return np.dot(Rz(pry[2]), np.dot(Ry(pry[1]), Rx(pry[0])))


def PrimeX(pry, v):
    return np.dot(np.array([1., 0., 0.]), np.dot(Prime(pry), v))


def PrimeY(pry, v):
    return np.dot(np.array([0., 1., 0.]), np.dot(Prime(pry), v))


def PrimeZ(pry, v):
    return np.dot(np.array([0., 0., 1.]), np.dot(Prime(pry), v))


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


def rotateXYZ(x, y, z, coeffs):
    """
    Rotates and shifts a vector with (x,y,z) coordinates.
    The first three coeffs represents shifts along (x,y,z).
    The last three coeffs represent angles of rotation around (x,y,z).
    """

    L = np.array([x.flatten() + coeffs[0], 
                  y.flatten() + coeffs[1], 
                  z.flatten() + coeffs[2]])
    
    xr, yr, zr = np.dot(invRx(coeffs[3]), 
                       np.dot(invRy(coeffs[4]), 
                              np.dot(invRz(coeffs[5]), L)))
    
    return xr, yr, zr


def shiftRotateXYZ(x, y, z, coeffs):
    """
    Shifts and rotates a vector with (x,y,z) coordinates.
    The first three coeffs represents shifts along (x,y,z).
    The last three coeffs represent angles of rotation around 
    (x,y,z) in radians.
    """

    L = np.ma.array([x.flatten(),
                     y.flatten(),
                     z.flatten()])

    xr, yr, zr = np.dot(Rz(coeffs[5]),
                        np.dot(Ry(coeffs[4]),
                               np.dot(Rx(coeffs[3]), L)))

    xr += coeffs[0]
    yr += coeffs[1]
    zr += coeffs[2]

    xr.shape = yr.shape = zr.shape = z.shape

    return xr, yr, zr


def fitXYZ(coeffs, x_obs, y_obs, z_obs, x_ref, y_ref, z_ref):
    """
    Cost function to determine the rotattion and shift that aligns
    observed points to a reference system.
    The first three coeffs represents shifts along (x,y,z).
    The last three coeffs represent angles of rotation around (x,y,z).
    """

    L = np.array([x_obs.flatten() + coeffs[0], 
                  y_obs.flatten() + coeffs[1], 
                  z_obs.flatten() + coeffs[2]])
    
    xr, yr, zr = np.dot(invRx(coeffs[3]), 
                       np.dot(invRy(coeffs[4]), 
                              np.dot(invRz(coeffs[5]), L)))

    xr = np.ma.masked_invalid(xr)
    yr = np.ma.masked_invalid(yr)
    zr = np.ma.masked_invalid(zr)
    
    return np.hstack(((xr - x_ref).compressed(), (yr - y_ref).compressed(), (zr - z_ref).compressed()))


def alignXYZ(obs, ref, guess=[0.,0.,0.,0.,0.,0.], bounds=None):
    """
    Finds the shift and rotation needed to align two sets of points and applies them.
    """

    x_obs, y_obs, z_obs = obs
    x_ref, y_ref, z_ref = ref

    loss = "soft_l1"
    f_scale = 1.0

    align_coeffs = least_squares(fitXYZ,
                                 guess,
                                 args=(x_obs.flatten(), y_obs.flatten(), z_obs.flatten(), 
                                       x_ref.flatten(), y_ref.flatten(), z_ref.flatten()),
                                 max_nfev=1000000,
                                 loss=loss,
                                 f_scale=f_scale,
                                 ftol=1e-15,
                                 xtol=1e-15)

    print("Shift vector: {}".format(align_coeffs.x[:3]))
    print("Rotation angles: {}".format(align_coeffs.x[3:]))

    x_rot, y_rot, z_rot = rotateXYZ(x_obs, y_obs, z_obs, align_coeffs.x)

    return (x_rot, y_rot, z_rot, align_coeffs.x)
