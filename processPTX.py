import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianRepresentation
from astropy.coordinates.matrix_utilities import rotation_matrix

def rotateXYaboutZ(xyz, rotDegrees):

    # define it as cartesian
    rep = CartesianRepresentation(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    # define our rotation about z
    # TBF: why must we negate this to match Mathematica results?
    rot = rotation_matrix(-rotDegrees * u.deg, axis='z')

    # apply the rotation
    new_rep = rep.transform(rot)

    new_xyz = new_rep.get_xyz().value

    # get it back into original form
    # TBF: must be a faster method!
    xyzNew = []
    x = new_xyz[0]
    y = new_xyz[1]
    z = new_xyz[2]
    for i in range(len(x)):
        xyzNew.append((x[i], y[i], z[i]))
        
    return np.array(xyzNew)    
        
    
def processPTX(fpath, rotationAboutZdegrees=None, searchRadius=None):

 
    if rotationAboutZdegrees is None:
        rotationAboutZdegrees = 150.
    if searchRadius is None:    
        searchRadius = 47
   
    with open(fpath, 'r') as f:
        ls = f.readlines()
    
    xyz = processPTXdata(ls, rotationAboutZdegrees, searchRadius)

    # write to CSV file
    outf = fpath + ".csv"
    np.savetxt(outf, xyz, delimiter=",")

def processPTXdata(lines, rotationAboutZdegrees, searchRadius, quiet=True):

    ls = lines

    if not quiet:
        print "Original file num lines: ", len(ls)

    #badLine = '0 0 0 0.500000\r\n'
    badLine = '0 0 0 0.500000\n'
    # reject the '0 0 0 0.500000' lines - TBF: what are those?
    ls = [l for l in ls if l != badLine]
    #for l in ls:
    #    print "'%s' vs '%s'" % (l, badLine)

    if not quiet:
        print "File minus '0 0 0 0.50000' lines: ", len(ls)

    #print "beginning of file: "
    #print ls[:12]

    # remove the header
    ls = ls[10:]

    #print "now beginning of file w/ out header: "
    #print ls[:12]

    # parse strings so we can ignore the fourth value
    xyz = []
    for l in ls:
        ll = l.split(' ')
        x = float(ll[0])
        y = float(ll[1])
        z = float(ll[2])
        xyz.append((x, y, z))

    xyz = np.array(xyz)    

    # rotation!  This takes a long time.  TBF: why? *)
    #rot=AffineTransform[{RotationMatrix[150Degree,{0,0,1}],{0,0,0}}];
    #lall=Map[rot,lall];
    xyz = rotateXYaboutZ(xyz, rotationAboutZdegrees)

    # and we only use those parts that are within this radius TBF? *)
    # ls=Select[ls,Norm[Take[#,2]-{-54,0}]<47&]
    # Here's what's going on:
    # Select will go through and apply the predicat # < 47 to each element of 'ls'
    # but, in this case each element, #, is passed to Norm[Take[#,2]-{-54, 0}]
    # What's happening there is that the x, y from each element in xyz above is taken,
    # and 54 is added to each x.  Norm == Sqrt(x**2 + y**2); that looks like a radius to me.
    # so, if the radius is less the 47, this data element is kept.

    return filterOutRadius(xyz, searchRadius=searchRadius)

def filterOutRadius(xyz, searchRadius=None, mysteryX=None):
    "return only those xyz points where sqrt(x**2 + y**2) is within a limit"

    if mysteryX is None:
        mysteryX = 54
    if searchRadius is None:    
        searchRadius = 47

    xyzInside = []
    for x, y, z in xyz:
        x2 = x + mysteryX
        r = np.sqrt(x2**2 + y**2)
        if r < searchRadius:
            xyzInside.append((x, y, z))
    xyzInside = np.array(xyzInside)

    return xyzInside

    

if __name__ == "__main__":

    #fpath = "/home/sandboxes/pmargani/LASSI/data/LeicaDropbox/PTX/Test1_STA14_Bump1_High-02_METERS.ptx"
    #fpath = sys.argv[1]
    #testRotation(90.)
    fpath = "data/Test1_STA14_Bump1_Hig-02_METERS_SAMPLE.ptx"
    processPTX(fpath)

