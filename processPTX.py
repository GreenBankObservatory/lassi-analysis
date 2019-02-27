import sys

import numpy as np

def processPTX(fpath):

    with open(fpath, 'r') as f:
        ls = f.readlines()

    print "Original file num lines: ", len(ls)

    badLine = '0 0 0 0.500000\r\n'
    # reject the '0 0 0 0.500000' lines - TBF: what are those?
    ls = [l for l in ls if l != badLine]

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

    # and we only use those parts that are within this radius TBF? *)
    # ls=Select[ls,Norm[Take[#,2]-{-54,0}]<47&]
    # Here's what's going on:
    # Select will go through and apply the predicat # < 47 to each element of 'ls'
    # but, in this case each element, #, is passed to Norm[Take[#,2]-{-54, 0}]
    # What's happening there is that the x, y from each element in xyz above is taken,
    # and 54 is added to each x.  Norm == Sqrt(x**2 + y**2); that looks like a radius to me.
    # so, if the radius is less the 47, this data element is kept.
    mysteryX = 54
    mysteryRadius = 47
    xyzInside = []
    for x, y, z in xyz:
        x += mysteryX
        r = np.sqrt(x**2 + y**2)
        if r < mysteryRadius:
            xyzInside.append((x, y, z))
    xyzInside = np.array(xyzInside)

    # write to CSV file
    outf = fpath + ".csv"
    numpy.savetxt(outf, xyzInside, delimiter=",")
    

if __name__ == "__main__":
    fpath = sys.argv[1]
    processPTX(fpath)
