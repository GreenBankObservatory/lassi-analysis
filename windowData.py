import numpy as np

def windowData(x, y, z, N):

    print("windowing by: ", N)
    #print "x: "
    #print x
    #print "y: "
    #print y
    #print "z: "
    #print z

    xMin = np.min(x)
    xMax = np.max(x)
    yMin = np.min(y)
    yMax = np.max(y)

    xRange = np.linspace(xMin, xMax, N)
    yRange = np.linspace(yMin, yMax, N)

    xLoc, yLoc = np.meshgrid(xRange, yRange) 

    xys = np.array(zip(x, y))

    #print "xys: ", xys

    ws = {}
    wi = 0
    for ix in range(N-1):
        for iy in range(N-1):
            #print "ix, iy: ", ix, iy
            x0 = xRange[ix]
            x1 = xRange[ix+1]
            y0 = yRange[iy]
            y1 = yRange[iy+1]
            rng = ((x0, x1), (y0, y1))
            #print "rng: ", rng
            #xyi = [i for i, xy in enumerate(xys) if xy[0] >= x0 and xy[0] < x1 and xy[1] >= y0 and xy[1] < y1]
            xyi = [i for i, xy in enumerate(xys) if xy[0] >= x0 and xy[0] <= x1 and xy[1] >= y0 and xy[1] <= y1]
            #print "xyi: ", xyi
            xysf = xys[xyi]
            #xw = np.where(np.logical_and(x <= x1, x >x0))
            #yw = np.where(np.logical_and(y <= y1, y >y0))
            #print "xw: ", xw
            #print "yw: ", yw
            #zw = list(set(list(xw[0])).intersection(set(list(yw[0]))))
            #ws[wi] = (rng, (x[xw], y[yw], z[zw]))
            ws[wi] = ((ix, iy), rng, (xysf, z[xyi]))
            wi += 1

    return ws

def windowDataOld(x, y, z, N):

    print("windowing by: ", N)
    #print "x: "
    #print x
    #print "y: "
    #print y
    #print "z: "
    #print z

    xMin = np.min(x)
    xMax = np.max(x)
    yMin = np.min(y)
    yMax = np.max(y)

    xRange = np.linspace(xMin, xMax, N)
    yRange = np.linspace(yMin, yMax, N)

    xLoc, yLoc = np.meshgrid(xRange, yRange) 

    xys = zip(x, y)

    ws = {}
    wi = 0
    for ix in range(N-1):
        for iy in range(N-1):
            #print "ix, iy: ", ix, iy
            x0 = xRange[ix]
            x1 = xRange[ix+1]
            y0 = yRange[iy]
            y1 = yRange[iy+1]
            rng = ((x0, x1), (y0, y1))
            #print "rng: ", rng
            #xyi = [i for i, xy in enumerate(xyz)]
            xw = np.where(np.logical_and(x <= x1, x >x0))
            yw = np.where(np.logical_and(y <= y1, y >y0))
            #print "xw: ", xw
            #print "yw: ", yw
            zw = list(set(list(xw[0])).intersection(set(list(yw[0]))))
            print("zw: ", zw)
            ws[wi] = (rng, (x[xw], y[yw], z[zw]))
            #ws[wi] = (rng, (xysf, z[zw]))
            wi += 1
    return ws

def getWeight(az, el, azLoc, elLoc, sigAz, sigEl, j, k):
    return 2*np.pi*np.exp( (- (az - azLoc[j,k])**2 /( 2.*sigAz**2 )-(el-elLoc[j,k])**2 /(2.*sigEl**2 )))
    #return 2*np.pi*np.exp((-(np.cos(elLoc[j,k])**2)*(az-azLoc[j,k])**2/(2.*sigAz**2)-(el-elLoc[j,k])**2/(2.*sigEl**2 )))

def getWeightHere(az, el, azLoc, elLoc, sigAz, sigEl):
    return 2*np.pi*np.exp( (- (az - azLoc)**2 /( 2.*sigAz**2 )-(el-elLoc)**2 /(2.*sigEl**2 )))
    
def assignWeight(w, r):
    norm=sum(w)
    if norm==0:
        norm=1
        v=np.nan #0 #min( r )
        #v=0 #0 #min( r )
    else:
        w = w / norm
        v = sum(r * w)   
    return v

def smooth(az, el, r, n, sigEl=None, sigAz=None):
    "smooth our data"

    azRange = np.linspace(min(az), max(az), n)
    elRange = np.linspace(min(el), max(el), n)

    azLoc, elLoc = np.meshgrid(azRange, elRange)

    if sigEl is None:
        sigEl=0.001
    if sigAz is None:    
        sigAz=0.001;

    #import ipdb; ipdb.set_trace()
    # init our smoothing result
    #rSm = np.ndarray(shape=(n,n))
    rSm = np.zeros((n, n))
    rSms = []
    for j in range(n):
        #print "J:", j
        for k in range(n):
            #print "K:", k
            #w=2*np.pi*np.exp( (- (az - azLoc[j,k])**2 /( 2.*sigAz**2 )-(el-elLoc[j,k])**2 /(2.*sigEl**2 )))
            #norm=sum(w)
            #if norm==0:
            #    norm=1
            #    rSm[j,k]=np.nan #0 #min( r )
            #else:
            #    w = w / norm
            #    rSm[j,k] = sum(r * w)
            w = getWeight(az, el, azLoc, elLoc, sigAz, sigEl, j, k)
            #print w
            tmp = assignWeight(w, r)
            #norm=sum(w)
            #print "norm: ", norm
            #w = w / norm
            #print "w: ", w
            #print "r: ", r
            #tmp = sum(r*w)
            #print "tmp: ", tmp
            #print "rsm[j, k]: ", rSm[j, k]

            rSm[j, k] = tmp 
            
    return (azLoc, elLoc, rSm)  

def smoothWin(ws, sigAz=None, sigEl=None):


    if sigEl is None:
        sigEl=0.001
    if sigAz is None:    
        sigAz=0.001;

    n = len(ws.keys())
    #rSm = np.zeros((n, n))
    rSm = np.zeros((n,))

    for k, v in ws.items():
        ixy, rng, values = v
        xys, zs = values
        x = [xy[0] for xy in xys]
        y = [xy[1] for xy in xys]
        xRng = rng[0]
        yRng = rng[1]
        xCntr = xRng[0] + ((xRng[1] - xRng[0]) / 2.)
        yCntr = yRng[0] + ((yRng[1] - yRng[0]) / 2.)
        w = getWeightHere(x, y, xCntr, yCntr, sigAz, sigEl)
        rSm[int(k)] = assignWeight(w, zs)

    return rSm

def mainTest():

    m = 5
    xs = np.array(range(m), dtype=float)
    ys = np.array(range(m), dtype=float)

    xm, ym = np.meshgrid(xs, ys)

    print("xm: ")
    print(xm)
    print("ym: ")
    print(ym)

    zs = xm * ym
    
    #np.zeros((m,m))

    #ws = windowData(xs, ys, 10)
    #print ws

    n = 3
    xLoc, yLoc, zsmooth = smooth(xm.flatten(), ym.flatten(), zs.flatten(), n, sigAz=1., sigEl=1.)

    print("data: ")
    print(zs)

    print("smoothed to: ")
    print(zsmooth)

    ws = windowData(xm.flatten(), ym.flatten(), zs.flatten(), n+1)
    print(ws.keys())
    for k, v in ws.items():
        ixy, rng, values = v
        _, zzz = values
        print(rng, np.mean(zzz))

    zWinSmooth = smoothWin(ws, sigAz=1.0, sigEl=1.0)    

    print("zWinSmooth: ")
    print(zWinSmooth)

def main():

    m = 100
    xs = np.array(range(m), dtype=float)
    ys = np.array(range(m), dtype=float)

    xm, ym = np.meshgrid(xs, ys)

    zs = xm * ym
    
    n = 10
    xLoc, yLoc, zsmooth = smooth(xm.flatten(), ym.flatten(), zs.flatten(), n, sigAz=1., sigEl=1.)

    print("smoothed to: ")
    print(zsmooth)

    ws = windowData(xm.flatten(), ym.flatten(), zs.flatten(), n+1)
    print(ws.keys())
    for k, v in ws.items():
        ixy, rng, values = v
        _, zzz = values
        #print rng, np.mean(zzz)

    zWinSmooth = smoothWin(ws, sigAz=1.0, sigEl=1.0)    

    print("zWinSmooth: ")
    zWinSmooth.shape = (n, n)
    print(zWinSmooth)

if __name__=='__main__':
    main()
