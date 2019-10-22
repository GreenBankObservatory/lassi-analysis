from copy import copy
import os

import numpy as np

import settings
from parabolas import fitLeicaScan, loadLeicaDataFromGpus
from plotting import imagePlot, scatter3dPlot, surface3dPlot
from main import smoothXYZGpu

def log(x):
    return np.log(np.abs(x))

def difflog(x):
    return np.log(np.abs(np.diff(x)))

def log10(x):
    return np.log10(np.abs(x))

def testApplyWeights(smoothFile, weightFile):
    print("testApplyWeights: ", smoothFile, weightFile)

    N = 512

    fileBasename = os.path.basename(smoothFile)

    ws = np.load(weightFile)
    ws.shape = (N, N)
    imagePlot(log(ws), "weights log")

    xs, ys, zs = loadLeicaDataFromGpus(smoothFile)

    # our weights need to be weighted by radius**4
    r2 = xs**2 + ys**2 + zs**2
    r2.shape = (N, N)
    wr4 = ws * r2**2

    wr = wr4
    # wr = wr4/np.nansum(wr4)

    xsc = copy(xs)
    xsc.shape = (N, N)
    ysc = copy(ys)
    ysc.shape = (N, N)

    # surface3dPlot(xsc, ysc, wr, "new weights")
    # surface3dPlot(xsc, ysc, log(wr), "new log weights")
    # surface3dPlot(xsc, ysc, 1/wr, "new 1 / weights")


    # xs, ys, zs = loadLeicaDataFromGPUs(smoothFile)
    diff, xf, yf = fitLeicaScan(smoothFile,
                                numpy=False,
                                # rFilter=True)
                                weights=wr)

    imagePlot(difflog(diff), "diff log")

    # calculate some stuff
    diffSqMn = (diff - np.nanmean(diff))**2

    sumWs = np.nansum(ws)
    # sigmaWeight2 = np.nansum(ws*(diff**2))
    sigmaWeight2 = np.nansum(ws*diffSqMn)
    sigma2 = np.nansum(diffSqMn)

    # print "sum of diff", np.nansum(diff)
    print("sum of weights", sumWs)
    print("sigmaWeight: ", np.sqrt(sigmaWeight2))
    print("sigma", np.sqrt(sigma2))

    # imagePlot(sigmaWeight2, "sigma weight squared")
    # imagePlot(sigma2, "sigma squared")
    # imagePlot(sigma2 - sigmaWeight2, "difference in sigmas")


    imagePlot(log10(wr), "wr log")

    wrc = copy(wr)
    wrc[wrc < 1.0] = np.nan

    imagePlot(log10(wrc), "weighted weights")

    diffw = copy(diff)
    diffw[wr < 1.0] = np.nan
    imagePlot(difflog(diffw), "diffw log")

    print("Comparing RMS: ")
    print("orginal diff; mean, std:", np.nanmean(diff), np.nanstd(diff))
    print("masked  diff; mean, std:", np.nanmean(diffw), np.nanstd(diffw))

    scatter3dPlot(xf.flatten(), yf.flatten(), diffw.flatten(), "diffw")
    surface3dPlot(xf, yf, diffw, "diffw")

    xstr = 215 - 20
    xend = 315 - 20
    ystr = 215
    yend = 315

    diffCnt = copy(diff[xstr:xend,ystr:yend])
    diffwCnt = copy(diffw[xstr:xend,ystr:yend])
    xf.shape = (N, N)
    yf.shape = (N, N)

    wsCnt = ws[xstr:xend, ystr:yend]

    xfCnt = copy(xf[xstr:xend, ystr:yend])
    yfCnt = copy(yf[xstr:xend, ystr:yend])

    print("Comparing RMS of center region: ")
    print("orginal diff; mean, std:", np.nanmean(diffCnt), np.nanstd(diffCnt))
    print("masked  diff; mean, std:", np.nanmean(diffwCnt), np.nanstd(diffwCnt))

    scatter3dPlot(xfCnt.flatten(), yfCnt.flatten(), diffCnt.flatten(), "diff cntr")
    surface3dPlot(xfCnt, yfCnt, diffCnt, "diff cnt")

    scatter3dPlot(xfCnt.flatten(), yfCnt.flatten(), diffwCnt.flatten(), "diffw cntr")
    surface3dPlot(xfCnt, yfCnt, diffwCnt, "diffw cnt")
    imagePlot(difflog(diffCnt), "diff cnt")
    imagePlot(difflog(diffwCnt), "diffw cnt")

   # calculate some stuff
    diffSqMn = (diffCnt - np.nanmean(diffCnt))**2

    # sigmaWeight2 = np.nansum(ws*(diff**2))
    sigmaWeight2 = np.nansum(wsCnt*diffSqMn)
    sigma2 = np.nansum(diffSqMn)

    # print "sum of diff", np.nansum(diff)
    print("sigmaWeight: ", np.sqrt(sigmaWeight2))
    print("sigma", np.sqrt(sigma2))

    return xs, ys, zs, xf, yf, diff, diffw, ws, wr

    # print "Regriding data ..."
    # filename = "%s.regrid" % fileBasename
    # xs2, ys2, diffs2 = smoothXYZGpu(xf, yf, diff, N, filename=filename)

    # xs2.shape = ys2.shape = diffs2.shape = (N, N)

    # imagePlot(diffs2, "Regridded Diff")

    # # diffsLog = np.log(np.abs(np.diff(diffs)))
    # imagePlot(difflog(diffs2), "Regridded Diff Log")

    # return xs, ys, zs, xf, yf, diff, ws, wr, xs2, ys2, diffs2
    
def reprocessScanPair(fn1, weightsFn1, fn2, weightsFn2):

    assert os.path.isfile(fn1 + ".x.csv")
    assert os.path.isfile(fn2 + ".x.csv")        
    assert os.path.isfile(weightsFn1)
    assert os.path.isfile(weightsFn2)

    r1 = testApplyWeights(fn1, weightsFn1)
    diffs1 = r1[-1]        

    r2 = testApplyWeights(fn2, weightsFn2)
    diffs2 = r2[-1]        

    diff = diffs2 - diffs1
    imagePlot(log(diff), "log two scan diff")

    print("mean, std: ", np.nanmean(diff), np.nanstd(diff))
    return diff


def testScanPair():
    fn9 = "Clean9.ptx.csv"
    fn9 = os.path.join(settings.GPU_PATH, fn9) 
    wfn9 = "weightsScan9.npy"

    fn11 = "Clean11.ptx.csv"
    fn11 = os.path.join(settings.GPU_PATH, fn11) 
    wfn11 = "weightsScan11.npy"

    return reprocessScanPair(fn9, wfn9, fn11, wfn11)

def main():
    fn = "Clean11.ptx.csv"
    # fn = "Clean9.ptx.csv"
    smoothFiles = os.path.join(settings.GPU_PATH, fn) 
    wfn = "weightsScan11.npy"
    # wfn = "weightsScan9.npy"
    testApplyWeights(smoothFiles, wfn)

if __name__=='__main__':
    main()
