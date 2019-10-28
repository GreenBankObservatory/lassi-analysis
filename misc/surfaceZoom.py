import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches

def imagePlot(z, title):
    fig = plt.figure()
    ax = fig.gca()
    cax = ax.imshow(z)
    fig.colorbar(cax)
    plt.title(title)

def surfaceZoom(data, dataLog, x, y, w, h, title):

    # display the original log image, with rectangle
    f = plt.figure()
    ax = f.gca()
    cax = ax.imshow(dataLog)
    plt.colorbar(cax)
    plt.title("zoom context")
   
    # here's our rectangle
    rect = patches.Rectangle((x, y),w,h,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    # zoom into rectangle
    # TBF: why is this reveresed?
    #dataZm = data[x:(x+w), y:(y+h)]
    #dataLogZm = dataLog[x:(x+w), y:(y+h)]
    dataZm = data[y:(y+h), x:(x+w)]
    dataLogZm = dataLog[y:(y+h), x:(x+w)]

    # display zooms
    imagePlot(dataZm, title)
    imagePlot(dataLogZm, title + " log")

    # calculate and return mean of zoomed area
    return np.mean(dataZm), np.std(dataZm), dataZm.shape

def analyzeZooms(data, dataLog):

    # x y w h title
    boxes = [
        # top panel
        # y = 150; x=430; h= 25; w=20;
        (430, 150, 20, 25, "top panel"),
        # top surface
        # y=180; h=20; x=395; w=20;
        (395, 180, 20, 20, "top surface"),
        # middle panel
        # y=220; h=40; x=395; w=25
        (395, 220, 25, 40, "middle panel"),
        # bottom surface
        # y=280; h=20; x=400; w=20
        (400, 280, 20, 20, "bottom surface"),
        # bottom panel
        # y=315; h=20; x=430; w=20
        (430, 315, 20, 20, "bottom panel")
    ]

    rs = []
    means = []
    for x, y, w, h, title in boxes:
        mean, std, shape = surfaceZoom(data, dataLog, x, y, w, h, title)
        n = shape[0]*shape[1]
        stdn = std / np.sqrt(n)
        rs.append((title, mean, std, n, stdn))
        means.append(mean)

    
    # do them all together
    f = plt.figure()
    ax = f.gca()
    cax = ax.imshow(dataLog)
    plt.colorbar(cax)
    plt.title("displacements")

    #disps = [topPanelDisp, None, mdlPanelDisp2, None, btmPanelDisp]
    for i, box in enumerate(boxes):
        x, y, w, h, title = box
        rect = patches.Rectangle((x, y),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, "%e" % means[i])

    for title, mean, std, n, stdn in rs:
        print("%s: mean=%e std=%e N=%e stdN=%e" % (title, mean, std, n, stdn))

    return rs    

def calculateVertexDisplacements(data, dataLog):


    rs = analyzeZooms(data, dataLog)

    topPanel = 0
    topSurf = 1
    mdlPanel = 2
    btmSurf = 3
    btmPanel = 4

    diffs = [(topPanel, topSurf), (mdlPanel, topSurf), (btmPanel, btmSurf)]
    titles = ["top panel", "middle panel", "bottom panel"]
    i = 0
    for pnl, srf in diffs:
        tPnl, meanPnl, stdPnl, nPnl, stdNpnl = rs[pnl]
        tSrf, meanSrf, stdSrf, nSrf, stdNsrf = rs[srf]
       
        disp = (meanSrf - meanPnl) * 1e6
        dispSig = np.sqrt(stdNpnl**2 + stdNsrf**2) * 1e6

        print("%s: disp(microm)=%e stdN=%e" % (titles[i], disp, dispSig))
        i += 1

