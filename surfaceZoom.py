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
    return np.mean(dataZm), np.std(dataZm)

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
        mean, std = surfaceZoom(data, dataLog, x, y, w, h, title)
        rs.append((title, mean, std))
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

    for title, mean, std in rs:
        print "%s: mean=%e std=%e" % (title, mean, std)
