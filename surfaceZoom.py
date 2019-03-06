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
    return np.mean(dataZm)
