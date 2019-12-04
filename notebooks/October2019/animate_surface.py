import pickle
import numpy as np
import pylab as plt
from matplotlib import rc, animation
from IPython.display import HTML
from matplotlib.dates import DateFormatter

from natsort import natsorted
from astropy import units as u
from astropy.time import Time, TimeDelta

rc('animation', html='jshtml')
rc('animation', embed_limit=20971520.0*4)

def show_animated_surface(surface_file='October9_surface_maskRadius49_reference_409_v1.npz',
                          surface2show='corrected_surface',
                          scans_dict_file='October9_surfaceRMS_maskRadius49_reference_409_v3b.pickle',
                          color_map='hot',
                          sunrise_str='2019-10-11 11:24:00',
                          sunset_str='2019-10-10 22:49:00',
                          utc_to_east = TimeDelta(-4.*3600.*u.s),
                          date_formatter=DateFormatter('%m/%d %H:%M'),
                          scan_dict_file='../../data/Oct2019/lassiScans9oct2019_v3.pickle'):
    """
    Creates an animation of the surface and shows the surface rms for a given scan.
    This works well with the October 2019 scans.
    """

    sunrise = Time(sunrise_str, format='iso', scale='utc')
    sunset = Time(sunset_str, format='iso', scale='utc')

    # Load the surface data.
    surface = np.load(surface_file)

    # Load the dictionary with the surface rms
    rmsDict = pickle.load(open(scans_dict_file, 'rb'))

    # Load the pickle containing the information about the October 9 scans.
    scanDict = pickle.load( open(scan_dict_file, "rb"), encoding='bytes')
    keys = np.sort(np.asarray(list(scanDict.keys())))

    # Extract the surface to show, its coordinates and the mask.
    surf = surface[surface2show]
    xx = surface['surface'][:,0]
    yy = surface['surface'][:,1]
    mask = surface['mask']

    # Set limits for the colobar.
    vmin = np.nanmin(surf)*0.1
    vmax = np.nanmax(surf)*0.1
    print("Color bar limits vmin: {0:.3e} m vmax: {1:.3e} m".format(vmin, vmax))
    
    # Set the color map.
    cmap = plt.get_cmap(color_map)

    # Extract the time and surface rms values.
    rms_vals = np.zeros((surf.shape[0]), dtype=np.float)
    times = Time(np.zeros((surf.shape[0]), dtype=np.float), format='mjd')
    for i,k in enumerate(sorted(list(rmsDict.keys()))):
        rms_vals[i] = rmsDict[k]['corrected surface rms with all masks']
        times[i] = scanDict[k]['t0'].mjd

    # Plot
    fig, ax = plt.subplots(6, 2, figsize=(4,5.5), dpi=150)

    gs = ax[0,0].get_gridspec()
    # remove the underlying axes
    for ax_ in ax[:,:]:
        for ax__ in ax_:
            ax__.remove()

    ax0 = fig.add_subplot(gs[:-2,:])
    ax1 = fig.add_subplot(gs[-2,:])

    # Top panels with surface.
    #ax0 = ax[0]

    im = ax0.imshow(np.ma.masked_where(mask[0], surf[0]), origin='lower', vmin=vmin, vmax=vmax, cmap=cmap, 
                    extent=[np.nanmin(xx[0]), np.nanmax(xx[0]), np.nanmin(yy[0]), np.nanmax(yy[0])])

    ax0.minorticks_on()
    ax0.tick_params('both', direction='in', which='both', top=True, right=True, left=True, bottom=True)
    ax0.set_xlabel('x offset (m)')
    ax0.set_ylabel('y offset (m)')

    # Bottom panels with surface rms.
    #ax1 = ax[1]

    ax1.plot_date((times[:-1] + utc_to_east).plot_date, rms_vals[:-1]/np.sqrt(2.), 'r.')
    ax1.semilogy()

    ax1.set_ylim(0.9e-4, 4e-3)
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()

    line = ax1.plot_date([(times[:-1] + utc_to_east).plot_date[0]]*2, ylim, 'y-')
    line = line[0]

    ax1.plot_date([(sunset + utc_to_east).plot_date]*2, ylim, ':', color='#f78112')
    ax1.plot_date([(sunrise + utc_to_east).plot_date]*2, ylim, ':', color='#a412f7')

    # Limit on acceptable surface rms (Frayer et al. 2018).
    ax1.plot(xlim, [230e-6]*2, 'k:', lw=0.8)

    ax1.minorticks_on()
    ax1.tick_params('both', direction='in', which='both', top=True, right=True, left=True, bottom=True)
    ax1.set_ylabel('Surface rms (m)')
    ax1.set_xlabel('Time (ET)')

    ax1.xaxis.set_major_formatter(date_formatter)
    locmin = plt.matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=10) 
    ax1.yaxis.set_minor_locator(locmin)

    fig.tight_layout(pad=-1.8)
    fig.subplots_adjust(left=0.16, bottom=0, right=1, top=1, wspace=None, hspace=None)

    for label in ax1.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(30)
    #fig.autofmt_xdate()

    #plt.show();

    def update_surface(frame):
        """
        Updates the values of the surface's image for a given frame.
        """
        
        #fig.suptitle(time[frame])
        im.set_array(surf[frame])
        im.set_extent([np.nanmin(xx[frame]), np.nanmax(xx[frame]), np.nanmin(yy[frame]), np.nanmax(yy[frame])])
        
        line.set_data([(times[:-1] + utc_to_east).plot_date[frame]]*2, ylim)

    # Make a movie of the surface
    ani = animation.FuncAnimation(fig, update_surface, frames=len(surf)-1, interval=100, repeat=True)
    #HTML(ani.to_jshtml())
    return ani
