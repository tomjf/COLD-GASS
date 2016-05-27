import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
import asciidata
import astropy.coordinates as coord
import math

def plot_mwd(RA,Dec,org=0,title='Mollweide projection', projection='mollweide'):
    ''' RA, Dec are arrays of the same length.
    RA takes values in [0,360), Dec in [-90,90],
    which represent angles in degrees.
    org is the origin of the plot, 0 or a multiple of 30 degrees in [0,360).
    title is the title of the figure.
    projection is the kind of projection: 'mollweide', 'aitoff', 'hammer', 'lambert'
    '''
    x = np.remainder(RA+360-org,360) # shift RA values
    ind = x>180
    x[ind] -=360    # scale conversion to [-180, 180]
    x=-x    # reverse the scale: East to the left
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360+org,360)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection=projection, axisbg ='LightCyan')
    ax.scatter(np.radians(x),np.radians(Dec))  # convert degrees to radians
    ax.set_xticklabels(tick_labels)     # we add the scale on the x axis
    ax.set_title(title)
    ax.title.set_fontsize(15)
    ax.set_xlabel("RA")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("Dec")
    ax.yaxis.label.set_fontsize(12)
    ax.grid(True)



df = pd.read_csv('data/PS_100701.csv')
coords = df[['GASS', 'ra', 'dec']].values
df = pd.read_csv('data/LOWMASS_MASTER.csv')
coords_L = df[['GASSID', 'RA', 'DEC']].values
df = pd.read_csv('data/COLDGASS_DR3_with_Z.csv')
COLD_GASS_H = df[['GASSID', 'RA', 'DEC']].values

lowM = asciidata.open('COLDGASS_LOW_29Sep15.ascii')
LMass = np.zeros((len(lowM[12]),3))
LMass[:,0] = list(lowM[0])                         # S_CO
LMass[:,1] = list(lowM[1])                               # z
LMass[:,2] = list(lowM[2])                        # flag

#ra = [x / 180.0 * 3.141593 for x in ra]

# ax = plt.subplot(111, projection='polar')
# ax.scatter(coords[:,1], coords[:,2], color='k',s=0.2, alpha=0.3)
# ax.scatter(coords_L[:,1], coords_L[:,2], color='k',s=0.2, alpha=0.3)
# ax.scatter(COLD_GASS_H[:,1], COLD_GASS_H[:,2], color='r',s=3)
# ax.scatter(LMass[:,1], LMass[:,2], color='r',s=3)
# #ax.set_rmax(2.0)
# ax.grid(True)
# plt.show()


# r = 90
# theta = np.arange(0,2*np.pi,0.1)
# x = np.array([0,np.pi/3,np.pi/3,0,0])
# y = np.array([0,0,np.pi/4,np.pi/4,0])
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111, projection="mollweide", axisbg ='LightCyan')
# ax.grid(True)
# plot_mwd(coords[:,1], coords[:,2])
#
# plt.show()

# ra = coord.Angle(coords[:,1].filled(np.nan)*u.degree)
# ra = ra.wrap_at(180*u.degree)
# dec = coord.Angle(coords[:,2].filled(np.nan)*u.degree)
GASS = np.vstack((coords, coords_L))
COLDGASS = np.vstack((COLD_GASS_H, LMass))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="mollweide")
ax.scatter(((GASS[:,1]-180)*(math.pi/180.0)), (GASS[:,2]*(math.pi/180.0)), s=0.1, alpha = 0.5, label = 'GASS')
ax.scatter(((COLDGASS[:,1]-180)*(math.pi/180.0)), (COLDGASS[:,2]*(math.pi/180.0)), s=2, color='red', label = 'COLD GASS')
ax.grid(True)
ax.legend(bbox_to_anchor=(0.5,1.2), loc='upper center', ncol=2, fontsize = 13)
fig.set_size_inches(10,6)
plt.savefig('img/footprint.pdf', format='pdf', dpi=1000, transparent = False)
