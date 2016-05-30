import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asciidata
import astropy.coordinates as coord
import math

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

GASS = np.vstack((coords, coords_L))
COLDGASS = np.vstack((COLD_GASS_H, LMass))

x = np.array([-np.pi/3.25,-np.pi/2.86,np.pi/3,np.pi/3,-np.pi/3.25])
y = np.array([0,np.pi/5,np.pi/5,0,0])

x1 = np.array([np.pi/1.195,np.pi/1.195,np.pi,np.pi,np.pi/1.195])
y1 = np.array([0,np.pi/120,np.pi/120,0,0])

x2 = np.array([-np.pi/1.34,-np.pi/1.34,-np.pi,-np.pi,-np.pi/1.34])
y2 = np.array([0,np.pi/120,np.pi/120,0,0])

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="mollweide")
ax.scatter(((GASS[:,1]-180)*(math.pi/180.0)), (GASS[:,2]*(math.pi/180.0)), s=0.1, alpha = 0.5, label = 'SDSS')
ax.scatter(((COLDGASS[:,1]-180)*(math.pi/180.0)), (COLDGASS[:,2]*(math.pi/180.0)), s=0.1, color='red', label = 'COLD GASS')

ax.scatter(np.pi/1.195, np.pi, color='m', s=0.1)
ax.scatter(np.pi/1.195, np.pi/1.05, color='m', s=0.1)

ax.plot(x,y,color='m', linewidth=0.3)
ax.plot(x1,y1,color='m', linewidth=0.3)
ax.plot(x2,y2,color='m', linewidth=0.3)
ax.grid(True)
ax.legend(bbox_to_anchor=(0.5,1.2), loc='upper center', ncol=2, fontsize = 13)
#ax.xaxis.set_major_formatter(plt.NullFormatter())
fig.set_size_inches(10,6)
plt.savefig('img/footprint.pdf', format='pdf', dpi=1000, transparent = False)
