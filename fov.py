import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asciidata
import astropy.coordinates as coord
import math
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def area(phiL, phiH, theL, theH):
    return (phiH - phiL)*(math.cos(theH) - math.cos(theL))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def area2():
    dtheta = theH-theL
    dx = 0.01
    n = dtheta/dx
    for i in range(0, n):
        A = area()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def stack(a, b):
	c = np.zeros((len(a),2))
	c[:,0], c[:,1] = a, b
	return c
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GASS = np.vstack((coords, coords_L))
COLDGASS = np.vstack((COLD_GASS_H, LMass))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x = np.array([-np.pi/3.25,-np.pi/2.86,np.pi/3,np.pi/3,-np.pi/3.25])
y = np.array([0,np.pi/5,np.pi/5,0,0])
A = stack(x,y)

x1 = np.array([np.pi/1.195,np.pi/1.195,np.pi,np.pi,np.pi/1.195])
y1 = np.array([0,np.pi/120,np.pi/120,0,0])
B = stack(x1,y1)

x2 = np.array([-np.pi/1.34,-np.pi/1.34,-np.pi,-np.pi,-np.pi/1.34])
y2 = np.array([0,np.pi/120,np.pi/120,0,0])
C = stack(x2,y2)

x3 = np.array([-np.pi/3.25,-np.pi/2.86,-np.pi/2.7,-np.pi/2.8,-np.pi/3.25])
y3 = np.array([0,np.pi/5,np.pi/6,np.pi/10,0])
D = stack(x3,y3)

x4 = np.array([-np.pi/1.23, -np.pi/1.23, -np.pi/1.1, -np.pi, -np.pi, -np.pi/1.1, -np.pi/1.23])
y4 = np.array([np.pi/15, np.pi/12.2, np.pi/11.1, np.pi/11, np.pi/13, np.pi/13.4, np.pi/15])
E = stack(x4,y4)

x5 = np.array([np.pi/1.2, np.pi/1.2, np.pi/1.1, np.pi, np.pi, np.pi/1.1, np.pi/1.2])
y5 = np.array([np.pi/16, np.pi/13, np.pi/11.6, np.pi/11, np.pi/13, np.pi/14, np.pi/16])
F = stack(x5,y5)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# print area(np.pi/1.195, np.pi, 0, np.pi/120)
A1 =  area(-np.pi/3, np.pi/3, 0, np.pi/5)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N = abs(A1*(float(len(COLDGASS))/float(len(GASS))))
print 1/N
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="mollweide")
ax.scatter(((GASS[:,1]-180)*(math.pi/180.0)), (GASS[:,2]*(math.pi/180.0)), s=0.1, alpha = 0.5, label = 'SDSS')
ax.scatter(((COLDGASS[:,1]-180)*(math.pi/180.0)), (COLDGASS[:,2]*(math.pi/180.0)), s=0.1, color='red', label = 'COLD GASS')
ax.plot(x,y,color='m', linewidth=0.3)
ax.plot(x1,y1,color='b', linewidth=0.3)
ax.plot(x2,y2,color='c', linewidth=0.3)
ax.plot(x3,y3,color='g', linewidth=0.3)
ax.plot(x4,y4, color ='k', linewidth=0.3)
ax.plot(x5,y5, color ='r', linewidth=0.3)
ax.grid(True)
ax.legend(bbox_to_anchor=(0.5,1.2), loc='upper center', ncol=2, fontsize = 13)
#ax.xaxis.set_major_formatter(plt.NullFormatter())
fig.set_size_inches(10,6)
plt.savefig('img/footprint.png', dpi=1000, transparent = False)
#plt.savefig('img/footprint.pdf', format='pdf', dpi=1000, transparent = False)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
