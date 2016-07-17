import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import asciidata
import astropy.coordinates as coord
import math
from spherical_geometry import polygon as poly
import random
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
def convertspher(data):
    a = np.zeros((len(data),3))
    for i in range(0, len(data)):
        a[i,0] = (math.cos(data[i,1]))*(math.sin(data[i,0]))
        a[i,1] = (math.sin(data[i,1]))*(math.sin(data[i,0]))
        a[i,2] = math.cos(data[i,0])
    return a
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def areaS(A, inside):
    A = convertspher(A)
    inside = convertspher(inside)
    A = poly.SphericalPolygon(A,inside)
    area = A.area()
    return area
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def totalArea(data):
    area = 0
    for i in range(0,len(data)):
        area += data[i][2]
    return area
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calcArea(phiL, phiH, thetaL, thetaH):
    phis = np.linspace(phiL, phiH, 500)
    Area = 0
    for i in range(0, len(phis)-1):
        dphi = abs(phis[i+1]-phis[i])
        thetacpt = abs(-math.cos(thetaH) + math.cos(thetaL))
        dA = dphi*thetacpt
        Area += dA
    return Area
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def interpolateEdge(phiL, phiR, thetaL, thetaR):
    phis = np.linspace(phiL, phiR, 20)
    thetas = []
    dtheta = thetaR-thetaL
    for i in range(0,len(phis)):
        frac = float(i)/float(len(phis))
        thetas.append(thetaL + (frac*dtheta))
    return phis, thetas

def interp(phis, xcoords, ycoords):
    thetas = []
    for index, phi in enumerate(phis):
        for i in range(0,len(xcoords)-1):
            if phi >= xcoords[i] and phi < xcoords[i+1]:
                dtheta = ycoords[i+1] - ycoords[i]
                theta = ycoords[i] + ((phi - xcoords[i])/(xcoords[i+1]-xcoords[i]))*dtheta
                thetas.append(theta)
        if phi == xcoords[-1]:
            thetas.append(ycoords[-1])
    return thetas
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def interpolateEdge2(top, bottom, n):
    topx, topy = top
    bottomx, bottomy = bottom
    phis = np.linspace(topx[0], topx[-1], n)
    topthetas = interp(phis, topx, topy)
    bottomthetas = interp(phis, bottomx, bottomy)
    return phis, topthetas, bottomthetas
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calcArea2(phis, top, bottom):
    Area = 0
    for i in range(0,len(phis)-1):
        phiL = phis[i]
        phiH = phis[i+1]
        thetaL = (bottom[i] + bottom[i+1])/2
        thetaH = (top[i] + top[i+1])/2
        dphi = abs(phis[i+1]-phis[i])
        thetacpt = abs(-math.cos(thetaH) + math.cos(thetaL))
        dA = dphi*thetacpt
        Area += dA
    return Area
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
areas = []
areas2 = []
print len(GASS)
################################################################################
# dark blue area
x = np.array([-np.pi/3.25,-np.pi/3.25,np.pi/3.0,np.pi/3.0,-np.pi/3.25])
y = np.array([0,np.pi/5,np.pi/5,0,0])
A = stack(x,y)
Ainside = stack(np.array([np.pi/10]), np.array([np.pi/10]))
areas.append([A, Ainside, abs(area(-np.pi/3.25, np.pi/3, 0, np.pi/5))])
areas2.append(calcArea(min(x), max(x), min(y), max(y)))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# green area
x1 = np.array([np.pi/1.195,np.pi/1.195,np.pi,np.pi,np.pi/1.195])
y1 = np.array([0,np.pi/120,np.pi/120,0,0])
B = stack(x1,y1)
Binside = stack(np.array([np.pi/1.1]), np.array([np.pi/200]))
areas.append([B, Binside, areaS(B, Binside)])
areas2.append(calcArea(min(x1), max(x1), min(y1), max(y1)))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# red area
x2 = np.array([-np.pi/1.34,-np.pi/1.34,-np.pi,-np.pi,-np.pi/1.34])
y2 = np.array([0,np.pi/120,np.pi/120,0,0])
C = stack(x2,y2)
Cinside = stack(np.array([-np.pi/1.1]), np.array([np.pi/200]))
areas.append([C, Cinside, areaS(C, Cinside)])
areas2.append(calcArea(min(x2), max(x2), min(y2), max(y2)))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# cyan area
x3 = np.array([-np.pi/3.25,-np.pi/3.25, -np.pi/2.86,-np.pi/2.7,-np.pi/2.8,-np.pi/3.25])
y3 = np.array([0,np.pi/5, np.pi/5,np.pi/6,np.pi/10,0])
D = stack(x3,y3)
Dinside = stack(np.array([-np.pi/3]), np.array([np.pi/10]))
areas.append([D, Dinside, areaS(D, Dinside)])
top3 = [[-np.pi/2.7, -np.pi/2.86, -np.pi/3.25], [np.pi/6, np.pi/5, np.pi/5]]
bottom3 = [[-np.pi/2.7, -np.pi/2.8, -np.pi/3.25], [np.pi/6, np.pi/10, 0]]
phis3, top3, bottom3 = interpolateEdge2(top3, bottom3, 500)
areas2.append(calcArea2(phis3, top3, bottom3))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# purple area
x4 = np.array([-np.pi/1.23, -np.pi/1.23, -np.pi/1.1, -np.pi, -np.pi, -np.pi/1.1, -np.pi/1.23])
y4 = np.array([np.pi/15, np.pi/12.2, np.pi/11.1, np.pi/11, np.pi/13, np.pi/13.4, np.pi/15])
E = stack(x4,y4)
Einside = stack(np.array([-np.pi/1.1]), np.array([np.pi/13]))
areas.append([E, Einside, areaS(E, Einside)])
top4 = [[-np.pi, -np.pi/1.1, -np.pi/1.23], [np.pi/11, np.pi/11.1, np.pi/12.2]]
bottom4 = [[-np.pi, -np.pi/1.1, -np.pi/1.23], [np.pi/13, np.pi/13.4, np.pi/15]]
phis4, top4, bottom4 = interpolateEdge2(top4, bottom4, 500)
areas2.append(calcArea2(phis4, top4, bottom4))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# yellow area
x5 = np.array([np.pi/1.2, np.pi/1.2, np.pi/1.1, np.pi, np.pi, np.pi/1.1, np.pi/1.2])
y5 = np.array([np.pi/16, np.pi/13, np.pi/11.6, np.pi/11, np.pi/13, np.pi/14, np.pi/16])
F = stack(x5,y5)
Finside = stack(np.array([np.pi/1.15]), np.array([np.pi/13]))
areas.append([F, Finside, areaS(F, Finside)])
top5 = [[np.pi/1.2, np.pi/1.1, np.pi], [np.pi/13, np.pi/11.6, np.pi/11]]
bottom5 = [[np.pi/1.2, np.pi/1.1, np.pi], [np.pi/16, np.pi/14, np.pi/13]]
phis5, top5, bottom5 = interpolateEdge2(top5, bottom5, 500)
areas2.append(calcArea2(phis5, top5, bottom5))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x6 = np.array([0, 0, np.pi, np.pi, 0])
y6 = np.array([0, np.pi/2, np.pi/2, 0, 0])
G = stack(x6,y6)
areaA = calcArea(0, np.pi/2, 0, np.pi)/np.pi
areaB = calcArea(0, np.pi/3.0, 0, np.pi/5)
areaC = calcArea(-np.pi/3.25, 0, 0, np.pi/5)
areaD = calcArea(-np.pi/3.25, np.pi/3.0, 0, np.pi/5)
areaE = calcArea(np.pi/1.195, np.pi, 0, np.pi/120)
areaF = calcArea(0, 2*np.pi, 0, np.pi)
print areaA, areaB, areaC, areaD, areaD-(areaB+areaC)
print areaE, areaF/np.pi
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# print area(np.pi/1.195, np.pi, 0, np.pi/120)
A1 =  area(-np.pi/3, np.pi/3, 0, np.pi/5)
totA = totalArea(areas)
print totA
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
phis, thetas = interpolateEdge(-np.pi/2.8, -np.pi/3.25, np.pi/10, 0)
topcoords = [[-np.pi/2.7, -np.pi/2.86, -np.pi/3.25], [np.pi/6, np.pi/5, np.pi/5]]
bottomcoords = [[-np.pi/2.7, -np.pi/2.8, -np.pi/3.25], [np.pi/6, np.pi/10, 0]]
phis2, top2, bottom2 = interpolateEdge2(topcoords, bottomcoords, 500)
print '@@@@', calcArea2(phis2, top2, bottom2)
print '1@@@@', np.sum(areas2)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="mollweide")
ax.scatter(((GASS[:,1]-180)*(math.pi/180.0)), (GASS[:,2]*(math.pi/180.0)), s=0.1, alpha = 0.5, label = 'SDSS')
ax.scatter(((COLDGASS[:,1]-180)*(math.pi/180.0)), (COLDGASS[:,2]*(math.pi/180.0)), s=0.1, color='red', label = 'COLD GASS')
for i in range(0,len(areas)):
    ax.plot(areas[i][0][:,0],areas[i][0][:,1], linewidth=1)
    ax.text(areas[i][1][:,0], areas[i][1][:,1], str(round(areas[i][2],4)), color = 'r', fontsize=12)
ax.plot(x6,y6, linewidth=1)
ax.scatter(phis2, top2, s=1)
ax.scatter(phis2, bottom2, s=1)
ax.grid(True)
ax.legend(bbox_to_anchor=(0.5,1.2), loc='upper center', ncol=2, fontsize = 13)
#ax.xaxis.set_major_formatter(plt.NullFormatter())
fig.set_size_inches(10,6)
plt.savefig('img/footprint.png', dpi=250, transparent = False)
plt.savefig('img/footprint.pdf', format='pdf', dpi=350, transparent = False)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
