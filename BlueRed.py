import numpy as np
import math
import matplotlib.pyplot as plt
import schechter
import random

def log_schechter(logL, log_rho, log_Lstar, alpha):
    rholist = []
    L = 10**logL
    pstar = 10**log_rho
    Lstar = 10**log_Lstar
    log = np.log(10)
    for i in range(0,len(L)):
        frac = L[i]/Lstar
        rho = pstar*(frac**(alpha+1))*math.exp(-frac)*log
        rholist.append(rho)
    return np.log10(rholist)

def schechter(L, Ls, dL, phis, alph):
    philist = []
    for i in range(0,len(L)):
        frac = 10**(L[i]-Ls)
        exp = math.exp(-frac)
        phibit = (phis*(frac**(alph+1)))
        log = np.log(10)
        philist.append((phibit*exp*log))
    return philist

def doubleschechter(L, Ls, dL, phi1, phi2, alph1, alph2):
    philist = []
    for i in range(0,len(L)):
        frac = 10**(L[i]-Ls)
        exp = math.exp(-frac)
        phibit1 = (phi1*(frac**(alph1+1)))
        phibit2 = (phi2*(frac**(alph2+1)))
        log = np.log(10)
        philist.append(exp*log*(phibit1+phibit2))
    return philist

d = 100
L = np.linspace(7.1,11.9,25)
LKeres = np.linspace(4,8,200)

Spheroid = (3.67/10000), 10.74, -0.525
Disk = (0.855/10000), 10.70, -1.39
Keres = (7.2/10000), 7.0, -1.30

#spheroid is red
ySpheroid = log_schechter(L, *Spheroid)
#disk is blue
yDisk = log_schechter(L, *Disk)
yKeres = schechter(LKeres, Keres[1], (L[1]-L[0]), Keres[0], Keres[2])
yBaldry = doubleschechter(L, 10.66, (L[1]-L[0]), 0.00396, 0.00079, -0.35, -1.47)
yred = schechter(L, 10.66, (L[1]-L[0]), 0.00396, -0.35)
yblue = schechter(L, 10.66, (L[1]-L[0]), 0.00079, -1.47)

###### make a table for the red galaxies #######################################
red = np.zeros((len(yred),4))
red[:,0] = L
red[:,1] = 0.2
red[:,2] = yred
redpop = []
for i in range(0,len(yred)):
    red[i,3] = int(red[i,2]*d*d*d)
    for j in range(0,int(red[i,3])):
        redpop.append(random.uniform((red[i,0]+(0.5*red[i,1])), (red[i,0]+(0.5*red[i,1]))))

###### make a table for the blue galaxies ######################################
blue = np.zeros((len(yblue),4))
blue[:,0] = L
blue[:,1] = 0.2
blue[:,2] = yblue
bluepop = []
for i in range(0,len(yblue)):
    blue[i,3] = int(blue[i,2]*d*d*d)
    for j in range(0,int(blue[i,3])):
        bluepop.append(random.uniform((blue[i,0]+(0.5*blue[i,1])), (blue[i,0]+(0.5*blue[i,1]))))
###### make a table for all the galaxies #######################################
Baldry = np.zeros((len(yBaldry),4))
Baldry[:,0] = L
Baldry[:,1] = 0.2
Baldry[:,2] = yBaldry

print red

# fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False)
# #plt.plot(L, ySpheroid, 'r')
# #plt.plot(L, yDisk, 'b')
# #ax[0,0].plot(LKeres, yKeres, 'g')
# ax[0,0].plot(L,np.log10(yBaldry), 'k')
# ax[0,0].plot(L,np.log10(yred), 'r')
# ax[0,0].plot(L,np.log10(yblue), 'b')
# ax[0,0].set_xlabel(r'$log_{10}(M/M_{\odot})$', fontsize = 20)
# ax[0,0].set_ylabel(r'$log_{10}(N density)$', fontsize = 20)

n, bins, patches = plt.hist(redpop, 25, normed=1, facecolor='red', alpha=0.5)
n1, bins1, patches1 = plt.hist(bluepop, 25, normed=1, facecolor='blue', alpha=0.5)
# plt.xlim(8,11.7)
# plt.ylim(-5,0)
plt.show()
