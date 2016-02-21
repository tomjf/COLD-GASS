import numpy as np
import math
import matplotlib.pyplot as plt
import schechter

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
    print rholist
    return np.log10(rholist)


L = np.linspace(8,11.7,200)

Spheroid = (3.67/10000), 10.74, -0.525
Disk = (0.855/10000), 10.70, -1.39
Keres = (7.2/10000), 7, -1.30

print Spheroid

#spheroid is red
ySpheroid = log_schechter(L, *Spheroid)
#disk is blue
yDisk = log_schechter(L, *Disk)
yKeres = log_schechter(L, *Keres)

plt.plot(L, ySpheroid, 'r')
plt.plot(L, yDisk, 'b')
#plt.plot(L, yKeres, 'g')
# plt.xlim(8,11.7)
# plt.ylim(-5,0)
plt.show()
