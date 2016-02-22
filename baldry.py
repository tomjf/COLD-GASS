import numpy as np
import math
import matplotlib.pyplot as plt


def powerten(lognum):
    return 10**lognum

def doubleschec(M, Mstar, dM, phi1, phi2, alph1, alph2):
    phis = []
    for i in range (0, len(M)):
        frac = 10**(M[i]-Mstar)
        exp = math.exp(-frac)
        phis.append(5*exp*((phi1*(frac**alph1))+(phi2*(frac**alph2)))*(dM/Mstar))
    return phis
baldry = [17.9, 43.1, 31.6, 34.8, 27.3, 28.3, 23.5, 19.2, 18.0, 14.3, 10.2, 9.59, 7.42, 6.21, 5.71, 5.51, 5.48, 5.12, 3.55, 2.41, 1.27, 0.338, 0.042, 0.021, 0.042]
for i in range(0,len(baldry)):
    baldry[i] = baldry[i]/1000.0
Mstar = 10.66
phi1 = 3.96/1000
phi2 = 0.79/1000
alph1 = -0.35
alph2 = -1.47

MassBins = np.linspace(7.10,11.90,25)
answer = doubleschec(MassBins, Mstar, (MassBins[1]-MassBins[0]), phi1, phi2, alph1, alph2)
data = np.zeros((len(MassBins,),3))
data[:,0] = MassBins
data[:,1] = answer
data[:,2] = baldry
print data

plt.plot(MassBins, np.log10(answer))
plt.plot(MassBins, np.log10(baldry))
plt.show()
