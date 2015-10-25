import numpy as np
import atpy
import math
from scipy import integrate
import matplotlib.pyplot as plt

def lumdistance(z):
    omega_m = 0.31
    omega_l = 0.69
    c = 3*math.pow(10,5)    # in km/s
    Ho = 70                 # in km/(s Mpc)
    f = lambda x : (((omega_m*((1+z)**3))+omega_l)**-0.5)
    integral = integrate.quad(f, 0.0, z)
    Dm = (c/Ho)*integral[0]
    Dl = (1+z)*Dm
    return Dl

t = atpy.Table('COLDGASS_DR3.fits')

lums = []
for rows in t:
    SCO_cor = rows[15]
    C = 3.25*math.pow(10,7)
    freq = 111
    Dl = lumdistance(rows[4])
    SDSS_z = math.pow((1+rows[4]),-3)
    L_CO = C*SCO_cor*freq*Dl*Dl*SDSS_z
    lums.append(L_CO)

lums = [i for i in lums if i > 0.0]
minimum = min(lums)
maximum = max(lums)
bins = np.logspace(14,17,num=15)
print bins

N = []
mid = []
for i in range (1,len(bins)):
    inbin = [x for x in lums if x > bins[i-1] and x < bins[i]]
    n = len(inbin)
    midpt = (bins[i]+bins[i-1])/2
    N.append(n)
    mid.append(midpt)

mid = np.log10(mid)
N = np.log10(N)


plt.plot(mid,N,'o')
plt.show()
