import numpy as np
import atpy
import math
from scipy import integrate
import matplotlib.pyplot as plt
import asciidata

def lumdistance(z):
    omega_m = 0.31                          # from Planck
    omega_l = 0.69                          # from Planck
    c = 3*math.pow(10,5)                    # in km/s
    Ho = 70                                 # in km/(s Mpc)
    f = lambda x : (((omega_m*((1+z)**3))+omega_l)**-0.5)
    integral = integrate.quad(f, 0.0, z)    # numerically integrate to calculate luminosity distance
    Dm = (c/Ho)*integral[0]
    Dl = (1+z)*Dm                           # calculate luminosity distance
    DH = (c*z)/Ho                           # calculate distance from Hubble law for comparison
    return Dl, DH

def sortIntoBins(l):
    low = min(l)
    high = max(l)     # min max in logspace
    bins = np.linspace(low, high,num=20) # log-spaced bins
    N = []
    mid = []
    for i in range (1,len(bins)):
        inbin = [x for x in l if x > bins[i-1] and x < bins[i]]
        n = len(inbin)
        midpt = (bins[i]+bins[i-1])/2
        N.append(n)
        mid.append(midpt)
    return np.log10(N), mid

t = atpy.Table('COLDGASS_DR3.fits')         # read web data from table
abc = asciidata.open('COLDGASS_LOW_29Sep15.ascii')

lumsnew = list(abc[12])
for q in range(0,len(lumsnew)):
    a = lumsnew[q]
    a = math.pow(10,a)
    a = a*6
    lumsnew[q] = math.log10(a)

lums = []
for rows in t:                              # for each galaxy in the dataset
    SCO_cor = rows[15]                      # find the integrated CO flux
    C = 3.25*math.pow(10,7)                 # numerical const from eqn 4 paper 1
    freq = 111                              # observing frequency
    Dl, DH = lumdistance(rows[4])           # luminosity distance
    SDSS_z = math.pow((1+rows[4]),-3)       # redshift component
    L_CO = C*SCO_cor*((Dl*Dl)/(freq*freq))*SDSS_z   # calculate CO luminosity
    lums.append(L_CO)

lums = [i for i in lums if i > 0.0]         # remove 0 detected CO flux galaxies
lums = np.log10(lums)
combined = np.append(lums, lumsnew)

N1, mid1 = sortIntoBins(lums)
N2, mid2 = sortIntoBins(lumsnew)
N3, mid3 = sortIntoBins(combined)

plt.plot(mid1,N1,'bo')
plt.plot(mid2,N2,'ro')
plt.plot(mid3,N3,'go')
plt.xlabel('log_10(L_CO)')
plt.ylabel('log_10(N)')
plt.show()
