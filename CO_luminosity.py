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
    bins = np.linspace(low, high,num=15) # log-spaced bins
    N = []
    mid = []
    for i in range (1,len(bins)):
        inbin = [x for x in l if x > bins[i-1] and x < bins[i]]
        n = len(inbin)
        midpt = (bins[i]+bins[i-1])/2
        N.append(n)
        mid.append(midpt)
    return N, mid

def lCalc(SCO, z, correction):
    lums = []
    for i in range(0,len(SCO)):                              # for each galaxy in the dataset
        if correction == True:
            SCO_cor = SCO[i]*6.0                      # find the integrated CO flux
        else:
            SCO_cor = SCO[i]
        C = 3.25*math.pow(10,7)                 # numerical const from eqn 4 paper 1
        freq = 111                              # observing frequency
        Dl, DH = lumdistance(z[i])           # luminosity distance
        SDSS_z = math.pow((1+z[i]),-3)       # redshift component
        L_CO = C*SCO_cor*((Dl*Dl)/(freq*freq))*SDSS_z   # calculate CO luminosity
        lums.append(L_CO)
    return lums

t = atpy.Table('COLDGASS_DR3.fits')         # read web data from table
abc = asciidata.open('COLDGASS_LOW_29Sep15.ascii')

lumsnew = list(abc[12])
print lumsnew[0], lumsnew[1]


SCO1, z1= [], []
for rows in t:                              # for each galaxy in the dataset
    SCO1.append(rows[15])                      # find the integrated CO flux
    z1.append(rows[4])
SCO2, z2 = list(abc[11]), list(abc[3])

lums1, lums2 = lCalc(SCO1,z1,False), lCalc(SCO2,z2,True)

lums1 = [i for i in lums1 if i > 0.0]         # remove 0 detected CO flux galaxies
lums2 = [i for i in lums2 if i > 0.0]         # remove 0 detected CO flux galaxies
lums1, lums2 = np.log10(lums1), np.log10(lums2)
combined = np.append(lums1, lums2)

N1, mid1 = sortIntoBins(lums1)
N2, mid2 = sortIntoBins(lums2)
N3, mid3 = sortIntoBins(combined)
N4, mid4 = sortIntoBins(lumsnew)

L = lCalc([0.49, 0.38], [0.01711, 0.01896], True)
print np.log10(L)


plt.plot(mid1,N1,'bo-')
plt.plot(mid2,N2,'ro-')
plt.plot(mid3,N3,'go-')
plt.plot(mid4,N4,'ko-')
plt.xlabel('log_10(L_CO)')
plt.ylabel('log_10(N)')
plt.show()
plt.savefig('a.png')
