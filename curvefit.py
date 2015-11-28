import numpy as np
import math
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def schechter(L, rhostar, Lstar, alpha):
    a = rhostar
    b = L*(1/Lstar)
    c = np.power(b,(alpha+1))
    d = np.exp(-b)
    e = np.log(10)
    # print 'a',a
    # print 'b',b
    # print 'c',c
    # print 'd',d
    # print 'e',e
    return a*c*d*e

schechX = np.logspace(8,12,100)
schechY = schechter(schechX, 0.00072, np.power(10.0,9.2), -1.30)
popt, pcov = curve_fit(schechter, schechX, schechY)
print popt
ynew = schechter(schechX, *popt)

plt.plot(np.log10(schechX),np.log(schechY),'o')
plt.plot(schechX,ynew,'r-')
plt.xlim(8,12)
plt.show()
