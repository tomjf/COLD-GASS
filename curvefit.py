import numpy as np
import math
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear(xdata, m, c):
    return m*xdata + c

def cubic(xdata, a, b, c, d):
    return a*xdata*xdata*xdata + b*xdata*xdata + c*xdata + d

def schechter(L, rhostar, Lstar, alpha):
    a = rhostar
    b = L*(1/Lstar)
    c = np.power(b,(alpha+1))
    d = np.exp(-b)
    e = np.log(10)
    return a*c*d*e

xdata = np.linspace(0,1.5,num=4)
ydata = linear(xdata,1.0,0.0)
ycubic = cubic(xdata,1.0, 2.0, -3.0, -1.0)
random = np.random.rand(len(xdata),1)
for i in range (0,len(ydata)):
    ydata[i] = ydata[i]-0.5+random[i]
    ycubic[i] = ycubic[i]-0.5+random[i]

popt, pcov = curve_fit(linear, xdata, ydata)
ynew = linear(xdata, *popt)

xdata2 = np.linspace(0,1.5,num=1000)
popt2, pcov2 = curve_fit(cubic, xdata, ycubic)
ycubicnew = cubic(xdata2, *popt2)

schechX = np.logspace(8,12,40)
schechY = schechter(schechX, 0.00072, math.pow(10,7), -1.30)
plt.plot(schechX,schechY,'o')
# plt.plot(xdata2,ycubicnew,'k')
plt.show()
