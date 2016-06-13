import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def first(x, a, b):
    return a*x + b

def second(x, a, b, c):
    return a*x*x + b*x + c

def third(x, a, b, c, d):
    return a*x*x*x + b*x*x + c*x + d

def fourth(x, a, b, c, d, e):
    return a*x*x*x*x + b*x*x*x + c*x*x + d*x + e

def second2var(x, a, b, c, d, e, f):
    m, sfr = x
    return a*m**2 + b*m*sfr + c*sfr**2 + d*m + e*sfr + f

def polyfit(data, order):
    if order == 1:
        fit = curve_fit(first, data[:,1], data[:,0])
    elif order == 2:
        fit = curve_fit(second, data[:,1], data[:,0])
    elif order == 3:
        fit = curve_fit(third, data[:,1], data[:,0])
    elif order == 4:
        fit = curve_fit(fourth, data[:,1], data[:,0])
    return fit

def fitdata():
    df = pd.read_csv('data/cold_gass_data_gio.csv')
    data = df[['Log_Mh2', 'Log_SFR', 'Log_M', 'Log_LCO']].values
    fit = polyfit(data, 3)
    x = np.linspace(-1.5,2.0,300)
    y = third(x, *fit[0])
    return data, fit

def fitdata2():
    df = pd.read_csv('data/cold_gass_data_gio.csv')
    data = df[['Log_Mh2', 'Log_SFR', 'Log_M', 'Log_LCO']].values
    fit = curve_fit(second2var, (data[:,2], data[:,1]), data[:,0])
    return data, fit

data, fit = fitdata()

fit2var = curve_fit(second2var, (data[:,2], data[:,1]), data[:,0])
#print fit2var[0]
MH2 = second2var((data[:,2], data[:,1]), *fit2var[0])
#print MH2

res = np.zeros((len(data[:,0]),1))
for i in range(0, len(data[:,0])):
    res[i,0] = data[i,0] - third(data[i,1], *fit[0])

# plt.scatter(data[:,1], MH2)
# plt.scatter(data[:,1], data[:,0], color = 'r')
# plt.show()
#
# fig, ax = plt.subplots(nrows = 2, ncols = 2, squeeze=False)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# s = ax[0,0].scatter(data[:,1], data[:,0], c=data[:,2])
# ax[0,0].plot(x,y, linewidth = 1)
# ax[0,0].set_ylabel(r'$Log(MH2)$', fontsize = 12)
# ax[0,0].set_xlabel(r'$Log(SFR)$', fontsize = 12)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# t = ax[0,1].scatter(data[:,1], data[:,3], c=data[:,2])
# ax[0,1].set_ylabel(r'$Log(LCO)$', fontsize = 12)
# ax[0,1].set_xlabel(r'$Log(SFR)$', fontsize = 12)
# cb = plt.colorbar(t)
# cb.set_label('Log M')
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ax[1,0].scatter(data[:,1],res)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #ax[1,1].scatter/(data[:,1],res)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# fig.set_size_inches(10,12)
# plt.savefig('img/scalrelns.pdf', format='pdf', dpi=1000, transparent = False)
#plt.show()
