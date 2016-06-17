import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
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

def plotgioscal(data):
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    s = ax[0,0].scatter(data[:,1], data[:,0], c=data[:,2])
    cbar = plt.colorbar(s)
    cbar.ax.set_ylabel(r'$\mathrm{log\, M_{*}\,[M_{sun}]}$',  fontsize = 18)
    # ax[0,0].plot(x,y, linewidth = 1)
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, SFR\, [M_{\odot} \, yr^{-1}]}$', fontsize=18)
    plt.savefig('img/scal/scalrelns.eps', format='eps', dpi=250, transparent = False)

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
plotgioscal(data)
