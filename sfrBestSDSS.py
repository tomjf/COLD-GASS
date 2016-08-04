import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import atpy
import pandas as pd
from scipy.optimize import curve_fit
import random

def sortTable2(info, sfrs, indices, mstar):
    newdata = np.zeros((len(info), len(indices)))
    for index, rows in enumerate(info):
        newdata[index, 0] = int(rows[indices['PLATEID']])
        newdata[index, 1] = int(rows[indices['MJD']])
        newdata[index, 2] = int(rows[indices['FIBERID']])
        newdata[index, 3] = sfrs[index][indices['SFR']]
        newdata[index, 4] = mstar[index][indices['M*']]
    return newdata

def sfrbest13k():
    SDSS_index = {'PLATEID':0, 'MJD':1, 'FIBERID':2, 'SFR':0, 'M*':5}
    df = pd.read_csv('data/allGASS_SFRbest_simple_t1.csv')
    SFRBL = df[['GASS', 'SFR_best', 'SFRerr_best']].values
    df = pd.read_csv('data/PS_100701.csv')
    coords_H = df[['GASS', 'PLATEID', 'MJD', 'FIBERID']].values
    df = pd.read_csv('data/LOWMASS_MASTER.csv')
    coords_L = df[['GASSID', 'PLATEID', 'MJD', 'FIBERID']].values
    galinfo = atpy.Table('data/sdss/gal_info_dr7_v5_2.fit')
    sfr = atpy.Table('data/sdss/gal_totsfr_dr7_v5_2.fits')
    mstar = atpy.Table('data/sdss/totlgm_dr7_v5_2.fit')
    S = sortTable2(galinfo, sfr, SDSS_index, mstar)
    GASS = np.vstack((coords_H, coords_L))
    IDs = np.zeros((len(SFRBL),5))
    for index, element in enumerate(SFRBL):
        print index
        for index1, element1 in enumerate(GASS):
            if element[0] == element1[0]:
                IDs[index, 0] = element1[1]
                IDs[index, 1] = element1[2]
                IDs[index, 2] = element1[3]
                for idx, el in enumerate(S):
                    if el[0]==element1[1] and el[1]==element1[2] and el[2]==element1[3]:
                        IDs[index,3] = el[3]
                        IDs[index,4] = el[4]
    SFRBL = np.hstack((SFRBL, IDs))
    np.savetxt('sfrC.txt', SFRBL)
    return SFRBL

def plotsfrs(sfr_sdss, sfr_best, solns, data, y):
    x = np.linspace(-3,2,500)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(data[:,6], np.log10(data[:,1]), color = 'm', label = 'Sim', s=10)
    ax[0,0].scatter(data[:,6], y, color = 'g', label = 'Sim2', s=10)
    # for i in range(0,len(solns)-1):
    #     ax[0,0].plot(solns[-1], solns[i], label = str(i+1))
    # ax[0,0].plot(x,x, color='k')
    ax[0,0].set_ylabel(r'$\mathrm{log\, SFR_{Best}\,[M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_xlabel(r'$\mathrm{log\, SFR_{SDSS}\,[M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-3, 2)
    ax[0,0].set_xlim(-3, 2)
    plt.legend(fontsize = 12)
    plt.savefig('img/sfrs.pdf', dpi=250, transparent = False)

def first(x, a, b):
    return a*x + b

def second(x, a, b, c):
    return a*x*x + b*x + c

def third(x, a, b, c, d):
    return a*x*x*x + b*x*x + c*x + d

def fourth(x, a, b, c, d, e):
    return a*x*x*x*x + b*x*x*x + c*x*x + d*x + e

def polyfit(sfr_sdss, sfr_best, order):
    fit, y = 0,0
    x = np.linspace(-3,2,500)
    dy = []
    if order == 1:
        p0 = curve_fit(first, sfr_sdss[:100], sfr_best[:100])[0]
        fit = curve_fit(first, sfr_sdss, sfr_best, p0 = p0)
        y = first(x, *fit[0])
    elif order == 2:
        p0 = curve_fit(second, sfr_sdss[:100], sfr_best[:100])[0]
        fit = curve_fit(second, sfr_sdss, sfr_best)
        y = second(x, *fit[0])
    elif order == 3:
        p0 = curve_fit(third, sfr_sdss[:100], sfr_best[:100])[0]
        fit = curve_fit(third, sfr_sdss, sfr_best)
        y = third(x, *fit[0])
    elif order == 4:
        p0 = curve_fit(fourth, sfr_sdss[:100], sfr_best[:100])[0]
        fit = curve_fit(fourth, sfr_sdss, sfr_best)
        y = fourth(x, *fit[0])
    return fit, x, y

def besstfit(sfr_sdss, sfr_best):
    solns = []
    for i in range(0,3):
        fit, x ,y  = polyfit(sfr_sdss, sfr_best, i+1)
        solns.append(y)
    solns.append(x)
    return solns

def convertsdss(newdata):
    num = 110
    data = np.loadtxt('sfrC.txt')
    data = data[data[:,6]!=0]
    sfr_sdss, sfr_best = data[:num,6], np.log10(data[:num,1])
    fit, x, y = polyfit(sfr_sdss, sfr_best, 2)
    y = second(newdata, *fit[0])
    # for i in range(0,len(y)):
    #     y[i] += random.gauss(0,0.4)
    return y

# SFRBL = sfrbest13k()
# solns = besstfit(sfr_sdss, sfr_best)
# plotsfrs(sfr_sdss, sfr_best, solns, data, y)
