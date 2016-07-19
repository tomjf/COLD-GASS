import numpy as np
import atpy
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import scal_relns

def sortTable(table, indices):
    newdata = np.zeros((len(table), len(indices)+1))
    for index, rows in enumerate(table):
        newdata[index, 0] = int(rows[indices['PLATEID']])
        newdata[index, 1] = int(rows[indices['MJD']])
        newdata[index, 2] = int(rows[indices['FIBERID']])
        newdata[index, 3] = rows[indices['M*']]
        newdata[index, 4] = rows[indices['SFR']]
        newdata[index, 5] = rows[indices['MH2']]
        newdata[index, 6] = rows[indices['limMH2']]
        newdata[index, 7] = newdata[index, 5] + newdata[index, 6]
    return newdata

def sortTable2(info, sfrs, mstar, indices):
    newdata = np.zeros((len(info), len(indices)))
    for index, rows in enumerate(info):
        newdata[index, 0] = int(rows[indices['PLATEID']])
        newdata[index, 1] = int(rows[indices['MJD']])
        newdata[index, 2] = int(rows[indices['FIBERID']])
        newdata[index, 3] = mstar[index][indices['M*']]
        newdata[index, 4] = sfrs[index][indices['SFR']]
    return newdata

def SFRMSTAR(CGData2):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(CGData2[:,3], CGData2[:,4], s = 1, color = 'k', label = 'COLD GASS')
    ax[0,0].scatter(CGData2[:,8], CGData2[:,9], s = 10, color = 'r', label = 'COLD GASS SDSS')
    ax[0,0].set_xlim(8, 11.5)
    ax[0,0].set_ylim(-2.5, 1)
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{*}\,[M_{sun}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, SFR\,[M_{sun}\,yr^{-1}]}$', fontsize=18)
    plt.savefig('img/scal/sfrmstar2.pdf', format='pdf', dpi=250, transparent = False)

def SFRcomparison(CGData2):
    x = np.linspace(-3,2,200)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(CGData2[:,4], CGData2[:,9]-CGData2[:,4], s = 10, color = 'k', label = 'COLD GASS')
    # ax[0,0].plot(x,x)
    ax[0,0].set_ylabel(r'$\mathrm{SDSS}$', fontsize=18)
    ax[0,0].set_xlabel(r'$\mathrm{COLD\,GASS}$', fontsize=18)
    plt.savefig('img/scal/sfrcomparison.pdf', format='pdf', dpi=250, transparent = False)

def MH2SFR(CGData2, datagio):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(CGData2[:,4], CGData2[:,5], s = 10, color = 'r', label = 'COLD GASS D')
    ax[0,0].scatter(CGData2[:,4], CGData2[:,6], s = 10, color = 'm', label = 'COLD GASS ND')
    ax[0,0].scatter(datagio[:,1], datagio[:,0], s = 10, color = 'b', label = 'GIO')
    ax[0,0].set_xlabel(r'$\mathrm{SFR}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{MH2}$', fontsize=18)
    ax[0,0].set_xlim(-3, 2)
    ax[0,0].set_ylim(7.5, 11)
    plt.savefig('img/scal/giocomparison.pdf', format='pdf', dpi=250, transparent = False)

Full = atpy.Table('data/COLDGASS_full.fits')
galinfo = atpy.Table('data/sdss/gal_info_dr7_v5_2.fit')
sfr = atpy.Table('data/sdss/gal_totsfr_dr7_v5_2.fits')
mstar = atpy.Table('data/sdss/totlgm_dr7_v5_2.fit')
CG_index = {'PLATEID':4, 'MJD':5, 'FIBERID':6, 'M*':20, 'SFR':23, 'MH2':51, 'limMH2':52}
SDSS_index = {'PLATEID':0, 'MJD':1, 'FIBERID':2, 'M*':6, 'SFR':0}
CGData = sortTable(Full, CG_index)
sdssData = sortTable2(galinfo, sfr, mstar, SDSS_index)
CGaddon = np.zeros((len(CGData),2))
for index, elements in enumerate(CGData):
    e = elements[:3]
    for index1, element1 in enumerate(sdssData):
        d = element1[:3]
        if d[0] == e[0] and d[2] == e[2] and d[2] == e[2]:
            print index, (float(index)/499.0)*100.0, '%'
            CGaddon[index,0] = sdssData[index1,3]
            CGaddon[index,1] = sdssData[index1,4]
CGData2 = np.hstack((CGData, CGaddon))
datagio, fit = scal_relns.fitdata()

np.savetxt('data/MH2.txt', CGData2, delimiter = ",")

SFRMSTAR(CGData2)
SFRcomparison(CGData2)
MH2SFR(CGData2, datagio)
