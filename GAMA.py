import numpy as np
import atpy
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from astropy.table import Table
import pandas as pd

t = Table.read('data/GAMA/EmLinesPhys.fits')
sfr = np.zeros((len(t),2))
sfr[:,0] = np.array(t['CATAID'])
sfr[:,1] = np.log10(np.array(t['SFR']))

df = pd.read_csv('data/GAMA/StellarMasses.csv')
mstar = df[['CATAID', 'Z', 'fluxscale', 'logmstar']].values
mstaradd = np.zeros((len(mstar),1))
mstaradd[:,0] = mstar[:,3] + np.log10(mstar[:,2])
mstar = np.hstack((mstar, mstaradd))

print len(mstar), len(sfr)
data = np.zeros((len(sfr),6))
for idx, element in enumerate(sfr):
    print idx
    for idx1, element1 in enumerate(mstar):
        if element[0] == element1[0]:
            data[idx,0] = element1[0]#cataid
            data[idx,1] = element1[1]#z
            data[idx,2] = element1[2]#fluxscale
            data[idx,3] = element1[3]#logmstar ap
            data[idx,4] = element1[4]#logmstar tot
            data[idx,5] = element[1] #sfr
np.savetxt('gama.txt', data)

def Plotmasssfr(Peng, SDSS, GAMA, Best):
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].errorbar(Peng[2], Peng[1], fmt = 'o', markersize = 8, color = 'k', label = 'SDSS (Peng)')
    ax[0,0].errorbar(SDSS[2], SDSS[1], fmt = 'o', markersize = 8, color = 'm', label = 'SDSS (SDSS)')
    ax[0,0].errorbar(GAMA[2], GAMA[1], fmt = 'o', markersize = 8, color = 'b', label = 'SDSS (GAMA)')
    ax[0,0].errorbar(Best[2], Best[1], fmt = 'o', markersize = 8, color = 'lawngreen', label = 'SDSS (Best)')
    ax[0,0].plot(np.log10(x_keres), y_keres, 'k--', label = 'Keres+03')
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.5)
    plt.legend(fontsize = 13)
    plt.savefig('img/scal/'+ 'StarMass' + '.pdf', format='pdf', dpi=250, transparent = False)



# print t

# SFR = atpy.Table('data/GAMA/EmLinesPhys.fits')
# Mass = atpy.Table('data/GAMA/StellarMasses.fits')
#
# mstar = np.zeros((len(Mass), 5))
# for idx, row in enumerate(Mass):
#     mstar[idx,0] = row[0] #CATAID
#     mstar[idx,1] = row[1] #z
#     mstar[idx,2] = row[5] #fluxscale
#     mstar[idx,3] = row[10] #Mstar ap
#     mstar[idx,4] = 0 #Mstar total
#
# print mstar[:,3]
