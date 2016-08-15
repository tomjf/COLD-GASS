import numpy as np
import atpy
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd

# 0: M*group | 1: dM* | 2: phi | 3: N | 4: M* | 5:SFR_SDSS | 6: SFR_Best
#|7: MH2_SDSS_G |8: MH2_Best_G | 9: Vm | 10: MH2_SDSS_A |11: MH2_Best_A

def testMassLimits(data):
    x = np.linspace(-12,-8,100)
    logfH2 = 6.02 + (0.704*x)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    Mstar = data[:,6]
    MH2 = data[:,4]
    fh2 = MH2-Mstar
    SSFR = data[:,14]
    fH2data = 6.02 + (0.704*SSFR)
    res = fh2 - fH2data
    sig = np.std(res)
    print sig, '!!!@@@@@@@@@@@@@@'
    yu = logfH2+sig
    yl = logfH2-sig
    yu2 = logfH2+(3*sig)
    yl2 = logfH2-(3*sig)
    ax[0,0].fill_between(x, yl, yu, alpha = 0.3, color = 'k', label = r'$1 \sigma$')
    ax[0,0].fill_between(x, yl2, yu2, alpha = 0.1, color = 'k', label = r'$3 \sigma$')
    ax[0,0].scatter(SSFR, fh2, label = 'COLD GASS detections', color = 'b', s=10)
    ax[0,0].plot(x,logfH2, label ='Saintonge+16')
    ax[0,0].set_xlabel(r'$\mathrm{log\, sSFR\,[yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, f_{H_{2}}}$', fontsize=18)
    ax[0,0].set_xlim(-12, -8)
    ax[0,0].set_ylim(-2.5, 0)
    plt.legend(fontsize=13, loc=4)
    plt.savefig('img/scal/gasfrac.pdf', format='pdf', dpi=250, transparent = False)
