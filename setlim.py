import numpy as np
import atpy
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
import schechter

# 0: M*group | 1: dM* | 2: phi | 3: N | 4: M* | 5:SFR_SDSS | 6: SFR_Best
#|7: MH2_SDSS_G |8: MH2_Best_G | 9: Vm | 10: MH2_SDSS_A |11: MH2_Best_A

def Schechter(data, LCOaxis, Vmaxis, bins):
    l = data[:,LCOaxis]
    # l = np.log10(l)
    rho, N, xbins, sigma, rhoH2 = [], [], [], [], []
    for i in range (1,len(bins)):
        p, Num, o, pH2 = 0, 0, 0, 0
        for j in range(0,len(l)):
            if l[j] >= bins[i-1] and l[j] < bins[i]:
                p += 1/data[j,Vmaxis]
                o += 1/(data[j,Vmaxis]**2)
                pH2 += data[j,LCOaxis]/data[j,Vmaxis]
                Num+=1
        N.append(Num)
        xbins.append((bins[i]+bins[i-1])/2)
        rho.append(p/(bins[1]-bins[0]))
        sigma.append(math.sqrt(o))
        rhoH2.append(pH2/(bins[1]-bins[0]))
    # return the Number of gals, log10(density), centre pt of each bin
    return [N, np.log10(rho), xbins, np.log10(sigma), np.log10(rhoH2)]

def testMassLimits(data):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    limits = np.linspace(7,10,10)
    massbins = np.linspace(7.5,11.0,25)
    for idx, element in enumerate(limits):
        newdata = data[data[:,4] >= element]
        print len(newdata)
        print min(newdata[:,8]), max(newdata[:,8])
        sch = Schechter(newdata, 8, 9, massbins)
        # print sch[1]
        ax[0,0].plot(sch[2], sch[1], label = str(round(element,1)))
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{\odot}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.5)
    plt.legend(fontsize=13)
    plt.savefig('img/scal/MassLims.pdf', format='pdf', dpi=250, transparent = False)
