import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def genzel(x):
    logAlphaCO = []
    for i in range(0,len(x)):
        logAlphaCO.append((12 - (1.3*x[i])))
    return logAlphaCO

def schruba(x):
    logAlphaCO = []
    for i in range(0,len(x)):
        aco =  np.log10(8) + (-2*(x[i] - 8.7))
        logAlphaCO.append(aco)
    return logAlphaCO

def PlotAlphaCO(x, galactic, genzel, schruba):
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].plot(x, genzel, 'k-', label = 'Genzel+12')
    ax[0,0].plot(x, galactic, 'k:', label = 'Milky Way')
    ax[0,0].plot(x, schruba, 'k-.', label = 'Schruba+12')
    ax[0,0].set_xlabel(r'$\mathrm{Z_{O}=12+log(O/H)}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \alpha_{CO}\, [M_{\odot}/(K\, km\, s^{-1}\, pc^{2})]}$', fontsize=18)
    # ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7, 9.1)
    ax[0,0].tick_params(axis='x',which='minor',bottom='on')
    plt.legend(fontsize = 13)
    plt.savefig('img/metallicity/alphaCO.pdf', format='pdf', dpi=250, transparent = False)

x = np.linspace(7,9.1,500)
galactic = np.log10(np.full(np.shape(x), 4.35))
genzel = genzel(x)
schruba = schruba(x)
PlotAlphaCO(x, galactic, genzel, schruba)
