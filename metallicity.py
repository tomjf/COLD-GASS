import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

plt.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]

def genzel(x):
    dx = 0.26
    a,b,c = [],[],[]
    for i in range(0,len(x)):
        a.append((12 - ((1.3)*x[i])))
        b.append((10 - ((1.3-dx)*x[i])))
        c.append((14 - ((1.3+dx)*x[i])))
    return a,b,c

def schruba(x):
    sx = 0.4
    logAlphaCO, u, d = [], [], []
    for i in range(0,len(x)):
        aco =  np.log10(8) + (-2*(x[i] - 8.7))
        acoU =  np.log10(8) + (-(2)*(x[i] - 8.7))+0.12
        acoD =  np.log10(8) + (-(2)*(x[i] - 8.7))-0.12
        logAlphaCO.append(aco)
        u.append(acoU)
        d.append(acoD)
    return logAlphaCO, u, d

def PlotAlphaCO(x, galactic, genzel, genzell, genzelu, schruba,u,d, data, galmin, galmax):
    xmajorLocator   = MultipleLocator(0.2)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].plot(x, genzel, 'g-', linewidth=2, label = 'Genzel+12')
    ax[0,0].plot(x, schruba, 'm-', linewidth=2, label = 'Schruba+12')
    ax[0,0].plot(x, galactic, 'r', linewidth=2, label = 'Galactic')
    ax[0,0].fill_between(x, galmin, galmax, color='r', alpha = 0.3)
    ax[0,0].fill_between(x, genzell, genzelu, color='g', alpha = 0.3)
    ax[0,0].fill_between(x, u, d, color='m', alpha = 0.3)
    # ax[0,0].errorbar(data[:,0], np.log10(data[:,7]), c='navy', alpha = 0.4,markersize = 4, fmt='o   ', mec='navy', label = 'COLD GASS, Accurso+16')
    ax[0,0].errorbar(data[:,0], np.log10(data[:,8]), c='navy',alpha=0.7, markersize = 4, fmt='o   ', mec='navy', label = 'COLD GASS, Accurso+16')
    ax[0,0].set_xlabel(r'$\mathrm{Z_{O}=12+log(O/H)}$', fontsize=30)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \alpha_{CO}\, [M_{\odot}/(K\, km\, s^{-1}\, pc^{2})]}$', fontsize=30)
    ax[0,0].set_ylim(0, 2.5)
    ax[0,0].set_xlim(8, 9)
    ax[0,0].tick_params(axis='x',which='minor',bottom='on')
    plt.legend(fontsize = 18)
    plt.tight_layout()
    plt.savefig('img/metallicity/alphaCO.pdf', format='pdf', dpi=250, transparent = False)

x = np.linspace(7,9.1,500)
data = np.loadtxt('giometallicity.txt')
# print len(data[:,0]), len(data[:,6])
data = data[data[:,7]>0]
print len(data)
galactic = np.log10(np.full(np.shape(x), 4.35))
galmax = np.log10(np.full(np.shape(x), 5.25))
galmin = np.log10(np.full(np.shape(x), 3.45))
genzel, genzell, genzelu = genzel(x)
schruba, u, d = schruba(x)
PlotAlphaCO(x, galactic, genzel, genzell, genzelu, schruba,u,d, data, galmin, galmax)
