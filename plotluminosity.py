import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
from main import GetFull
import atpy
#
def extrapolate(data, up, down, n):
    x = np.linspace(down,up,n)
    newdata = np.zeros((len(x),2))
    for j in range(0,len(x)):
        for i in range(0,len(data)-1):
            dx = data[i+1,0] - data[i,0]
            dy = data[i+1,1] - data[i,1]
            y = data[i,1]
            if x[j] < data[i+1,0] and x[j]>=data[i,0]:
                DX = x[j] - data[i,0]
                if DX == 0:
                    newdata[j,1] = y
                    newdata[j,0] = x[j]
                else:
                    ans = y + ((DX/dx)*dy)
                    newdata[j,1] = ans
                    newdata[j,0] = x[j]
            elif x[j]<data[0,0]:
                dx = data[1,0] - data[0,0]
                dy = data[1,1] - data[0,1]
                DX = x[j] - data[0,0]
                y = data[0,1]
                ans = y + ((DX/dx)*dy)
                newdata[j,1] = ans
                newdata[j,0] = x[j]
            elif x[j]>data[-1,0]:
                print x[j], data[-1,0]
                dx = data[-1,0] - data[-2,0]
                dy = data[-1,1] - data[-2,1]
                DX = x[j] - data[-1,0]
                y = data[-1,1]
                ans = y + ((DX/dx)*dy)
                newdata[j,1] = ans
                newdata[j,0] = x[j]
    return newdata


def PlotLum(data, uperr, lowerr, keres, lowerr2, uperr2, LCOSch):
    xmajorLocator   = MultipleLocator(1)
    xminorLocator   = MultipleLocator(0.2)
    ymajorLocator   = MultipleLocator(1)
    yminorLocator   = MultipleLocator(0.2)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].plot(data[:,0], data[:,1], label = 'Valini+16', color = 'crimson', linewidth=3)
    # ax[0,0].plot(uperr[:,0], uperr[:,1], label = 'upper limit', color = 'crimson')
    # ax[0,0].plot(lowerr[:,0], lowerr[:,1], label = 'lower limit', color = 'crimson')
    ax[0,0].fill_between(lowerr2[:,0], lowerr2[:,1], uperr2[:,1], label = 'Valini+16 error', color = 'r', alpha = 0.2)
    # ax[0,0].scatter(lowerr2[:,0], lowerr2[:,1], label = 'uperr', color = 'g')
    # ax[0,0].scatter(uperr2[:,0], uperr2[:,1], label = 'uperr', color = 'g')
    ax[0,0].errorbar(keres[:,0], keres[:,1], yerr=[keres[:,3], keres[:,2]], fmt='bo', markersize = 12, linewidth=2, mew=2, capthick=3, mfc='b', mec='navy' , label='Keres+03')
    ax[0,0].errorbar(LCOSch[2], LCOSch[1], fmt='h', markersize = 12, linewidth=2, mew=2, capthick=3, mfc='limegreen', mec='g' , label='COLD GASS')
    ax[0,0].set_xlabel(r'$\mathrm{log\, L\'_{CO}\, [K \, km \,s^{-1}\, pc^{2}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_xlim(5.5, 12)
    ax[0,0].set_ylim(-7, -1)
    plt.legend(fontsize = 13)
    plt.savefig('img/schechter/luminosity.pdf', format='pdf', dpi=250, transparent = False)

output = {  'ID':0, 'S_CO':1, 'z':2, 'flag':3, 'M*':4, 'Zo':5, 'SFR':6, 'sSFR':7,
            'NUV-r':8,'D_L':9, 'V/Vm':10, 'Vm':11, 'L_CO':12, 'AlphaCO':13,
            'MH2':14, 'dalpha':15}

keres = [ 980.859,6850.55, 6921.05, 6781.25 ]
keres= np.vstack((keres,[1058.83,6699.49, 6760.39, 6660.82 ]))
keres= np.vstack((keres,[1136.41,6868.16, 6931.02, 6816.09]))
keres= np.vstack((keres,[1214.3,6791.76, 6846.91, 6766.52]))
keres= np.vstack((keres,[1292.03,6816.88, 6866.29, 6806.91]))
keres= np.vstack((keres,[1371.21,6765.35, 6807.11, 6743.91]))
keres= np.vstack((keres,[1447.73,6719.57, 6759.41, 6703.87]))
keres= np.vstack((keres,[1527.11,6532.11, 6575.78, 6535.59]))
keres= np.vstack((keres,[1603.75,6369.53, 6441.95, 6332.77 ]))
keres= np.vstack((keres,[1683.2,6076.76, 6154.92, 6041.91]))

data = pd.read_csv('data/lum/test33a.csv').values
uperr = pd.read_csv('data/lum/uperr.csv').values
lowerr = pd.read_csv('data/lum/lowerr.csv').values
lowerr = lowerr[lowerr[:,0].argsort()]
lowerr2 = extrapolate(lowerr, 5.5, 11, 200)
print lowerr2
uperr2 = extrapolate(uperr, 5.5, 11, 200)


Full = atpy.Table('data/COLDGASS_full.fits')
#0'z', 1'flag', 2'MH2', 3'limMH2', 4'MH2both', 5'D_L', 6'M*', 7'V/Vm', 8'Vm', 9'Weight', 10'LCO'
LND, HND, FullData, weights, LCOSch = GetFull(Full, output)
# SFRBL = df[['GASS', 'SFR_best', 'SFRerr_best', 'SFRcase_best']].values
# data = np.loadtxt('data/lum/test33a.csv')
# uperr = np.loadtxt('data/lum/valiniErr.csv')
# lowerr = np.loadtxt('data/lum/valiniErr2.csv')

xcal, ycal = np.zeros((2,2)), np.zeros((2,2))
xcal[1,0] = 871.051
xcal[1,1] = 1066.37
xcal[0,0] = 6.0
xcal[0,1] = 7.0

ycal[1,0] = 6021.72
ycal[1,1] = 6217.03
ycal[0,0] = -6.0
ycal[0,1] = -5.0

for idx, element in enumerate(keres):
    keres[idx,0] = xcal[0,0] + (element[0]-xcal[1,0])/(xcal[1,1]-xcal[1,0])
    keres[idx,1] = ycal[0,0] + (element[1]-ycal[1,0])/(ycal[1,1]-ycal[1,0])+0.1
    keres[idx,2] = ycal[0,0] + (element[2]-ycal[1,0])/(ycal[1,1]-ycal[1,0]) - keres[idx,1]
    keres[idx,3] = keres[idx,1] -(ycal[0,0] + (element[3]-ycal[1,0])/(ycal[1,1]-ycal[1,0]))
# uperr = uperr[uperr[:,0].argsort()]
# lowerr = lowerr[lowerr[:,0].argsort()]
# # uperr = uperr[uperr[:,0]>5.5]
# # uperr = uperr[uperr[:,0]<11.2]
# lowerr = lowerr[lowerr[:,0]>5.5]
# lowerr = lowerr[lowerr[:,0]<11.2]
# np.savetxt('data/lum/upasaerr.csv', uperr)
# np.savetxt('data/lum/lowerrjk.csv', lowerr)
# print data
PlotLum(data, uperr, lowerr, keres, lowerr2, uperr2, LCOSch)
