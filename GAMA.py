import numpy as np
import atpy
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from astropy.table import Table
import pandas as pd
from matplotlib.colors import LogNorm
import schechter

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

def residuals(data):
    return round(np.mean(data),2), round(np.std(data),2)

def nosillypts(x, y):
    data = np.vstack((x,y))
    data = data[abs(data[:,1])<100]
    print data
    x,y = list(data[:,0]), list(data[:,1])
    print type(x)
    return x,y

def boundary(data, MH2axis, Vmaxis, bins, sig, u,c,l):
    x = np.linspace(7.5,10.5,500)
    centre = Schechter(data, MH2axis, Vmaxis, bins)
    # x1,y1 = nosillypts(centre[2], centre[1])
    para = schechter.log_schechter_fit(centre[2][:c], centre[1][:c])
    centreline = schechter.log_schechter(x, *para)
    data[:,MH2axis] -= 1.0*sig
    lower = Schechter(data, MH2axis, Vmaxis, bins)
    # x1,y1 = nosillypts(lower[2], lower[1])
    para = schechter.log_schechter_fit(lower[2][:l],lower[1][:l])
    lowerline = schechter.log_schechter(x, *para)
    data[:,MH2axis] += 2.0*sig
    upper = Schechter(data, MH2axis, Vmaxis, bins)
    # x1,y1 = nosillypts(upper[2], upper[1])
    para = schechter.log_schechter_fit(upper[2][:u],upper[1][:u])
    upperline = schechter.log_schechter(x, *para)
    return [x, centreline, upperline, lowerline, centre, lower, upper]

def MH2varSFR(total, bins, x_keres, y_keres, res):
    a = 0.1
    # 0: M*group | 1: dM* | 2: phi | 3: N | 4: M* | 5:SFR_SDSS | 6: SFR_Best
    #|7: MH2_SDSS_G |8: MH2_Best_G | 9: Vm | 10: MH2_SDSS_A |11: MH2_Best_A
    bl = len(bins)
    PG = boundary(total, 7, 9, bins, res, bl-1, bl, bl-3)
    BG = boundary(total, 8, 9, bins, res, bl-1, bl-2, bl-3)
    PA = boundary(total, 10, 9, bins, res, bl-1, bl-2, bl-3)
    BA = boundary(total, 11, 9, bins, res, bl-1, bl-2, bl-3)
    print total[:,8]
    # print y2
    # data = np.loadtxt('gama.txt')
    # data = data[data[:,5]!= float('nan')]
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    # ax[0,0].errorbar(PengAmelie[2], PengAmelie[1], fmt = 's', markersize = 8, color = 'k', label = 'PengAmelie')
    # ax[0,0].errorbar(BestAmelie[2], BestAmelie[1], fmt = 's', markersize = 8, color = 'm', label = 'BestAmelie')
    # ax[0,0].scatter(PengGio[2], PengGio[1], label = 'PengGio', color='r')
    ax[0,0].plot(PG[0], PG[1], label = 'SFR_Peng + MH2-SFR', color='r', linewidth = 3)
    ax[0,0].plot(PG[0], PG[2], color='none')
    ax[0,0].plot(PG[0], PG[3], color='none')
    ax[0,0].fill_between(PG[0], PG[2], PG[3], color ='r', alpha = a)
    ax[0,0].plot(BG[0], BG[1], label = 'SFR_Best + MH2-SFR', color='b', linewidth = 3)
    ax[0,0].plot(BG[0], BG[2], color='none')
    ax[0,0].plot(BG[0], BG[3], color='none')
    ax[0,0].fill_between(BG[0], BG[2], BG[3], color ='b', alpha = a)
    ax[0,0].plot(PA[0], PA[1], label = 'SFR_Peng + fH2', color='g', linewidth = 3)
    ax[0,0].plot(PA[0], PA[2], color='none')
    ax[0,0].plot(PA[0], PA[3], color='none')
    # ax[0,0].fill_between(PA[0], PA[2], PA[3], color ='g', alpha = a)
    ax[0,0].plot(BA[0], BA[1], label = 'SFR_Best + fH2', color='k', linewidth = 3)
    ax[0,0].plot(BA[0], BA[2], color='none')
    ax[0,0].plot(BA[0], BA[3], color='none')
    # ax[0,0].fill_between(BA[0], BA[2], BA[3], color ='k', alpha = a)

    # ax[0,0].plot(BestGio[2], BestGio[1], label = 'BestGio')
    # ax[0,0].plot(FullSchech_Best_D[2], FullSchech_Best_D[1], label = 'Det Am')
    # ax[0,0].plot(FullSchech_Best_D_G[2], FullSchech_Best_D_G[1], label = 'Det Am')
    # ax[0,0].plot(FullSchech_Best_D_S_G[2], FullSchech_Best_D_S_G[1], label = 'Det S G')
    # ax[0,0].plot(FullSchech_Best_D_S_A[2], FullSchech_Best_D_S_A[1], label = 'Det S A')
    # ax[0,0].scatter(FullDetSchech[2], FullDetSchech[1], label = 'Fulldet')
    ax[0,0].plot(np.log10(x_keres), y_keres, 'k--', label = 'Keres+03', linewidth = 3)
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{\odot}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.5)
    plt.legend(fontsize = 13, loc=3)
    plt.savefig('img/scal/'+ 'StarMass' + '.pdf', format='pdf', dpi=250, transparent = False)

def PlotdMH2(FullDet, Fulldetmains):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    MH2 = np.log10(Fulldetmains[:,4])
    x = MH2
    mean, std = residuals(MH2 - FullDet[:,8])
    # ax[0,0].scatter(x, MH2 - FullDet[:,8], color='g', label = 'AS'+str(mean)+str(std))
    mean, std = residuals(MH2 - FullDet[:,9])
    ax[0,0].scatter(x, MH2 - FullDet[:,9], color = 'm', label = 'AB'+str(mean)+str(std))
    mean, std = residuals(MH2 - FullDet[:,10])
    ax[0,0].scatter(x, MH2 - FullDet[:,10], color = 'b', label = 'GB'+str(mean)+str(std))
    mean, std = residuals(MH2 - FullDet[:,11])
    # ax[0,0].scatter(x, MH2 - FullDet[:,11], color = 'r', label = 'GS'+str(mean)+str(std))
    ax[0,0].hlines(0,6,11.5)
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{*}\,[M_{\odot}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, M_{H2,CG}- log\,M_{H2,model}\,[M_{\odot}]}$', fontsize=18)
    ax[0,0].set_xlim(7.5, 10.5)
    ax[0,0].set_ylim(-2, 3)
    plt.legend(fontsize = 13)
    plt.savefig('img/scal/test.pdf', format='pdf', dpi=250, transparent = False)


# t = Table.read('data/GAMA/EmLinesPhys.fits')
# sfr = np.zeros((len(t),2))
# sfr[:,0] = np.array(t['CATAID'])
# sfr[:,1] = np.array(t['SFR'])

#
# df = pd.read_csv('data/GAMA/StellarMasses.csv')
# mstar = df[['CATAID', 'Z', 'fluxscale', 'logmstar']].values
# mstaradd = np.zeros((len(mstar),1))
# mstaradd[:,0] = mstar[:,3] + np.log10(mstar[:,2])
# mstar = np.hstack((mstar, mstaradd))
#
# print len(mstar), len(sfr)
# data = np.zeros((len(sfr),6))
# for idx, element in enumerate(sfr):
#     print idx
#     for idx1, element1 in enumerate(mstar):
#         if element[0] == element1[0]:
#             data[idx,0] = element1[0]#cataid
#             data[idx,1] = element1[1]#z
#             data[idx,2] = element1[2]#fluxscale
#             data[idx,3] = element1[3]#logmstar ap
#             data[idx,4] = element1[4]#logmstar tot
#             data[idx,5] = element[1] #sfr
# np.savetxt('gama.txt', data)

# Plotmasssfr()

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
