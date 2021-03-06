import numpy as np
import atpy
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from astropy.table import Table
import pandas as pd
from matplotlib.colors import LogNorm
import schechter

def CalcOmega(massfitx, massfity):
    yrho = []
    dMH2 = massfitx[1] - massfitx[0]
    for i in range(0,len(massfity)):
        yrho.append(massfity[i]+massfitx[i])
    yrho = np.array(yrho)
    rhocrit = 9.2*(10**(-27))
    rhoH2 = (np.sum((10**yrho)*dMH2)*(2*(10**30)))/((3.086*(10**22))**3)
    OmegaH2 = (rhoH2/rhocrit)*(10000)
    return OmegaH2, yrho, np.sum((10**yrho)*dMH2)/(10**7)

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
    x,y = list(data[:,0]), list(data[:,1])
    return x,y

def boundary(data, MH2axis, Vmaxis, bins, sig, u,c,l):
    x = np.linspace(7.5,11,500)
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

def Plotrho(x, PGy, BGy, PAy, BAy, rhos):
    PGrho, BGrho, PArho, BArho = rhos
    alpha=-1.18
    mst1 = (2.81*(10**9))/(0.7**2)
    phist1 = 0.0089*(0.7**3)
    x_keres = 10**np.linspace(7, 11, 500)
    y_keres = np.log10((phist1)*((x_keres/(mst1))**(alpha+1))*np.exp(-x_keres/mst1)*np.log(10))
    x_keres = np.log10(x_keres)
    yrhokeres = y_keres + x_keres
    omegakeres =  CalcOmega(x_keres, y_keres)[2]
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].plot(x,PGy, label = 'SFR_Peng + MH2-SFR ' + str(round(rhos[0],2)), color='r')
    ax[0,0].plot(x,BGy, label = 'SFR_Best + MH2-SFR '+ str(round(rhos[1],2)), color='b')
    ax[0,0].plot(x,PAy, label = 'SFR_Peng + fH2 '+ str(round(rhos[2],2)), color='g')
    ax[0,0].plot(x,BAy, label = 'SFR_Best + fH2 '+ str(round(rhos[3],2)), color='k')
    ax[0,0].plot(x_keres,yrhokeres, 'k--', label = 'keres ' + str(round(omegakeres,2)))
    ax[0,0].set_xlim(6.8,11.2)
    ax[0,0].set_ylim(4.5,7.2)
    plt.legend(loc=3)
    plt.savefig('img/scal/'+ 'rhoh2' + '.pdf', format='pdf', dpi=250, transparent = False)

def MH2varSFR(total, bins, x_keres, y_keres, res, FullSchech, FullDetSchech,  sdssData, y_ober):
    bl = len(bins)
    data3 = np.loadtxt('gzl.txt')
    xbins = np.linspace(7.5,11,300)
    sdssSchechAm = Schechter(sdssData, 5, 4, bins)
    SA = boundary(sdssData, 5,4, bins, 0.23, bl-1, bl-1, bl-3)
    SAOmega, SAy, SArho = CalcOmega(SA[0], SA[1])
    SAOmegau, SAyu, SArhou = CalcOmega(SA[0], SA[2])
    SAOmegal, SAyl, SArhol = CalcOmega(SA[0], SA[3])
    print SAOmega, SAOmegal, SAOmegau
    para = schechter.log_schechter_fit(sdssSchechAm[2][:-1], sdssSchechAm[1][:-1])
    centreline = schechter.log_schechter(xbins, *para)
    a = 0.1
    print 'ressssssssssssssssss', res
    # 0: M*group | 1: dM* | 2: phi | 3: N | 4: M* | 5:SFR_SDSS | 6: SFR_Best
    #|7: MH2_SDSS_G |8: MH2_Best_G | 9: Vm | 10: MH2_SDSS_A |11: MH2_Best_A

    ###################################################################
    PG = boundary(total, 7, 9, bins, res, bl-1, bl, bl-3)
    PGOmega, PGy, PGrho = CalcOmega(PG[0], PG[1])
    PGOmegau, PGyu, PGrhou = CalcOmega(PG[0], PG[2])
    PGOmegal, PGyl, PGrhol = CalcOmega(PG[0], PG[3])
    dPGOmega = max([PGOmega- PGOmegau, PGOmega - PGOmegal ])
    print 'Peng, Accurso', PGOmega, PGOmegau, PGOmegal
    ###################################################################
    BG = boundary(total, 8, 9, bins, res, bl-1, bl-2, bl-3)
    BGOmega, BGy, BGrho = CalcOmega(BG[0], BG[1])
    BGOmegau, BGyu, BGrhou = CalcOmega(BG[0], BG[2])
    BGOmegal, BGyl, BGrhol = CalcOmega(BG[0], BG[3])
    print 'Best, Accurso', BGOmega, BGOmegau, BGOmegal
    dBGOmega = max([BGOmega- BGOmegau, BGOmega - BGOmegal ])
    ###################################################################
    PA = boundary(total, 10, 9, bins, 0.23, bl-1, bl-2, bl-3)
    PAOmega, PAy, PArho = CalcOmega(PA[0], PA[1])
    PAOmegau, PAyu, PArhou = CalcOmega(PA[0], PA[2])
    PAOmegal, PAyl, PArhol = CalcOmega(PA[0], PA[3])
    print 'Peng, Saintonge', PAOmega, PAOmegau, PAOmegal
    dPAOmega = max([PAOmega- PAOmegau, PAOmega - PAOmegal ])
    ###################################################################
    BA = boundary(total, 11, 9, bins, 0.23, bl-1, bl-2, bl-4)
    BAOmega, BAy, BArho = CalcOmega(BA[0], BA[1])
    BAOmegau, BAyu, BArhou = CalcOmega(BA[0], BA[2])
    BAOmegal, BAyl, BArhol = CalcOmega(BA[0], BA[3])
    print 'Best, Saintonge', BAOmega, BAOmegau, BAOmegal
    dBAOmega = max([BAOmega- BAOmegau, BAOmega - BAOmegal ])
    ###################################################################
    Plotrho(PG[0], PGy, BGy, PAy, BAy, [PGrho, BGrho, PArho, BArho])
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
    N, rho, xbin, sigma, rhoH2 = FullSchech
    # ax[0,0].errorbar(xbin, rho, fmt='h', markersize = 12, linewidth=2, mew=2, capthick=3, mfc='limegreen', mec='g' , label='COLD GASS D+ND + Genzel+12', zorder=2)
    # ax[0,0].errorbar(FullDetSchech[2], FullDetSchech[1], alpha = 0.2, fmt='h', markersize = 12, linewidth=2, mew=2, capthick=3, mfc='blue', mec='navy' , label='COLD GASS D + Genzel+12', zorder=1)
    # ax[0,0].plot(PG[0], PG[1], label = r'$\mathrm{SFR_{Peng}}$'+ ' + '+'Accurso+16', color='r', linewidth = 2)
    # ax[0,0].plot(PG[0], PG[2], color='none')
    # ax[0,0].plot(PG[0], PG[3], color='none')
    # ax[0,0].fill_between(PG[0], PG[2], PG[3], color ='r', alpha = a)
    ax[0,0].plot(BG[0], BG[1], label = r'$\mathrm{SFR_{Best}}$'+ ' + '+'Accurso+16', color='b', linewidth = 2)
    # ax[0,0].plot(BG[0], BG[2], color='none')
    # ax[0,0].plot(BG[0], BG[3], color='none')
    ax[0,0].fill_between(BG[0], BG[2], BG[3], color ='b', alpha = a)
    # ax[0,0].plot(PA[0], PA[1], label = r'$\mathrm{SFR_{Peng}}$'+ ' + '+'Saintonge+16', color='g', linewidth = 2)
    # ax[0,0].plot(PA[0], PA[2], color='m')
    # ax[0,0].plot(PA[0], PA[3], color='m')
    # ax[0,0].fill_between(PA[0], PA[2], PA[3], color ='g', alpha = a)
    # ax[0,0].plot(BA[0], BA[1], label = r'$\mathrm{SFR_{Best}}$'+ ' + '+'Saintonge+16', color='orange', linewidth = 2)
    # ax[0,0].scatter(BA[0], BA[2], color='k')
    # ax[0,0].scatter(BA[0], BA[3], color='g')
    # ax[0,0].fill_between(BA[0], BA[2], BA[3], color ='k', alpha = a)
    # ax[0,0].plot(xbins, centreline, color = 'm', linewidth = 2, label = r'$\mathrm{SFR_{SDSS}}$'+ ' + '+'Saintonge+16')
    # ax[0,0].plot(BestGio[2], BestGio[1], label = 'BestGio')
    # ax[0,0].plot(FullSchech_Best_D[2], FullSchech_Best_D[1], label = 'Det Am')
    # ax[0,0].plot(FullSchech_Best_D_G[2], FullSchech_Best_D_G[1], label = 'Det Am')
    # ax[0,0].plot(FullSchech_Best_D_S_G[2], FullSchech_Best_D_S_G[1], label = 'Det S G')
    # ax[0,0].plot(FullSchech_Best_D_S_A[2], FullSchech_Best_D_S_A[1], label = 'Det S A')
    # ax[0,0].scatter(FullDetSchech[2], FullDetSchech[1], label = 'Fulldet')
    # ax[0,0].errorbar(data3[:,0], data3[:,1], fmt='h', markersize = 12, linewidth=2, mew=2, capthick=3, mfc='limegreen', mec='g' , label='COLD GASS D+ND + Genzel+12', zorder=2)
    # ax[0,0].plot(np.log10(x_keres), y_keres, 'k--', label = 'Keres+03', linewidth = 2)
    # ax[0,0].plot(np.log10(x_keres), y_ober, 'k--', label = 'Obreschkow+09', linewidth = 2)
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{\odot}]}$', fontsize=30)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=30)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.75)
    plt.legend(fontsize = 18, loc=3)
    plt.tight_layout()
    plt.savefig('img/scal/StarMass.pdf', format='pdf', dpi=250, transparent=True)

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
