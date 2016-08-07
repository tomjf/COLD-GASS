import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from schechter import log_schechter_fit, log_schechter
import random
import scal_relns
import atpy
from scipy import integrate
import pandas as pd
from scipy.stats import kde
from matplotlib import cm
from sfrBestSDSS import convertsdss
from GAMA import MH2varSFR
from GAMA import PlotdMH2

# Omega H2 ################################################################
def CalcOmega(massfitx, massfity):
    yrho = []
    dMH2 = massfitx[1] - massfitx[0]
    for i in range(0,len(massfity)):
        yrho.append(massfity[i]+massfitx[i])
    rhocrit = 9.2*(10**(-27))
    rhoH2 = (np.sum((10**yrho)*dMH2)*(2*(10**30)))/((3.086*(10**22))**3)
    OmegaH2 = (rhoH2/rhocrit)*(10000)
    return OmegaH2
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def log_schechter(logL, log_rho, log_Lstar, alpha):
#     rholist = []
#     L = 10**logL
#     pstar = 10**log_rho
#     Lstar = 10**log_Lstar
#     log = np.log(10)
#     for i in range(0,len(L)):
#         frac = L[i]/Lstar
#         rho = pstar*(frac**(alpha+1))*math.exp(-frac)*log
#         rholist.append(rho)
#     return np.log10(rholist)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def schechter(L, Ls, dL, phis, alph):
    philist = []
    for i in range(0,len(L)):
        frac = 10**(L[i]-Ls)
        exp = math.exp(-frac)
        phibit = (phis*(frac**(alph+1)))
        log = np.log(10)
        philist.append((phibit*exp*log))
    return philist
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def doubleschechter(L, Ls, dL, phi1, phi2, alph1, alph2):
    philist = []
    for i in range(0,len(L)):
        frac = 10**(L[i]-Ls)
        exp = math.exp(-frac)
        phibit1 = (phi1*(frac**(alph1+1)))
        phibit2 = (phi2*(frac**(alph2+1)))
        log = np.log(10)
        philist.append(exp*log*(phibit1+phibit2))
    return philist
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def mainSequence(blues, spread, Mindex, AnotP):
    newblues = np.zeros((np.shape(blues)[0], np.shape(blues)[1]+1))
    for index, M in enumerate(blues[:,Mindex]):
        newblues[index, :5] = blues[index, :5]
        if AnotP == True:
            newblues[index,5] = - (2.332*M) + (0.4156*M*M) - (0.01828*M*M*M)
            if spread == True:
                newblues[index,5] += random.gauss(0,0.3)
        else:
            logsSFR = -10.0 - (0.1*(M-10.0))
            newblues[index,5] = logsSFR + M
            if spread == True:
                newblues[index,5] += random.gauss(0,0.3)
    return newblues
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def cloud(reds):
    newreds = np.zeros((np.shape(reds)[0], np.shape(reds)[1]+1))
    for index, M in enumerate(reds[:,4]):
        newreds[index, :5] = reds[index, :5]
        newreds[index, 5] = -1 + random.gauss(0,0.4)
    return newreds
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def OmegaH2(bins, yrho):
    rhocrit = 9.2*(10**(-27))
    dMH2 = bins[1] - bins[0]
    rhoH2 = (np.sum((10**yrho)*dMH2)*(2*(10**30)))/((3.086*(10**22))**3)
    OmegaH2 = (rhoH2/rhocrit)*(10000)
    return OmegaH2
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Vm(data, minz, maxz):
    Omega = 2.295
    VVmlist = np.zeros((len(data),1))
    Vmlist = np.zeros((len(data),1))
    x,y = np.zeros((1,1)), np.zeros((1,1))
    x[0,0], y[0,0] = minz, maxz
    D_in = float(lumdistance(x,0)[0,1])
    D_out = float(lumdistance(y,0)[0,1])
    Vm =  (1.0/3.0)*((D_out**3)-(D_in**3))*(Omega)
    Vmlist = np.full((len(data),1), Vm)
    data = np.hstack((data,Vmlist))
    return data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def lumdistance(data, zaxis):
    omega_m = 0.31                          # from Planck
    omega_l = 0.69                          # from Planck
    c = 3*math.pow(10,5)                    # in km/s
    Ho = 75                                 # in km/(s Mpc)
    f = lambda x : (((omega_m*((1+z)**3))+omega_l)**-0.5)
    Dlvals = np.zeros((len(data),1))
    for i in range(0,len(data)):
        z = data[i,zaxis]
        integral = integrate.quad(f, 0.0, z)    # numerically integrate to calculate luminosity distance
        Dm = (c/Ho)*integral[0]
        Dl = (1+z)*Dm                           # calculate luminosity distance
        #DH = (c*z)/Ho                          # calculate distance from Hubble law for comparison
        Dlvals[i,0] = Dl
    data = np.hstack((data,Dlvals))
    return data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def PlotBaldry(L, yBaldry, yred, yblue):
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].set_xlim(8,11.6)
    ax[0,0].set_ylim(-5,-1)
    ax[0,0].plot(L,np.log10(yBaldry), 'k', linewidth = 3, label = 'Total')
    ax[0,0].plot(L,np.log10(yred), 'r', linewidth = 3, label = 'Red')
    ax[0,0].plot(L,np.log10(yblue), 'b', linewidth = 3, label = 'Blue')
    ax[0,0].set_xlabel(r'$\mathrm{log \, M_{*}\, [M_{\odot}]}$', fontsize = 20)
    ax[0,0].set_ylabel(r'$\mathrm{log \, \phi \,[Mpc^{-3}\, dex^{-1}]}$', fontsize = 20)
    plt.legend(fontsize=13)
    plt.savefig('img/scal/Baldry.pdf', format='pdf', dpi=250, transparent = False)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def PlotMSFR(bluepop, redpop, x, z, data):
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].set_xlim(8,11.5)
    ax[0,0].set_ylim(-2.5,1)
    ax[0,0].scatter(bluepop, x, color = 'b', s = 5)
    ax[0,0].scatter(redpop, z, color = 'r', s = 5)
    ax[0,0].plot(data[:,0], data[:,1], '-', color = 'limegreen', linewidth = 5)
    ax[0,0].set_xlabel(r'$\mathrm{log \, M_{*}\, [M_{sun}]}$', fontsize = 20)
    ax[0,0].set_ylabel(r'$\mathrm{log \, SFR\, [M_{\odot}\,yr^{-1}]}$', fontsize = 20)
    plt.savefig('img/scal/MSFR.pdf', format='pdf', dpi=250, transparent = False)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def PlotHist(bluepop, redpop):
    bins = np.linspace(7.5,11.5,25)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].hist(bluepop, bins, normed = 1, alpha=0.5, color = 'b')
    ax[0,0].hist(redpop, bins, normed = 1, alpha=0.5, color = 'r')
    ax[0,0].set_xlabel(r'$\mathrm{log \, M_{*}\, [M_{sun}]}$', fontsize = 20)
    ax[0,0].set_ylabel(r'$\mathrm{Number \, Count}}$', fontsize = 20)
    ax[0,0].set_xlim(7.8,11.5)
    plt.savefig('img/scal/Hist.pdf', format='pdf', dpi=250, transparent = False)
# schechter only ###############################################################
def PlotRhoH2(PengAmelie,x, y):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(PengAmelie[2], PengAmelie[4], marker = 's', s = 100, edgecolor='blue', linewidth='2', facecolor='none', label = 'Low Mass')
    ax[0,0].scatter(x, y, 'k--')
    plt.savefig('img/scal/pH2.png')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def PlotSimMH2(blues, reds):
    bins = np.linspace(7.5,11.5,25)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(blues[:,5], blues[:,6], color = 'b')
    ax[0,0].scatter(reds[:,5], reds[:,6], color = 'r')
    ax[0,0].set_ylabel(r'$\mathrm{log \, M_{H2}\, [M_{sun}]}$', fontsize = 20)
    ax[0,0].set_xlabel(r'$\mathrm{log \, SFR\, [M_{sun}\,yr^{-1}]}$', fontsize = 20)
    # ax[0,0].set_xlim(7.8,11.5)
    plt.savefig('img/scal/MH2SFR.pdf', format='pdf', dpi=250, transparent = False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def PlotSchechter(PengAmelie,redSch, blueSch, x, y_scalfit, x_scal, y_keres):
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].errorbar(PengAmelie[2], PengAmelie[1], fmt = 'o', markersize = 10, color = 'red', label = 'Scaling Relation Method')
    ax[0,0].plot(x, y_scalfit, color ='red', label = 'Scaling Relation Fit')
    ax[0,0].plot(x_scal, y_keres, 'k--', label = 'Keres+03')
    # ax[0,0].scatter(x_scal, y_scal)
    # ax[0,0].errorbar(redSch[2], redSch[1], fmt = 'o', markersize = 10, color = 'red', label = 'Red')
    # ax[0,0].errorbar(blueSch[2], blueSch[1], fmt = 'o', markersize = 10, color = 'blue', label = 'Blue')
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.5)
    plt.legend(fontsize = 13)
    plt.savefig('img/scal/MH2.pdf', format='pdf', dpi=250, transparent = False)
    # plt.savefig('img/MH2.png', transparent = False ,dpi=250)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def PlotSchechterMass(MassSchB, MassSchR, L, yred, yblue):
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].errorbar(MassSchB[2], MassSchB[1], fmt = 'o', markersize = 10, color = 'blue', label = 'Blue')
    ax[0,0].errorbar(MassSchR[2], MassSchR[1], fmt = 'o', markersize = 10, color = 'red', label = 'Red')
    ax[0,0].plot(L,np.log10(yred), 'r', linewidth = 3)
    ax[0,0].plot(L,np.log10(yblue), 'b', linewidth = 3)
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{*}\,[M_{sun}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{M}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(8, 11.5)
    plt.legend(fontsize = 13)
    plt.savefig('img/scal/Mstar.pdf', format='pdf', dpi=250, transparent = False)
    # plt.savefig('img/MH2.png', transparent = False ,dpi=250)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def PlotSchechSDSS(FullSchech, sdssSchech, sdssSchechAm, PengAmelie,PengGio, x_keres, y_keres, y_ober):
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].errorbar(sdssSchech[2], sdssSchech[1], fmt = 'o', markersize = 8, color = 'm', label = 'SDSS-G')
    ax[0,0].errorbar(PengAmelie[2], PengAmelie[1], fmt = 's', markersize = 8, color = 'm', label = 'Sim-G')
    ax[0,0].errorbar(PengGio[2], PengGio[1], fmt = 's', markersize = 8, color = 'c', label = 'Sim-A')
    ax[0,0].errorbar(FullSchech[2], FullSchech[1], fmt = 's', markersize = 8, color = 'r', label = 'COLD GASS SFRs')
    ax[0,0].errorbar(sdssSchechAm[2], sdssSchechAm[1], fmt = 'o', markersize = 8, color = 'c', label = 'SDSS-A')
    ax[0,0].plot(np.log10(x_keres), y_keres, 'k--', label = 'Keres+03')
    ax[0,0].plot(np.log10(x_keres), y_ober, 'k', label = 'Obreschkow+09')
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.5)
    plt.legend(fontsize = 13)
    plt.savefig('img/scal/'+ 'SDSS' + '.pdf', format='pdf', dpi=250, transparent = False)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def PlotSchechSDSSv2(GASS_Schech, FullSchech, FullSchech_Best, totSch_data, PengAmelie,PengGio, LSch, HSch, NDSch, NDSch2, sigma, x_keres, y_keres, y_ober, FullDetSchech, FullSchech_m):
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].errorbar(FullSchech_Best[2], FullSchech_Best[1], fmt = 'o', markersize = 8, color = 'g', label = 'CG with SFRbest')
    # ax[0,0].errorbar(FullSchech_m[2], FullSchech_m[1], fmt = 's', markersize = 4, color = 'k', label = 'All data')
    # ax[0,0].errorbar(FullSchech[2], FullSchech[1], fmt = 'o', markersize = 8, color = 'c', label = 'COLD GASS, using SFR_SDSS')
    # ax[0,0].errorbar(totSch_data[2], totSch_data[1], yerr=sigma, fmt = 's', markersize = 8, color = 'r', label = 'COLD GASS data pts')
    # ax[0,0].scatter(PengGio[2], PengGio[1], marker = 'h', s = 100, edgecolor = 'm', linewidth='2', facecolor='hotpink', label = 'Amelie Sim')
    # ax[0,0].scatter(PengAmelie[2], PengAmelie[1], marker = 'h', s = 100, edgecolor = 'green', linewidth='2', facecolor='lawngreen', label = 'Gio Sim')
    ax[0,0].scatter(FullDetSchech[2], FullDetSchech[1], marker = 's', s = 100, edgecolor = 'blue', linewidth='2', facecolor='none', label = 'CG det')
    ax[0,0].scatter(GASS_Schech[2], GASS_Schech[1], marker = 'o', s = 50, edgecolor = 'red', linewidth='2', facecolor='red', label = 'GASS')
    # ax[0,0].fill_between(PengGio[2], PengGio[1], PengAmelie[1], alpha = 0.2, color = 'k')
    # ax[0,0].scatter(LSch[2], LSch[1], marker = 's', s = 100, edgecolor = 'coral', linewidth='2', facecolor='none', label = 'L')
    # ax[0,0].scatter(HSch[2], HSch[1], marker = 's', s = 100, edgecolor = 'green', linewidth='2', facecolor='none', label = 'H')
    # ax[0,0].scatter(NDSch[2], NDSch[1], marker = 's', s = 100, edgecolor = 'black', linewidth='2', facecolor='none', label = 'ND 5sig')
    # ax[0,0].scatter(NDSch2[2], NDSch2[1], marker = 's', s = 100, edgecolor = 'm', linewidth='2', facecolor='none', label = 'ND 3sig')
    ax[0,0].plot(np.log10(x_keres), y_keres, 'k--', label = 'Keres+03')
    # ax[0,0].plot(np.log10(x_keres), y_ober, 'k', label = 'Obreschkow+09')
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{\odot}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.5)
    plt.legend(fontsize = 10)
    plt.savefig('img/scal/'+ 'SDSSv4' + '.pdf', format='pdf', dpi=250, transparent = False)
# create galaxies ##############################################################

def createGals(red, V):
    # make a large list with the property of each galaxy.
    for i in range(0,len(red)):
        red[i,3] = int(red[i,2]*V*red[i,1])
        redpop = []
        for j in range(0,int(red[i,3])):
            redpop.append(random.uniform((red[i,0]-(0.5*red[i,1])), (red[i,0]+(0.5*red[i,1]))))
        redpopsec = np.zeros((len(redpop),5))
        redpopsec[:,0] = red[i,0]
        redpopsec[:,1] = red[i,1]
        redpopsec[:,2] = red[i,2]
        redpopsec[:,3] = red[i,3]
        redpopsec[:,4] = redpop
        if i == 0:
            reds = redpopsec
        else:
            reds = np.vstack((reds,redpopsec))
    return reds

# schechter bins ###############################################################
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
# Error Sampling ###############################################################
def errors(data, x, y, output):
    frac = 0.5
    eridx = int(len(data)*frac)
    idx = np.linspace(0,len(data)-1,len(data))
    spread = np.zeros((eridx, len(x)-1))
    for i in range(0, eridx):
        random.shuffle(idx)
        idx1 = idx[:eridx]
        newdata = np.zeros((eridx, np.shape(data)[1]))
        for j in range(0,len(newdata)):
            newdata[j,:] = data[idx[j],:]
        newdata[:,output['Vm']] = newdata[:,output['Vm']]*0.8
        totSch = Schechter(newdata, 6, 7, x)
        drho = PengAmelie[1] - y
        spread[i,:] = drho
    return spread
################################################################################
def sdssMethod(zl, zh):
    galinfo = atpy.Table('data/sdss/gal_info_dr7_v5_2.fit')
    sfr = atpy.Table('data/sdss/gal_totsfr_dr7_v5_2.fits')
    mstar = atpy.Table('data/sdss/totlgm_dr7_v5_2.fit')
    sdssData = np.zeros((len(galinfo),3))
    for i in range(0,len(galinfo)):
        # redshift from galinfo
        sdssData[i,0] = galinfo[i][12]
        # stellar mass from totlgm
        sdssData[i,1] = mstar[i][6]
        # total sfr from totsfr
        sdssData[i,2] = sfr[i][0]
    sdssData = sdssData[sdssData[:,0] > zl]
    sdssData = sdssData[sdssData[:,0] < zh]
    sdssData = sdssData[sdssData[:,1] > 4]
    return sdssData
################################################################################
def SFRMH2(data):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(data[:,2], data[:,3], s = 1)
    ax[0,0].set_xlim(-1.5, 2)
    ax[0,0].set_ylim(7.5, 11)
    ax[0,0].set_xlabel(r'$\mathrm{log\, SFR\,[M_{sun}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
    plt.savefig('img/scal/sfrmh2.pdf', format='pdf', dpi=250, transparent = False)
################################################################################
def SFRMSTAR(data, FullData):
    nbins=200
    data2 = np.vstack((data[:,1], data[:,2]))
    x, y = data[:,1], data[:,2]
    k = kde.gaussian_kde(data2)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].contourf(xi, yi, zi.reshape(xi.shape), 15, cmap = cm.gray_r)
    ax[0,0].scatter(data[:,1], data[:,2], s = .1, alpha=0.5, color = 'b', label = 'COLD GASS')
    # ax[0,0].scatter(FullData[:,3], FullData[:,2], s = 10, color = 'r', label = 'COLD GASS')
    ax[0,0].set_xlim(8, 11.5)
    ax[0,0].set_ylim(-2.5, 1)
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{*}\,[M_{sun}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, SFR\,[M_{sun}\,yr^{-1}]}$', fontsize=18)
    plt.savefig('img/scal/sfrmstar.pdf', format='pdf', dpi=250, transparent = False)
################################################################################
def SFRHist(sdssData, gioData, FullData, total):
    bins = np.linspace(-3.0, 2.0, 25)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].hist(sdssData[:,2], bins, normed = 1, alpha=0.3, color = 'g', label = 'SDSS')
    ax[0,0].hist(gioData[:,1], bins, normed = 1, alpha=0.3, color = 'm', label = 'Gio Data')
    ax[0,0].hist(FullData[:,2], bins, normed = 1, alpha=0.3, color = 'b', label = 'COLD GASS')
    ax[0,0].set_ylabel(r'$\mathrm{Number \,(normalised)}$', fontsize=18)
    ax[0,0].hist(total[:,5], bins, normed = 1, alpha=0.2, color = 'c', label = 'Simulated')
    ax[0,0].set_xlabel(r'$\mathrm{log\, SFR\,[M_{\odot}\,yr^{-1}]}$', fontsize=18)
    plt.legend(fontsize = 13)
    plt.savefig('img/scal/sfrhist.pdf', format='pdf', dpi=250, transparent = False)
################################################################################
def AmGasFrac(data, Mindex, SFRindex):
    MH2list = np.zeros((len(data),1))
    for i in range(0,len(data)):
        logM, logSFR = data[i,Mindex], data[i,SFRindex]
        logfH2 = 6.02 + (0.704*logSFR) - (0.704*logM)
        fH2 = 10**logfH2
        MH2list[i,0] = np.log10(fH2*(10**logM))
    return np.hstack((data, MH2list))
################################################################################
def GetCOLDGASS(fit, res):
    Full = atpy.Table('data/COLDGASS_full.fits')
    FullData = np.zeros((len(Full),6))
    lorh = []
    for i,rows in enumerate(Full):
        #z|flag|SFR|SFR_BEST|M*|Lumdist
        FullData[i,0] = rows[9]
        FullData[i,1] = rows[37]
        FullData[i,2] = rows[23] # 23 for sfr_sdss 24 for sfr_best
        FullData[i,3] = np.log10(rows[24])
        FullData[i,4] = rows[20]
        FullData[i,5] = rows[10]
        lorh.append(rows[1])
    data = pd.DataFrame({   'z': FullData[:,0],
                            'flag': FullData[:,1],
                            'SFR': FullData[:,2],
                            'SFR_Best': FullData[:,3],
                            'M*': FullData[:,4],
                            'D_L': FullData[:,5]})
    data['group'] = lorh
    LMass = (data.loc[data['group'] == 'L'])
    HMass = (data.loc[data['group'] == 'H'])
    LMassa = LMass[['z', 'flag', 'SFR', 'SFR_Best', 'M*', 'D_L']].values
    HMassa = HMass[['z', 'flag', 'SFR', 'SFR_Best', 'M*', 'D_L']].values
    LMassa = Vm1(LMassa, 5, min(LMassa[:,0]), max(LMassa[:,0]), 1)
    HMassa = Vm1(HMassa, 5, min(HMassa[:,0]), max(HMassa[:,0]), 2)
    #0:z|1:flag|2:SFR|3:SFR_BEST|4:M*|5:Lumdist|6:V/Vm|7:Vm|8:MH2_SDSS_A|9:MH2_BEST_A|10:MH2_BEST_G|11:MH2_SDSS_G|
    FullData = np.vstack((LMassa, HMassa))
    FullData = AmGasFrac(FullData, 4, 2) # SFR_SDSS
    FullData = AmGasFrac(FullData, 4, 3) # SFR_BEST
    FullData = np.hstack((FullData, np.zeros((len(FullData),2))))
    FullData[:,10] = scal_relns.third(FullData[:,3], *fit[0])
    FullData[:,11] = scal_relns.third(FullData[:,2], *fit[0])
    FullDet = FullData[FullData[:,1] == 1]
    return FullData, FullDet
################################################################################
def sfrbest(FullData, SFRBL):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(FullData[:,2], FullData[:,8], alpha = 0.1, label='SDSS', color = 'b')
    ax[0,0].scatter(FullData[:,3], FullData[:,9], alpha = 0.1, label='best', color = 'r')
    ax[0,0].scatter(SFRBL[:,1], SFRBL[:,6], alpha = 0.1, label='GASS', color = 'g')
    ax[0,0].set_xlabel(r'$\mathrm{log \, SFR}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, MH2}$', fontsize=18)
    plt.legend(fontsize = 13)
    plt.savefig('img/scal/SFR_Best.pdf', format='pdf', dpi=250, transparent = False)
################################################################################
def Vm1(data, Dlaxis, minz, maxz, L):
    # Omega = 0.483979888662
    Omega = 0.427304474238
    if L == 1:
        Omega = 0.353621392624
        N_COLDGASS = 89.0
        N_SDSS = 764.0
    elif L == 2:
        N_COLDGASS = 366.0
        N_SDSS = 12006.0
    elif L == 3:
        Omega = 0.427304474238
        N_COLDGASS = 1255.0
        N_SDSS = 12770
    VVmlist = np.zeros((len(data),1))
    Vmlist = np.zeros((len(data),1))
    x,y = np.zeros((1,1)), np.zeros((1,1))
    x[0,0], y[0,0] = minz, maxz
    D_in = float(lumdistance(x,0)[0,1])
    D_out = float(lumdistance(y,0)[0,1])
    Vm =  (1.0/3.0)*(N_COLDGASS/N_SDSS)*((D_out**3)-(D_in**3))*(Omega)
    for i in range(0,len(data)):
        Dl = data[i,Dlaxis]
        V = ((4*math.pi)/3)*Dl*Dl*Dl
        VVmlist[i,0] = (V/Vm)
        Vmlist[i,0] = Vm
    data = np.hstack((data,VVmlist))
    data = np.hstack((data,Vmlist))
    return data
################################################################################
def sfrbest13k():
    df = pd.read_csv('data/allGASS_SFRbest_simple_t1.csv')
    SFRBL = df[['GASS', 'SFR_best', 'SFRerr_best', 'SFRcase_best']].values
    df = pd.read_csv('data/PS_100701.csv')
    coords_H = df[['GASS', 'z', 'ra', 'dec', 'MASS_P50']].values
    df = pd.read_csv('data/LOWMASS_MASTER.csv')
    coords_L = df[['GASSID', 'Z', 'RA', 'DEC', 'MASS']].values
    GASS = np.vstack((coords_H, coords_L))
    mstar, z = np.zeros((len(SFRBL),1)), np.zeros((len(SFRBL),1))
    for index, element in enumerate(SFRBL):
        for index1, element1 in enumerate(GASS):
            if element[0] == element1[0]:
                mstar[index] = element1[4]
                z[index] == element1[1]
    SFRBL = np.hstack((SFRBL, mstar))
    SFRBL = np.hstack((SFRBL, z))
    SFRBL = SFRBL[SFRBL[:,4]>0]
    SFRBL[:,1] = np.log10(1.46*SFRBL[:,1])
    SFRBL = AmGasFrac(SFRBL, 4, 1)
    SFRBL = np.hstack((SFRBL, np.zeros((len(SFRBL),1))))
    SFRBL = Vm1(SFRBL, 7, min(GASS[:,1]), max(GASS[:,1]), 3)
    # 0:ID|1:SFGRbest|2:err|3:case|4:M*|5:z|6:MH2|7:Dl|8:V/Vm|9:Vm
    return SFRBL
################################################################################
def main(bins, totSch_data, PengGio, totSch3, sigma, LSch, HSch, NDSch, NDSch2, FullDetSchech, FullSchech_m, Fulldetmain):
    random.seed(13990)
    V = 100000
    L = np.linspace(6,12.5,35)
    LKeres = np.linspace(4,8,200)
    Spheroid = (3.67/10000), 10.74, -0.525
    Disk = (0.855/10000), 10.70, -1.39
    Keres = (7.2/10000), 7.0, -1.30
    output = { 'Vm':6, 'MH2':13}
    ################################################################################
    # #spheroid is red
    # ySpheroid = log_schechter(L, *Spheroid)
    # #disk is blue
    # yDisk = log_schechter(L, *Disk)
    # yKeres = schechter(LKeres, Keres[1], (L[1]-L[0]), Keres[0], Keres[2])

    yBaldry = doubleschechter(L, 10.66, (L[1]-L[0]), 0.00396, 0.00079, -0.35, -1.47)
    yred = schechter(L, 10.66, (L[1]-L[0]), 0.00396, -0.35)
    yblue = schechter(L, 10.66, (L[1]-L[0]), 0.00079, -1.47)

    ###### make a table for the red galaxies #######################################
    red = np.zeros((len(yred),4))
    # the luminosity bin
    red[:,0] = L
    # spacing between bins
    red[:,1] = L[1]-L[0]
    # the number of galaxies in this luminosity bin from the schechter function
    red[:,2] = yred
    #list of all the galaxies over all the bins
    reds = createGals(red, V)
    ###### make a table for the blue galaxies ######################################
    blue = np.zeros((len(yblue),4))
    # the luminosity bin
    blue[:,0] = L
    # spacing between bins
    blue[:,1] = L[1]-L[0]
    # the number of galaxies in this luminosity bin from the schechter function
    blue[:,2] = yblue
    #list of all the galaxies over all the bins
    bluepop = []
    # list all gals
    blues = createGals(blue, V)
    # M*group | dM* | phi | N | M*
    ###### make a table for all the galaxies #######################################
    Baldry = np.zeros((len(yBaldry),5))
    Baldry[:,0] = L
    Baldry[:,1] = 0.2
    Baldry[:,2] = yBaldry
    # total = np.append(bluepop, redpop)

    # 0: M*group | 1: dM* | 2: phi | 3: N | 4: M* | 5:SFR_SDSS | 6: SFR_Best
    #|7: MH2_SDSS_G |8: MH2_Best_G | 9: Vm | 10: MH2_SDSS_A |11: MH2_Best_A

    # add starformation rates

    blues = mainSequence(blues, True, 4, False)
    reds = cloud(reds)
    blues = np.hstack((blues, np.zeros((len(blues),4))))
    reds = np.hstack((reds, np.zeros((len(reds),4))))
    reds[:,6] = convertsdss(reds[:,5])
    blues[:,6] = convertsdss(blues[:,5])
    trend = mainSequence(Baldry, False, 0, False)
    data = np.zeros((len(L),2))
    data[:,0] = L
    data[:,1] = trend[:,5]

    datagio, fit, res = scal_relns.fitdata()
    # blues[:,6] = scal_relns.second2var((blues[:,4], blues[:,5]), *fit[0])
    # reds[:,6] = scal_relns.second2var((reds[:,4], reds[:,5]), *fit[0])
    blues[:,7] = scal_relns.third(blues[:,5], *fit[0])
    reds[:,7] = scal_relns.third(reds[:,5], *fit[0])
    blues[:,8] = scal_relns.third(blues[:,6], *fit[0])
    reds[:,8] = scal_relns.third(reds[:,6], *fit[0])
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SFRBL = sfrbest13k()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # add V
    blues[:,9] = V
    reds[:,9] = V
    total = np.vstack((blues, reds))
    total = AmGasFrac(total, 4, 5)
    total = AmGasFrac(total, 4, 6)
    # bins  = np.linspace(6, 10.5, 30)
    # bins  = np.linspace(min(total[:,6]), max(total[:,6]), 16)
    massbins = np.linspace(8,11.5,25)
    PengAmelie = Schechter(total, 10, 9, bins) # peng amelie
    BestAmelie = Schechter(total, 11, 9, bins) # peng amelie
    PengGio = Schechter(total, 7, 9, bins) # peng amelie
    BestGio = Schechter(total, 8, 9, bins) # peng amelie
    redSch = Schechter(reds, 6, 7, bins)
    blueSch = Schechter(blues, 6, 7, bins)
    MassSchB = Schechter(blues, 4, 7, massbins)
    MassSchR = Schechter(reds, 4, 7, massbins)
    x = np.linspace(7.5,10.5,200)
    scalfit = log_schechter_fit(PengAmelie[2], PengAmelie[1])
    x3 = np.linspace(7.5,10.5,16)
    y_scalfit = log_schechter(x3, *scalfit)

    x_scal = np.linspace(7.5, 10.5, 25)
    x_keres = 10**x_scal
    mst=np.log10((2.81*(10**9))/(0.7**2))
    alpha=-1.18
    phist=np.log10(0.0089*(0.7**3))
    mst1 = 10**mst
    phist1 = 10**phist
    y_keres = np.log10((phist1)*((x_keres/(mst1))**(alpha+1))*np.exp(-x_keres/mst1)*np.log(10))
    #####################################
    # corrected values from Obreschkow and Rawlings
    mstcorr = (7.5*(10**8))/(0.7**2)
    alphacorr = -1.07
    phistcorr = 0.0243*(0.7**3)
    y_ober = np.log10((phistcorr)*((x_keres/(mstcorr))**(alphacorr+1))*np.exp(-x_keres/mstcorr)*np.log(10))


    scal_para = log_schechter_fit(PengAmelie[2][9:], PengAmelie[1][9:])
    y_scal = log_schechter(x_scal, *scal_para)
    rhoscal = y_scal + x_scal

    # er = errors(total, bins, PengAmelie[1], output)
    # sigma = []
    # for i in range(0, np.shape(er)[1]):
    #     eri = er[:,i]
    #     eri = eri[abs(eri)<99]
    #     sigma.append(np.std(eri))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #COLD GASS
    FullData, FullDet = GetCOLDGASS(fit, res)
    FullSchech = Schechter(FullData, 8, 7, bins)
    FullSchech_Best = Schechter(FullData, 9, 7, bins)
    FullSchech_Best_D = Schechter(FullDet, 11, 7, bins)
    FullSchech_Best_D_G = Schechter(FullDet, 10, 7, bins)
    FullSchech_Best_D_S_A = Schechter(FullDet, 8, 7, bins)
    FullSchech_Best_D_S_G = Schechter(FullDet, 11, 7, bins)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ### SDSS METHOD~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get the SDSS data from the fits tables
    sdssData = sdssMethod(0.005, 0.02)
    sdssData = np.hstack((sdssData, np.zeros((len(sdssData),1))))
    # sdssData[:,3] = scal_relns.second2var((sdssData[:,1], sdssData[:,2]), *fit[0])
    # calculate the H2 mass by fitting to gio's scaling relation
    sdssData[:,3] = scal_relns.third(sdssData[:,2], *fit[0])
    # calculate Vm from the redshift limit and area of sky covered z,M*,SFR,MH2,Vm
    sdssData = Vm(sdssData, min(sdssData[:,0]), max(sdssData[:,0]))
    # bin the H2 data
    sdssSchech = Schechter(sdssData, 3, 4, bins)
    # calculate gas fraction using amelie's method z,M*,SFR,MH2_gio,Vm,MH2_am
    sdssData = AmGasFrac(sdssData, 1, 2)
    sdssSchechAm = Schechter(sdssData, 5, 4, bins)
    GASS_Schech = Schechter(SFRBL, 6, 9, bins)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #SFR COMPARISON
    MH2varSFR(total, bins, x_keres, y_keres, res)
    PlotdMH2(FullDet, Fulldetmain)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PlotSchechSDSS(FullSchech, sdssSchech, sdssSchechAm, PengAmelie,PengGio, x_keres, y_keres, y_ober)
    # SFRMH2(sdssData)
    # SFRMSTAR(sdssData, FullData)
    PlotBaldry(L, yBaldry, yred, yblue)
    PlotMSFR(blues[:,4], reds[:,4], blues[:,5], reds[:,5], data)
    # PlotHist(blues[:,4], reds[:,4])
    # PlotSimMH2(blues, reds)
    # PlotRhoH2(PengAmelie,x_scal, rhoscal)
    # sfrbest(FullData, SFRBL)
    # PlotSchechter(PengAmelie,redSch, blueSch, x3, y_scalfit, x_scal, y_keres)
    # PlotSchechterMass(MassSchB, MassSchR, L, yred, yblue)
    PlotSchechSDSSv2(GASS_Schech, FullSchech, FullSchech_Best_D, totSch_data, PengAmelie,PengGio,  LSch, HSch, NDSch, NDSch2, sigma, x_keres, y_keres, y_ober, FullDetSchech, FullSchech_m)
    # SFRHist(sdssData, datagio, FullData, total)
    return FullSchech, sdssSchech, sdssSchechAm, PengAmelie,PengGio, FullData

# FullSchech, sdssSchech, sdssSchechAm, PengAmelie,PengGio = main()
# bins = np.linspace(7.5,11.5,25)
# fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
# ax[0,0].hist(bluepop, bins, normed = 1, facecolor='blue', alpha = 0.1)
# ax[0,0].hist(redpop, bins, normed = 1, facecolor='red', alpha = 0.1)
# # ax[0,0].set_xlabel(r'$\mathrm{log \, M_{*}\, [M_{sun}]}$', fontsize = 20)
# # ax[0,0].set_ylabel(r'$\mathrm{Number \, Count}}$', fontsize = 20)
# # ax[0,0].set_xlim(7.8,11.5)
# plt.savefig('img/scal/Hist.eps', format='eps', dpi=250, transparent = False)
#
# fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False)
# #plt.plot(L, ySpheroid, 'r')
# #plt.plot(L, yDisk, 'b')
# #ax[0,0].plot(LKeres, yKeres, 'g')
# ax[0,0].plot(L,np.log10(yBaldry), 'k')
# ax[0,0].plot(L,np.log10(yred), 'r')
# ax[0,0].plot(L,np.log10(yblue), 'b')
# ax[0,0].set_xlabel(r'$log_{10}(M/M_{\odot})$', fontsize = 20)
# ax[0,0].set_ylabel(r'$log_{10}(N density)$', fontsize = 20)
# #
#
# ax[0,0].hist(bluepop, 25, normed = 1, facecolor='blue', alpha=0.5)
# ax[0,0].hist(redpop, 25, normed = 1, facecolor='red', alpha=0.5)
#
# # n1, bins1, patches1 = plt.hist(bluepop, 25, normed = 1, facecolor='blue', alpha=0.5)
# # n, bins, patches = plt.hist(redpop, 25, normed = 1, facecolor='red', alpha=0.5)
# # plt.savefig('BlueRedhist.png', transparent = False ,dpi=250)
# # #n2, bins2, patches2 = plt.hist(total, 25, normed=1, facecolor='k', alpha=0.5)
# # # plt.xlim(8,11.7)
# # # plt.ylim(-5,0)
# plt.savefig('img/scal/BlueRedhist.pdf', dpi=250, transparent = False)
#
#
# # fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False)
# ax[1,0].scatter(bluepop, x, color = 'b')
# ax[1,0].scatter(redpop, z, color = 'r')
# ax[1,0].plot(data[:,0], data[:,1], linewidth = 2, color='r')
# ax[1,0].set_xlabel(r'$log({M_{*}})$')
# ax[1,0].set_ylabel(r'$log({\mathrm{SFR}})$')
# ax[1,0].set_xlim(min(L),max(L))
# # # # plt.ylim(-2.5,1)
# ax[1,1].scatter(bluepop1[:,1], bluepop1[:,2], color = 'b')
# ax[1,1].scatter(redpop1[:,1], redpop1[:,2], color = 'r')
# ax[1,1].scatter(datagio[:,1], datagio[:,0], color = 'g')
# ax[1,1].set_xlabel(r'$log({\mathrm{SFR}})$')
# ax[1,1].set_ylabel(r'$log({\mathrm{MH2}})$')
# # # plt.show()
# plt.savefig('img/scal/all.png', dpi=250, transparent = False)
# plt.show()
