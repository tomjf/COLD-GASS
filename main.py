import numpy as np
import atpy
import math
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import asciidata
from scipy.optimize import curve_fit
import schechter
import csv
import pandas as pd
import random
import BlueRed
from gasfrac import testMassLimits

# Function to calculate the luminosity distance from z #########################
def lumdistance(data, zaxis):
    omega_m = 0.3                          # from Planck
    omega_l = 0.7                       # from Planck
    c = 3*math.pow(10,5)                    # in km/s
    Ho = 70                                 # in km/(s Mpc)
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

# Calculate CO Luminosity ######################################################
def lCalc(data, SCOaxis, zaxis, Dlaxis, correction):
    lums = np.zeros((len(data),1))
    for i in range(0,len(data)):                # for each galaxy in the dataset
        if correction == True:
            SCO_cor = data[i,SCOaxis]*6.0       # find the integrated CO flux
        else:
            SCO_cor = data[i,SCOaxis]
        C = 3.25*math.pow(10,7)                 # numerical const from eqn 4 paper 1
        freq = 111                              # observing frequency
        Dl = data[i,Dlaxis]
        SDSS_z = math.pow((1+data[i,zaxis]),-3)         # redshift component
        L_CO = C*SCO_cor*((Dl*Dl)/(freq*freq))*SDSS_z   # calculate CO luminosity
        lums[i,0] = L_CO
    data = np.hstack((data, lums))
    return data

# gio's conv factor#############################################################

def conversion_factor_equation(met, log_sfr, log_m, redshift):
    delta_ms_whitaker_2012 = log_sfr - log_m + 10.12 - 1.14*redshift + 0.19*redshift*redshift + (0.3 + 0.13*redshift)*(log_m - 10.5)
    if met > 8.8:
        if delta_ms_whitaker_2012 < -0.8:
            alpha_co_redshift_dependent = 10**(15.69 - 1.732*8.8 + 0.051*(-0.8))
        if delta_ms_whitaker_2012 > np.log10(20.0):
            alpha_co_redshift_dependent = 'nan'
        else:
            alpha_co_redshift_dependent = 10**(15.69 - 1.732*8.8 + 0.051*delta_ms_whitaker_2012)
    if met < 7.9:
        alpha_co_redshift_dependent = 'nan'
    else:
        if delta_ms_whitaker_2012 < -0.8:
            alpha_co_redshift_dependent = 10**(15.69 - 1.732*met + 0.051*(-0.8))
        if delta_ms_whitaker_2012 > np.log10(20.0):
            alpha_co_redshift_dependent = 'nan'
        else:
            alpha_co_redshift_dependent = 10**(15.69 - 1.732*met + 0.051*delta_ms_whitaker_2012)
    return alpha_co_redshift_dependent

def conversion_factor_equation2(met, log_sfr, log_m, redshift):
    delta_ms_whitaker_2012 = log_sfr - log_m + 10.12 - 1.14*redshift + 0.19*redshift*redshift + (0.3 + 0.13*redshift)*(log_m - 10.5)
    if met > 8.8:
        if delta_ms_whitaker_2012 < -0.8:
            alpha_co_redshift_dependent = 10**(15.69 - 1.732*8.8 + 0.051*(-0.8))
        elif delta_ms_whitaker_2012 > np.log10(20.0):
            alpha_co_redshift_dependent = 'nan'
        else:
            alpha_co_redshift_dependent = 10**(15.69 - 1.732*8.8 + 0.051*delta_ms_whitaker_2012)
    elif met < 7.9:
        alpha_co_redshift_dependent = 'nan'
    else:
        if delta_ms_whitaker_2012 < -0.8:
            alpha_co_redshift_dependent = 10**(15.69 - 1.732*met + 0.051*(-0.8))
        elif delta_ms_whitaker_2012 > np.log10(20.0):
            alpha_co_redshift_dependent = 'nan'
        else:
            alpha_co_redshift_dependent = 10**(15.69 - 1.732*met + 0.051*delta_ms_whitaker_2012)
    return alpha_co_redshift_dependent

def genzel(met):
    aco = 12.0 - (1.3*met)
    return 10**aco
def schruba(met):
    aco =  np.log10(8) + (-2*(met - 8.7))
    return 10**aco


# Remove non-detections ########################################################
def NonDetect(data, flagrow, detections):
    init = True
    newdata = np.zeros((1,np.shape(data)[1]))
    if detections == True:
        flag = 1.0
    else:
        flag = 2.0
    for i in range (0,len(data)):
        if data[i,flagrow] == flag:
            if init == True:
                newdata[0,:] = data[i,:]
                init = False
            else:
                newdata = np.vstack((newdata, list(data[i,:])))
    return newdata

# Sort into bins ###############################################################
def sortIntoBins(l,number):
    low, high = min(l), max(l)     # min max in logspace
    bins = np.linspace(low, high,num=number) # log-spaced bins
    N, xbins = [], []
    for i in range (1,len(bins)):
        inbin = [x for x in l if x > bins[i-1] and x < bins[i]]
        n = len(inbin)
        N.append(n)
        N.append(n)
        xbins.append(bins[i-1])
        xbins.append(bins[i])
    return np.log10(N), xbins

# Conversion to H2 mass ########################################################
def H2Conversion(data, Zindex, LCOindex, *args):
    alphaCOHMass = []
    for arg in args:
        alphaCOHMass = arg
    # alpha_CO = mZ + c (from Genzel et al)
    c = 12.0
    dc = 2.0
    m = -1.3
    dm = 0.26
    H2mass = np.zeros((len(data),3))
    if len(alphaCOHMass) > 0:
        for i in range(0,len(data)):
            H2mass[i,0] = alphaCOHMass[i]
            H2mass[i,1] = alphaCOHMass[i]*data[i,LCOindex]
            H2mass[i,2] = 0
    else:
        for i in range(0,len(data)):
            if data[i,3] > 10:
                alpha_CO_gal = 4.35
                dalpha = 0
            else:
                log_alpha_CO_gal = (m*data[i,Zindex]) + c
                alpha_CO_gal = math.pow(10,log_alpha_CO_gal)
                shallow = ((m+dm)*data[i,Zindex]) + (c-dc)
                steep = ((m-dm)*data[i,Zindex]) + (c+dc)
                shallow = math.pow(10,shallow)
                steep = math.pow(10,steep)
                dalpha = ([abs(alpha_CO_gal-shallow), abs(alpha_CO_gal-steep)])
                dalpha = max(dalpha)
            H2mass[i,0] = alpha_CO_gal
            H2mass[i,1] = alpha_CO_gal*data[i,LCOindex]
            H2mass[i,2] = dalpha
            # if alpha_CO_gal<4.3:
            #     print alpha_CO_gal
    data = np.hstack((data,H2mass))
    return data

# Vm calc ######################################################################
def Vm(data, Dlaxis, minz, maxz, L):
    # Omega = 0.483979888662
    Omega = 0.427304474238
    if L == 1: # low mass
        Omega = 0.353621392624
        N_COLDGASS = 89.0
        N_SDSS = 764.0
    elif L == 2: # high mass
        N_COLDGASS = 366.0
        N_SDSS = 12006.0
    elif L == 3:
        N_COLDGASS = 500.0
        N_SDSS = 12006.0
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

# schechter bins ###############################################################
def Schechter(data, LCOaxis, Vmaxis, bins):
    l = data[:,LCOaxis]
    l = np.log10(l)
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

# schechter density functional form ############################################
def schechfunc(L, rhostar, Lstar, alpha):
    a = rhostar
    b = ((L/Lstar)**alpha)
    x = -L*(1/Lstar)
    c = np.exp(x)
    d = np.log(10)
    return a*b*c*d
# fgas #########################################################################
def fgas(Mgas, Mstar):
    fgas = Mgas - Mstar
    return fgas

# sSFR #########################################################################
def sSFR(SFRs, Mgals):
    sSFR = []
    for i in range(0, len(SFRs)):
        a = np.log10(SFRs[i]) - Mgals[i]
        sSFR.append(a)
    return sSFR

# sSFR #########################################################################
def FindIndex(element, somelist):
    j = False
    for i, value in enumerate(somelist):
        if value == element:
            j = i
    return j

# quarter round ################################################################
def quarterRound(num, L):
    if L == True:
        ans = round((num*4)-0.5)/4
    else:
        ans = round((num*4)+0.5)/4
    return ans

# Omega H2 ################################################################
def OmegaH2(bins, yrho):
    rhocrit = 9.2*(10**(-27))
    dMH2 = bins[1] - bins[0]
    rhoH2 = (np.sum((10**yrho)*dMH2)*(2*(10**30)))/((3.086*(10**22))**3)
    OmegaH2 = (rhoH2/rhocrit)*(10000)
    return OmegaH2

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
################################################################################
def errors2(data, x, y):
    output = {'Vm':8, 'LCO':10}
    frac = 0.5
    eridx = int(len(data)*frac)
    idx = np.linspace(0,len(data)-1,len(data))
    spread = np.zeros((eridx, len(x)-1))
    for i in range(0, eridx):
        random.shuffle(idx)
        idx1 = idx[:eridx]
        newdata = np.zeros((eridx, np.shape(data)[1]))
        for j in range(0,len(newdata)):
            newdata[j,:] = data[int(idx[j]),:]
        newdata[:,output['Vm']] = newdata[:,output['Vm']]*frac
        totSch = Schechter(newdata, output['LCO'], output['Vm'], x)
        drho = totSch[1] - y
        spread[i,:] = drho
    er = spread
    sigma = []
    for i in range(0, np.shape(er)[1]):
        eri = er[:,i]
        eri = eri[abs(eri)<10]
        if np.std(eri)>0.00000001:
            sigma.append(np.std(eri))
        else:
            sigma.append(0)
    return sigma

# schechter only ###############################################################
def PlotSchechter(LSch, HSch, NDSch, totSch, xkeres, ykeres2, y_CG, sigma, y_ober, newdat, bins, LCOtot):
    xbins = np.linspace(7.5,10.5,300)
    LMassD, HMassD, ND = newdat
    tot = np.vstack((LMassD, HMassD))
    tot = np.vstack((tot,ND))
    # galactic
    a = np.zeros((len(LCOtot),1))
    for i in range(0,len(LCOtot)):
        a[i,0] = LCOtot[i,10]*LCOtot[i,16]
    LCOtot = np.hstack((LCOtot,a))
    totGal = Schechter(LCOtot, 24, 8,bins)
    Para_Gal = schechter.log_schechter_fit(totGal[2][5:-2], totGal[1][5:-2])
    y_Gal = schechter.log_schechter(xbins, *Para_Gal)
    print 'galactic', OmegaH2(totGal[2], totGal[1]+totGal[2]), OmegaH2(xbins, y_Gal+xbins)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #genzel
    totGen = Schechter(tot, 17, 10, bins)
    Para_Gen = schechter.log_schechter_fit(totGen[2][4:], totGen[1][4:])
    y_Gen = schechter.log_schechter(xbins, *Para_Gen)
    print 'genzel', OmegaH2(totGen[2], totGen[1]+totGen[2]), OmegaH2(xbins, y_Gen+xbins)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #schruba
    totSr = Schechter(tot, 18, 10, bins)
    Para_Sr = schechter.log_schechter_fit(totSr[2][6:], totSr[1][6:])
    y_Sr = schechter.log_schechter(xbins, *Para_Sr)
    print 'Sr', OmegaH2(totSr[2], totSr[1]+totSr[2]), OmegaH2(xbins, y_Sr+xbins)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #gio
    totGio = Schechter(tot, 19, 10, bins)
    Para_Gio = schechter.log_schechter_fit(totGio[2][4:-2], totGio[1][4:-2])
    y_Gio = schechter.log_schechter(xbins, *Para_Gio)
    print 'giio', OmegaH2(totGio[2], totGio[1]+totGio[2]), OmegaH2(xbins, y_Gio+xbins)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ax[0,0].plot(xbins, y_Gen, 'r-')
    # ax[0,0].plot(xbins, y_Sr, 'g-')
    # ax[0,0].plot(xbins, y_Gio, 'b-')
    # ax[0,0].plot(xbins, y_Gal, 'm-')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ax[0,0].scatter(LSch[2], LSch[1], marker = 's', s = 100, edgecolor='blue', linewidth='3', facecolor='none', label = 'Low Mass')
    # ax[0,0].scatter(HSch[2], HSch[1], marker = 's', s = 100, edgecolor='g', linewidth='3', facecolor='none', label = 'High Mass')
    # ax[0,0].scatter(NDSch[2], NDSch[1], marker = 's', s = 100, edgecolor='orange', linewidth='3', facecolor='none', label = 'Non Detection')
    # ax[0,0].errorbar(totSch[2], totSch[1], yerr=sigma, fmt='h', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='red', mec='crimson', ecolor='crimson', label = 'Total')
    ax[0,0].errorbar(totGen[2], totGen[1], fmt='h', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='r', mec='crimson', ecolor='crimson', alpha=0.5,  label = 'Genzel+12')
    ax[0,0].errorbar(totSr[2], totSr[1], fmt='o', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='limegreen', mec='g', ecolor='g', alpha=0.5, label = 'Schruba+12')
    ax[0,0].errorbar(totGio[2], totGio[1], fmt='s', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='b', mec='navy', ecolor='navy', alpha=0.5, label = 'Accurso+16')
    ax[0,0].errorbar(totGal[2], totGal[1], fmt='^', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='m', mec='m', ecolor='m', alpha=0.5, label = 'Galactic')
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax[0,0].plot(x_keres, y_ober, 'k--', label = 'Obreschkow+09')
    ax[0,0].plot(xkeres, ykeres2, 'k-', label = 'Keres+03')
    # ax[0,0].plot(xkeres, y_CG, linestyle = '-', color = 'crimson', label = 'COLD GASS fit')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{\odot}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.5)
    ax[0,0].tick_params(axis='x',which='minor',bottom='on')
    plt.legend(fontsize = 13, loc=3)
    plt.savefig('img/schechter/MH2.pdf', format='pdf', dpi=250, transparent = False)

def prepGraph(data, fracL, fracU, Vmaxis, aaxis, Laxis, mh2scatter):
    a = np.zeros((len(data),6))
    a[:,0] = mh2scatter/np.log10(data[:,aaxis]) # 28 frac from alpha
    print 'average from measure...............', np.average(data[:,fracL])
    print 'average from alpha.................', np.average(a[:,0])
    # print a[:,0]
    a[:,1] = np.sqrt((data[:,fracL]**2)+(a[:,0]**2))  #29 total frac error
    a[:,2] = np.sqrt((data[:,fracU]**2)+(a[:,0]**2))  #30 total frac error
    # print a[:,1]
    print 'tot error..........................', np.average(a[:,1])
    a[:,3] = data[:,Laxis]*data[:,aaxis]  #31     mean mass
    a[:,4] = a[:,3] - (a[:,1]*a[:,3])  #32     low mass
    a[:,5] = a[:,3] + (a[:,2]*a[:,3])  #33     highmass  mass
    data = np.hstack((data,a))
    return data
################################################################################
def PlotSchechter3(LSch, HSch, NDSch, totSch, xkeres, ykeres2, y_ober, bins, LCOtot, y_CG):
    xbins = np.linspace(7.5,10.5,300)
    ND = LCOtot[LCOtot[:,1]==2]
    D = LCOtot[LCOtot[:,1]==1]
    LMass = D[D[:,17]==0]
    HMass = D[D[:,17]==1]
    LCOD = prepGraph(D, 21, 22, 8, 25, 10, 0.23)
    totGen2 = Schechter(LCOD, 31, 8, bins)
    totGenl2 = Schechter(LCOD, 32, 8, bins)
    totGenu2 = Schechter(LCOD, 33, 8, bins)
    a = np.zeros((len(totGen2[1]),3))
    a[:,0] = totGen2[1]
    a[:,1] = totGenl2[1]
    a[:,2] = totGenu2[1]
    np.savetxt('confirmplot.txt',a)
    LCOtot2 = prepGraph(LCOtot, 21, 22, 8, 25, 10, 0.23)
    # a = np.zeros((len(LCOtot),12))
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # a[:,0] = LCOtot[:,10]*LCOtot[:,25]  #28     Gen
    # a[:,1] = LCOtot[:,23]*LCOtot[:,25]  #29     genl
    # a[:,2] = LCOtot[:,24]*LCOtot[:,25]  #30     genu
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # a[:,3] = LCOtot[:,10]*LCOtot[:,26]  #31     sr
    # a[:,4] = LCOtot[:,23]*LCOtot[:,26]  #32     srl
    # a[:,5] = LCOtot[:,24]*LCOtot[:,26]  #33     sru
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # a[:,6] = LCOtot[:,10]*LCOtot[:,27]  #34     gio
    # a[:,7] = LCOtot[:,23]*LCOtot[:,27]  #35     giol
    # a[:,8] = LCOtot[:,24]*LCOtot[:,27]  #36     giou
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # a[:,9] = LCOtot[:,10]*LCOtot[:,16]  #37     gal
    # a[:,10] = LCOtot[:,23]*LCOtot[:,16] #38     gall
    # a[:,11] = LCOtot[:,24]*LCOtot[:,16] #39     galu
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LCOtot = np.hstack((LCOtot, a))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # totGen = Schechter(LCOtot, 28, 8, bins)
    # totGenl = Schechter(LCOtot, 29, 8, bins)
    # totGenu = Schechter(LCOtot, 30, 8, bins)
    totGen2 = Schechter(LCOtot2, 31, 8, bins)
    totGenl2 = Schechter(LCOtot2, 32, 8, bins)
    totGenu2 = Schechter(LCOtot2, 33, 8, bins)
    df = pd.read_csv('data/genzelERRS.csv')
    GenErr = df[['low', 'high']].values
    GenErrS = errors2(LCOtot2, bins, totGen2[1])
    for i in range(0,len(GenErr)):
        GenErr[i,0] = np.sqrt((GenErr[i,0]**2) + (GenErrS[i]**2))
        GenErr[i,1] = np.sqrt((GenErr[i,1]**2) + (GenErrS[i]**2))
    Det = LCOtot2[LCOtot2[:,1]==1]
    ND = LCOtot2[LCOtot2[:,1]==2]
    L = Det[Det[:,17]==0]
    H = Det[Det[:,17]==1]
    LSch = Schechter(L, 31, 8, bins)
    HSch = Schechter(H, 31, 8, bins)
    NDSch = Schechter(ND, 31, 8, bins)
    xbins = np.linspace(7.5,10.5,300)
    CG_para = schechter.log_schechter_fit(totGen2[2][4:], totGen2[1][4:])
    print '@@@@@@MMMMMgenzel para', CG_para
    y_CG = schechter.log_schechter(xbins, *CG_para)
    CG_parau = schechter.log_schechter_fit(totGen2[2][5:], totGen2[1][5:]+GenErr[:,1][5:])
    y_CGu = schechter.log_schechter(xbins, *CG_parau)
    CG_paral = schechter.log_schechter_fit(totGen2[2][4:-2], totGen2[1][4:-2]-GenErr[:,0][4:-2])
    y_CGl = schechter.log_schechter(xbins, *CG_paral)
    ans = OmegaH2(totGen2[2], totGen2[1]+totGen2[2])
    low = OmegaH2(totGen2[2], totGen2[1]-GenErr[:,0]+totGen2[2])
    high = OmegaH2(totGen2[2], totGen2[1]+GenErr[:,1]+totGen2[2])
    ansfit = OmegaH2(xbins, y_CG+xbins)
    ansfitL = OmegaH2(xbins, y_CGl+xbins)
    ansfitH = OmegaH2(xbins, y_CGu+xbins)
    print 'Omega H2 Genzel....................', round(ans,2)
    print 'min Omega..........................', -round(ans-low,2)
    print 'max Omega..........................', +round(high-ans,2)
    print 'Omega H2 Genzel fit ...............', round(ansfit,2)
    print 'min.................................', -round(ansfit-ansfitL,2)
    print 'max.................................', round(ansfitH-ansfit,2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # totSr = Schechter(LCOtot, 31, 8, bins)
    # totSrl = Schechter(LCOtot, 32, 8, bins)
    # totSru = Schechter(LCOtot, 33, 8, bins)
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # totGio = Schechter(LCOtot, 34, 8, bins)
    # totGiol = Schechter(LCOtot, 35, 8, bins)
    # totGiou = Schechter(LCOtot, 36, 8, bins)
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # totGal = Schechter(LCOtot, 37, 8, bins)
    # totGall = Schechter(LCOtot, 38, 8, bins)
    # totGalu = Schechter(LCOtot, 39, 8, bins)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ax[0,0].scatter(LSch[2], LSch[1], marker = 's', s = 100, edgecolor='blue', linewidth='3', facecolor='none', label = 'Low Mass')
    # ax[0,0].scatter(HSch[2], HSch[1], marker = 's', s = 100, edgecolor='g', linewidth='3', facecolor='none', label = 'High Mass')
    # ax[0,0].scatter(NDSch[2], NDSch[1], marker = 's', s = 100, edgecolor='orange', linewidth='3', facecolor='none', label = 'Non Detection')
    # ax[0,0].errorbar(totSch[2], totSch[1], yerr=sigma, fmt='h', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='red', mec='crimson', ecolor='crimson', label = 'Total')
    # ax[0,0].errorbar(totGen[2], totGen[1], fmt='h', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='b', mec='b', ecolor='crimson', alpha=0.5,  label = 'Genzel+12')
    # ax[0,0].errorbar(totGenl[2], totGenl[1], fmt='h', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='r', mec='r', ecolor='crimson', alpha=0.5,  label = 'Genzell+12')
    # ax[0,0].errorbar(totGenu[2], totGenu[1], fmt='h', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='g', mec='g', ecolor='crimson', alpha=0.5,  label = 'Genzelu+12')
    #ax[0,0].errorbar(totGenl2[2], totGenl2[1], fmt='^', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='r', mec='r', ecolor='crimson', alpha=0.5,  label = 'Genzell+12')
    # ax[0,0].errorbar(totGenu2[2], totGenu2[1], fmt='^', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='g', mec='g', ecolor='crimson', alpha=0.5,  label = 'Genzelu+12')
    # ax[0,0].errorbar(totSr[2], totSr[1], fmt='o', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='limegreen', mec='g', ecolor='g', alpha=0.5, label = 'Schruba+12')
    # ax[0,0].errorbar(totGio[2], totGio[1], fmt='s', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='b', mec='navy', ecolor='navy', alpha=0.5, label = 'Accurso+16')
    # ax[0,0].errorbar(totGal[2], totGal[1], fmt='^', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='m', mec='m', ecolor='m', alpha=0.5, label = 'Galactic')
    ax[0,0].plot(xbins, y_CG, linestyle = '-', color = 'crimson', label = 'COLD GASS Schechter fit', linewidth=1, zorder=3)
    ax[0,0].plot(x_keres, y_ober, 'k--', label = 'Obreschkow+09', linewidth=1, zorder=1)
    ax[0,0].plot(xkeres, ykeres2, 'k-', label = 'Keres+03', linewidth=1, zorder=2)
    # ax[0,0].plot(xbins, y_CGl, linestyle = '-', color = 'crimson')
    # ax[0,0].plot(xbins, y_CGu, linestyle = '-', color = 'crimson')
    a = np.zeros((len(totGen2[1]),4))
    a[:,0] = totGen2[2]
    a[:,1] = totGen2[1]
    a[:,2] = GenErr[:,0]
    a[:,3] = GenErr[:,1]
    np.savetxt('gzl.txt', a)
    ax[0,0].fill_between(xbins, y_CGl, y_CGu, color = 'r', alpha =0.1)
    ax[0,0].errorbar(totGen2[2], totGen2[1], yerr = [GenErr[:,0], GenErr[:,1]], fmt='h', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='r', mec='crimson', ecolor='crimson', zorder = 7, label = 'COLD GASS + Genzel+12')
    ### ax[0,0].scatter(LSch[2], LSch[1], marker = 's', s = 100, edgecolor='blue', linewidth='3', facecolor='none', label = 'Low Mass Detections', zorder=6)
    # ax[0,0].scatter(HSch[2], HSch[1], marker = 's', s = 100, edgecolor='g', linewidth='3', facecolor='none', label = 'High Mass Detections', zorder=5)
    # ax[0,0].scatter(NDSch[2], NDSch[1], marker = 's', s = 100, edgecolor='orange', linewidth='3', facecolor='none', label = 'Non Detections', zorder=4)
    # ax[0,0].plot(xkeres, y_CG, linestyle = '-', color = 'crimson', label = 'COLD GASS fit')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{\odot}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.75)
    b = np.zeros((len(xbins),3))
    b[:,0] = xbins
    b[:,1] = y_CGl
    b[:,2] = y_CGu
    np.savetxt('pptgraph.txt', b)
    ax[0,0].tick_params(axis='x',which='minor',bottom='on')
    plt.legend(fontsize = 13, loc=3)
    plt.tight_layout()
    plt.savefig('img/schechter/MH2abc.pdf', format='pdf', dpi=250, transparent = True)
################################################################################
def PlotGalactic(LSch, HSch, NDSch, totSch, xkeres, ykeres2, y_ober, bins, LCOtot, y_CG):
    genzel = np.loadtxt('pptgraph.txt')
    xbins = np.linspace(7.5,10.5,300)
    ND = LCOtot[LCOtot[:,1]==2]
    D = LCOtot[LCOtot[:,1]==1]
    LMass = D[D[:,17]==0]
    HMass = D[D[:,17]==1]
    LCOtot2 = prepGraph(LCOtot, 21, 22, 8, 16, 10, 0.9)
    totGen2 = Schechter(LCOtot2, 31, 8, bins)
    totGenl2 = Schechter(LCOtot2, 32, 8, bins)
    totGenu2 = Schechter(LCOtot2, 33, 8, bins)
    # a = np.zeros((len(totGen2[1]),3))
    # a[:,0] = totGen2[1]
    # a[:,1] = totGenl2[1]
    # a[:,2] = totGenu2[1]
    # np.savetxt('data/galacticERRS.txt',a)
    df = pd.read_csv('data/galacticERRS.csv')
    GenErr = df[['low', 'high']].values
    GenErrS = errors2(LCOtot2, bins, totGen2[1])
    for i in range(0,len(GenErr)):
        GenErr[i,0] = np.sqrt((GenErr[i,0]**2) + (GenErrS[i]**2))
        GenErr[i,1] = np.sqrt((GenErr[i,1]**2) + (GenErrS[i]**2))
    xbins = np.linspace(7.5,10.5,300)
    CG_para = schechter.log_schechter_fit(totGen2[2][2:], totGen2[1][2:])
    print 'gal para', CG_para
    y_CG = schechter.log_schechter(xbins, *CG_para)
    CG_parau = schechter.log_schechter_fit(totGen2[2][2:], totGen2[1][2:]+GenErr[:,1][2:])
    y_CGu = schechter.log_schechter(xbins, *CG_parau)
    CG_paral = schechter.log_schechter_fit(totGen2[2][2:-1], totGen2[1][2:-1]-GenErr[:,0][2:-1])
    y_CGl = schechter.log_schechter(xbins, *CG_paral)
    ans = OmegaH2(totGen2[2], totGen2[1]+totGen2[2])
    low = OmegaH2(totGen2[2], totGen2[1]-GenErr[:,0]+totGen2[2])
    high = OmegaH2(totGen2[2], totGen2[1]+GenErr[:,1]+totGen2[2])
    ansfit = OmegaH2(xbins, y_CG+xbins)
    ansfitL = OmegaH2(xbins, y_CGl+xbins)
    ansfitH = OmegaH2(xbins, y_CGu+xbins)
    print 'Omega H2 Gal....................', round(ans,2)
    print 'min Omega..........................', -round(ans-low,2)
    print 'max Omega..........................', +round(high-ans,2)
    print 'Omega H2 Gal fit ...............', round(ansfit,2)
    print 'min.................................', -round(ansfit-ansfitL,2)
    print 'max.................................', round(ansfitH-ansfit,2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax[0,0].plot(xbins, y_CG, linestyle = '-', color = 'g', label = 'COLD GASS Schechter fit', linewidth=1, zorder=3)
    ax[0,0].plot(x_keres, y_ober, 'k--', label = 'Obreschkow+09', linewidth=1, zorder=1)
    ax[0,0].plot(xkeres, ykeres2, 'k-', label = 'Keres+03', linewidth=1, zorder=2)
    # ax[0,0].plot(xbins, y_CGl, linestyle = '-', color = 'crimson')
    # ax[0,0].plot(xbins, y_CGu, linestyle = '-', color = 'crimson')
    ax[0,0].fill_between(xbins, y_CGl, y_CGu, color = 'limegreen', alpha =0.4)
    ax[0,0].fill_between(genzel[:,0], genzel[:,1], genzel[:,2], color = 'red', alpha =0.1)
    ax[0,0].errorbar(totGen2[2], totGen2[1], yerr=[GenErr[:,0], GenErr[:,1]], fmt='h', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='limegreen', mec='g', ecolor='g', zorder=7,label = 'COLD GASS + Galactic')
    # ax[0,0].errorbar(totGenl2[2], totGenl2[1], fmt='s', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='r', mec='crimson', ecolor='crimson', zorder=7,label = 'COLD GASS + Genzel+12')
    # ax[0,0].errorbar(totGenu2[2], totGenu2[1], fmt='^', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='r', mec='crimson', ecolor='crimson', zorder=7,label = 'COLD GASS + Genzel+12')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    b = np.zeros((len(xbins),3))
    b[:,0] = xbins
    b[:,1] = y_CGl
    b[:,2] = y_CGu
    np.savetxt('pptgraphGal.txt', b)
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{\odot}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.75)
    ax[0,0].tick_params(axis='x',which='minor',bottom='on')
    plt.legend(fontsize = 13, loc=3)
    plt.tight_layout()
    plt.savefig('img/schechter/MH2Gal.pdf', format='pdf', dpi=250, transparent = True)
################################################################################
def PlotSchruba(LSch, HSch, NDSch, totSch, xkeres, ykeres2, y_ober, bins, LCOtot, y_CG):
    gal = np.loadtxt('pptgraphGal.txt')
    genzel = np.loadtxt('pptgraph.txt')
    xbins = np.linspace(7.5,10.5,300)
    ND = LCOtot[LCOtot[:,1]==2]
    D = LCOtot[LCOtot[:,1]==1]
    LMass = D[D[:,17]==0]
    HMass = D[D[:,17]==1]
    LCOtot2 = prepGraph(LCOtot, 21, 22, 8, 26, 10, 0.12)
    totGen2 = Schechter(LCOtot2, 31, 8, bins)
    totGenl2 = Schechter(LCOtot2, 32, 8, bins)
    totGenu2 = Schechter(LCOtot2, 33, 8, bins)
    # a = np.zeros((len(totGen2[1]),3))
    # a[:,0] = totGen2[1]
    # a[:,1] = totGenl2[1]
    # a[:,2] = totGenu2[1]
    # np.savetxt('data/schrubaERRS.txt',a)
    df = pd.read_csv('data/schrubaERRS.csv')
    GenErr = df[['low', 'high']].values
    GenErrS = errors2(LCOtot2, bins, totGen2[1])
    for i in range(0,len(GenErr)):
        GenErr[i,0] = np.sqrt((GenErr[i,0]**2) + (GenErrS[i]**2))
        GenErr[i,1] = np.sqrt((GenErr[i,1]**2) + (GenErrS[i]**2))
    xbins = np.linspace(7.5,11,300)
    CG_para = schechter.log_schechter_fit(totGen2[2][7:], totGen2[1][7:])
    print 'sr para', CG_para
    y_CG = schechter.log_schechter(xbins, *CG_para)
    CG_parau = schechter.log_schechter_fit(totGen2[2][7:], totGen2[1][7:]+GenErr[:,1][7:])
    y_CGu = schechter.log_schechter(xbins, *CG_parau)
    CG_paral = schechter.log_schechter_fit(totGen2[2][6:], totGen2[1][6:]-GenErr[:,0][6:])
    y_CGl = schechter.log_schechter(xbins, *CG_paral)
    ans = OmegaH2(totGen2[2], totGen2[1]+totGen2[2])
    low = OmegaH2(totGen2[2], totGen2[1]-GenErr[:,0]+totGen2[2])
    high = OmegaH2(totGen2[2], totGen2[1]+GenErr[:,1]+totGen2[2])
    ansfit = OmegaH2(xbins, y_CG+xbins)
    ansfitL = OmegaH2(xbins, y_CGl+xbins)
    ansfitH = OmegaH2(xbins, y_CGu+xbins)
    print 'Omega H2 sr....................', round(ans,2)
    print 'min Omega..........................', -round(ans-low,2)
    print 'max Omega..........................', +round(high-ans,2)
    print 'Omega H2 sr fit ...............', round(ansfit,2)
    print 'min.................................', -round(ansfit-ansfitL,2)
    print 'max.................................', round(ansfitH-ansfit,2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax[0,0].plot(xbins, y_CG, linestyle = '-', color = 'navy', label = 'COLD GASS Schechter fit', linewidth=1, zorder=3)
    ax[0,0].plot(x_keres, y_ober, 'k--', label = 'Obreschkow+09', linewidth=1, zorder=1)
    ax[0,0].plot(xkeres, ykeres2, 'k-', label = 'Keres+03', linewidth=1, zorder=2)
    # ax[0,0].plot(xbins, y_CGl, linestyle = '-', color = 'crimson')
    # ax[0,0].plot(xbins, y_CGu, linestyle = '-', color = 'crimson')
    ax[0,0].fill_between(xbins, y_CGl, y_CGu, color = 'b', alpha =0.4)
    ax[0,0].fill_between(gal[:,0], gal[:,1], gal[:,2], color = 'limegreen', alpha =0.1)
    ax[0,0].fill_between(genzel[:,0], genzel[:,1], genzel[:,2], color = 'red', alpha =0.1)
    ax[0,0].errorbar(totGen2[2], totGen2[1], yerr=[GenErr[:,0], GenErr[:,1]], fmt='h', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='b', mec='navy', ecolor='navy', zorder=7,label = 'COLD GASS + Schruba+12')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    b = np.zeros((len(xbins),3))
    b[:,0] = xbins
    b[:,1] = y_CGl
    b[:,2] = y_CGu
    np.savetxt('pptgraphSr.txt', b)
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{\odot}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.75)
    ax[0,0].tick_params(axis='x',which='minor',bottom='on')
    plt.legend(fontsize = 13, loc=3)
    plt.tight_layout()
    plt.savefig('img/schechter/MH2Sr.pdf', format='pdf', dpi=250, transparent = True)
################################################################################
def PlotAccurso(LSch, HSch, NDSch, totSch, xkeres, ykeres2, y_ober, bins, LCOtot, y_CG):
    gal = np.loadtxt('pptgraphGal.txt')
    genzel = np.loadtxt('pptgraph.txt')
    sr = np.loadtxt('pptgraphSr.txt')
    xbins = np.linspace(7.5,10.5,300)
    ND = LCOtot[LCOtot[:,1]==2]
    D = LCOtot[LCOtot[:,1]==1]
    LMass = D[D[:,17]==0]
    HMass = D[D[:,17]==1]
    LCOtot2 = prepGraph(LCOtot, 21, 22, 8, 27, 10, 0.1)
    totGen2 = Schechter(LCOtot2, 31, 8, bins)
    totGenl2 = Schechter(LCOtot2, 32, 8, bins)
    totGenu2 = Schechter(LCOtot2, 33, 8, bins)
    df = pd.read_csv('data/AccursoERRS.csv')
    GenErr = df[['low', 'high']].values
    GenErrS = errors2(LCOtot2, bins, totGen2[1])
    for i in range(0,len(GenErr)):
        GenErr[i,0] = np.sqrt((GenErr[i,0]**2) + (GenErrS[i]**2))
        GenErr[i,1] = np.sqrt((GenErr[i,1]**2) + (GenErrS[i]**2))
    xbins = np.linspace(7.5,11,300)
    CG_para = schechter.log_schechter_fit(totGen2[2][4:-2], totGen2[1][4:-2])
    print 'accurso para', CG_para
    y_CG = schechter.log_schechter(xbins, *CG_para)
    CG_parau = schechter.log_schechter_fit(totGen2[2][5:-1], totGen2[1][5:-1]+GenErr[:,1][5:-1])
    y_CGu = schechter.log_schechter(xbins, *CG_parau)
    CG_paral = schechter.log_schechter_fit(totGen2[2][3:-4], totGen2[1][3:-4]-GenErr[:,0][3:-4])
    y_CGl = schechter.log_schechter(xbins, *CG_paral)
    ans = OmegaH2(totGen2[2], totGen2[1]+totGen2[2])
    low = OmegaH2(totGen2[2], totGen2[1]-GenErr[:,0]+totGen2[2])
    high = OmegaH2(totGen2[2], totGen2[1]+GenErr[:,1]+totGen2[2])
    ansfit = OmegaH2(xbins, y_CG+xbins)
    ansfitL = OmegaH2(xbins, y_CGl+xbins)
    ansfitH = OmegaH2(xbins, y_CGu+xbins)
    print 'Omega H2 Accurso....................', round(ans,2)
    print 'min Omega..........................', -round(ans-low,2)
    print 'max Omega..........................', +round(high-ans,2)
    print 'Omega H2 Accurso fit ...............', round(ansfit,2)
    print 'min.................................', -round(ansfit-ansfitL,2)
    print 'max.................................', round(ansfitH-ansfit,2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax[0,0].plot(xbins, y_CG, linestyle = '-', color = 'm', label = 'COLD GASS Schechter fit', linewidth=1, zorder=3)
    ax[0,0].plot(x_keres, y_ober, 'k--', label = 'Obreschkow+09', linewidth=1, zorder=1)
    ax[0,0].plot(xkeres, ykeres2, 'k-', label = 'Keres+03', linewidth=1, zorder=2)
    # ax[0,0].plot(xbins, y_CGl, linestyle = '-', color = 'crimson')
    # ax[0,0].plot(xbins, y_CGu, linestyle = '-', color = 'crimson')
    ax[0,0].fill_between(xbins, y_CGl, y_CGu, color = 'deeppink', alpha =0.1)
    # ax[0,0].fill_between(gal[:,0], gal[:,1], gal[:,2], color = 'limegreen', alpha =0.1)
    # ax[0,0].fill_between(genzel[:,0], genzel[:,1], genzel[:,2], color = 'red', alpha =0.1)
    # ax[0,0].fill_between(sr[:,0], sr[:,1], sr[:,2], color = 'b', alpha =0.1)
    a = np.zeros((len(totGen2[1]),4))
    a[:,0] = totGen2[2]
    a[:,1] = totGen2[1]
    a[:,2] = GenErr[:,0]
    a[:,3] = GenErr[:,1]
    print (a)
    np.savetxt('data/AccursoERRS.txt',a)
    # ax[0,0].errorbar(totGen2[2], totGen2[1], yerr=[GenErr[:,0], GenErr[:,1]], fmt='h', markersize = 10, linewidth=2, mew=2, capthick=3, mfc='deeppink', mec='m', ecolor='m', zorder=7,label = 'COLD GASS + Accurso+12')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{\odot}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.75)
    ax[0,0].tick_params(axis='x',which='minor',bottom='on')
    plt.legend(fontsize = 13, loc=3)
    plt.tight_layout()
    plt.savefig('img/schechter/MH2Accurso.pdf', format='pdf', dpi=250, transparent = True)
################################################################################
def PlotSchechter2(totSch, sigmatot, y_CG, detSch, sigmadet, y_det, xkeres, ykeres2):
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].errorbar(totSch[2], totSch[1], yerr=sigmatot, fmt = 'o', markersize = 10, color = 'red', label = 'COLD GASS ND+D')
    ax[0,0].errorbar(detSch[2], detSch[1], yerr=sigmadet, fmt = 's', markersize = 10, color = 'blue', label = 'COLD GASS D')
    # ax[0,0].scatter(detSch[2], detSch[1], yerr=sigmadet, marker = 's', s = 100, edgecolor='blue', linewidth='2', facecolor='none', label = 'Detections')
    ax[0,0].fill_between(xkeres, y_CG, y_det, alpha = 0.5, color = 'green')
    ax[0,0].plot(xkeres, ykeres2, 'k--', label = 'Keres+03')
    ax[0,0].plot(xkeres, y_CG, 'r-', label = 'COLD GASS ND+D')
    ax[0,0].plot(xkeres, y_det, 'b-', label = 'COLD GASS D')
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.5)
    ax[0,0].tick_params(axis='x',which='minor',bottom='on')
    #ax[0,1].set_title('Schechter', fontsize=20)
    # ax[0,0].text(9, -5.1, (r'$\phi_{*}$ = '+str(round(phi1,2))+'\n'+ r'$L_{*}$ = '+str(round(L01,2))+'\n'+ r'$\alpha$ = '+str(round(alpha1,2))), fontsize=18, color='b')
    # ax[0,0].text(9, -5.8, (r'$\phi_{*}$ = '+str(round(phi2,2))+'\n'+ r'$L_{*}$ = '+str(round(L02,2))+'\n'+ r'$\alpha$ = '+str(round(alpha2,2))), fontsize=18, color='r')
    plt.legend(fontsize = 12)
    plt.savefig('img/schechter/MH2poster.pdf', format='pdf', dpi=250, transparent = False)
    # plt.savefig('img/MH2.png', transparent = False ,dpi=250)

# schechter only ###############################################################
def PlotRhoH2(LSch, HSch, NDSch, totSch, x, x1, ykeresph2, yrhoCG, yrhoCGpts, yrhoCG2, yrhokeres, x_keres):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(LSch[2], LSch[4], marker = 's', s = 100, edgecolor='blue', linewidth='2', facecolor='none', label = 'Low Mass')
    ax[0,0].scatter(HSch[2], HSch[4], marker = 's', s = 100, edgecolor='green', linewidth='2', facecolor='none', label = 'High Mass')
    ax[0,0].scatter(NDSch[2], NDSch[4], marker = 's', s = 100, edgecolor='orange', linewidth='2', facecolor='none', label = 'Non Detection')
    ax[0,0].scatter(x_keres, yrhoCG2, marker = 'o', s = 100, edgecolor='black', linewidth='2', facecolor='none', label = 'pts')
    ax[0,0].scatter(x_keres, yrhokeres, marker = 'o', s = 100, edgecolor='black', linewidth='2', facecolor='none', label = 'pts')
    ax[0,0].errorbar(totSch[2], totSch[4], fmt = 'o', markersize = 10, color = 'red', label = 'Total')

    ax[0,0].plot(x1,ykeresph2, 'k--')
    ax[0,0].plot(x1,yrhoCG, 'k-')
    #ax[0,0].plot(xkeres, ykeres2, 'k--', label = 'Keres+03')
    #ax[0,0].plot(xkeres, y_CG, 'k-', label = 'COLD GASS fit')
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \rho_{H2}\, [M_{\odot}\, Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(4, 7.5)
    ax[0,0].set_xlim(7, 11)
    ax[0,0].tick_params(axis='x',which='minor',bottom='on')
    #ax[0,1].set_title('Schechter', fontsize=20)
    # ax[0,0].text(9, -5.1, (r'$\phi_{*}$ = '+str(round(phi1,2))+'\n'+ r'$L_{*}$ = '+str(round(L01,2))+'\n'+ r'$\alpha$ = '+str(round(alpha1,2))), fontsize=18, color='b')
    # ax[0,0].text(9, -5.8, (r'$\phi_{*}$ = '+str(round(phi2,2))+'\n'+ r'$L_{*}$ = '+str(round(L02,2))+'\n'+ r'$\alpha$ = '+str(round(alpha2,2))), fontsize=18, color='r')
    #plt.legend(fontsize = 13)
    plt.savefig('img/schechter/pH2.pdf', format='pdf', dpi=250, transparent = False)
    # plt.savefig('img/MH2.png', transparent = False ,dpi=250)

# schechter only ###############################################################
def PlotAlphaCO(data, output):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(np.log10(data[:,output['M*']]), data[:, output['AlphaCO']], marker = 'o', s = 1, label = 'Low Mass')
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{\alpha_{CO}}$', fontsize=18)
    #ax[0,0].set_ylim(-5, -1)
    #ax[0,0].set_xlim(7.5, 10.5)
    ax[0,0].tick_params(axis='x',which='minor',bottom='on')
    plt.savefig('img/schechter/aCO.pdf', format='pdf', dpi=250, transparent = False)
# schechter only ###############################################################
def Plotweights(data):
    x = np.linspace(9,12,2)
    y = [1,1]
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter((data[:,4]), data[:,13], marker = 'o', s = 1,)
    ax[0,0].plot(x,y)
    # ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
    # ax[0,0].set_ylabel(r'$\mathrm{\alpha_{CO}}$', fontsize=18)
    #ax[0,0].set_ylim(-5, -1)
    #ax[0,0].set_xlim(7.5, 10.5)
    # ax[0,0].hlines(10 ,12, 1, color='k')
    ax[0,0].tick_params(axis='x',which='minor',bottom='on')
    plt.savefig('img/schechter/weights.pdf', format='pdf', dpi=250, transparent = False)
# schechter only ###############################################################
def PlotMsunvsMH2(data, output):
    x = np.linspace(8.5,12,200)
    y = np.zeros((200,1))
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].scatter(np.log10(data[:,output['MH2']]), np.log10(data[:, output['MH2']])-data[:,output['M*']], marker = 'o', s = 1, label = 'Low Mass')
    # ax[0,0].plot(x,0)
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \frac{M_{H2}}{M_{sun}}}$', fontsize=18)
    plt.savefig('img/schechter/MMH2.pdf', format='pdf', dpi=250, transparent = False)
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
            newdata[j,:] = data[int(idx[j]),:]
        newdata[:,output['Vm']] = newdata[:,output['Vm']]*frac
        totSch = Schechter(newdata, output['MH2'], output['Vm'], x)
        drho = totSch[1] - y
        spread[i,:] = drho
    return spread
################################################################################
def PlotMstarMH2(total, FullData, ND, FullND, output, FullDatasim):
    det = FullDatasim[FullDatasim[:,1]==1]
    nondet = FullDatasim[FullDatasim[:,1]==2]
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    # ax[0,0].scatter(total[:,output['M*']], np.log10(total[:,output['MH2']]), color = 'r', label = 'Mine', s=10)
    ax[0,0].scatter(FullData[:,6], np.log10(FullData[:,4]), color = 'g', label = 'COLD GASS D', s=10)
    ax[0,0].scatter(FullND[:,6], np.log10(FullND[:,4]), color = 'crimson', label = 'COLD GASS ND', s=10)
    # ax[0,0].scatter(ND[:,output['M*']], np.log10(ND[:,output['MH2']]), color = 'g', label = 'My ND', s=10)
    ax[0,0].scatter(det[:,4], det[:,9], color = 'k', label = 'Scaling Relation:'+r'$\mathrm{SFR_{Best}}$' + '+'+ r'$f_{\mathrm{H_2}}$'+' ,D', s=10)
    ax[0,0].scatter(nondet[:,4], nondet[:,9], color = 'b', label = 'Scaling Relation:'+r'$\mathrm{SFR_{Best}}$' + '+'+ r'$f_{\mathrm{H_2}}$'+' ,ND', s=10)
    ax[0,0].vlines(10,7.5,10.5, color='k')
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{*}\,[M_{\odot}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, M_{H2}\,[M_{\odot}]}$', fontsize=18)
    ax[0,0].set_ylim(7.5, 10.5)
    ax[0,0].set_xlim(9, 11.5)
    plt.legend(fontsize = 13, loc=2)
    plt.tight_layout()
    plt.savefig('img/schechter/MstarvsMH2.pdf', dpi=250, transparent = False)
# Plot Schechter from Full dataset #############################################
def PlotSchechterFull(FullDetSchech, FullNDSchech, FullSchech, x_keres, y_keres, y_ober):
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].errorbar(FullDetSchech[2], FullDetSchech[1], fmt = 'o', markersize = 10, color = 'red', label = 'Detections')
    ax[0,0].errorbar(FullNDSchech[2], FullNDSchech[1], fmt = 's', markersize = 10, color = 'blue', label = 'Non-Detections')
    ax[0,0].errorbar(FullSchech[2], FullSchech[1], fmt = 's', markersize = 10, color = 'green', label = 'Both')
    ax[0,0].plot(x_keres, y_keres, 'k--', label = 'Keres+03')
    ax[0,0].plot(x_keres, y_ober, 'k-', label = 'Obreschkow+09')
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.5)
    ax[0,0].tick_params(axis='x',which='minor',bottom='on')
    plt.legend(fontsize = 10)
    plt.savefig('img/schechter/MH2_FULL.pdf', format='pdf', dpi=250, transparent = False)
################################################################################
def comparefull(Full, compare):
    data = np.zeros((len(Full), len(compare)))
    for i,rows in enumerate(Full):
        data[i,0] = rows[compare['ID']]
        data[i,1] = rows[compare['flag']]
        data[i,2] = rows[compare['M*']]
        data[i,3] = rows[compare['AlphaCO']]
        data[i,4] = rows[compare['D_L']]
        data[i,5] = rows[compare['L_CO']]
        data[i,6] = rows[compare['MH2']]
        data[i,7] = rows[compare['limMH2']]
    return data
################################################################################
def compareIDforID(Full, total, output, compareoutput):
    # totalH = total[total[:,output['M*']] > 10.0]
    # totalHND = totalH[totalH[:,output['flag']] == 2]
    # FullH = Full[Full[:,compareoutput['M*']] > 10.0]
    # FullHND = FullH[FullH[:,compareoutput['flag']] == 2]
    Full = Full[Full[:,0].argsort()]
    total = total[total[:,0].argsort()]
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(total[:,output['D_L']], (total[:,output['D_L']]-Full[:,compareoutput['D_L']]), color = 'r', label = 'Mine', s=10)
    x = np.linspace(-4,20,200)
    y = np.zeros((len(x),1))
    ax[0,0].plot(x,y)
    ax[0,0].set_xlim(-5, 20)
    ax[0,0].set_ylim(-10, 12)
    ax[0,0].set_xlabel(r'$\mathrm{my\, \alpha_{CO}}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{my\, \alpha_{CO}\, - \,G12 \,from\, full}$', fontsize=18)
    plt.savefig('img/schechter/COMPAREID.pdf', format='pdf', dpi=250, transparent = False)
################################################################################
def GetFull(Full, output):
    df = pd.read_csv('data/XCO.csv')
    XCOH = df[['ID', 'XCO']].values
    FullData = np.zeros((len(Full),20))
    lorh = []
    lorh2 = []
    for j,rows1 in enumerate(Full):
        lorh.append(rows1[1])
        if rows1[1] == 'L':
            lorh2.append(0)
        elif rows1[1] == 'H':
            lorh2.append(1)
    for i,rows in enumerate(Full):
        #z|flag|MH2|limMH2|MH2_both|Lumdist|M*|
        FullData[i,output['ID']] = rows[0]
        FullData[i,output['S_CO']] = rows[47]
        FullData[i,output['z']] = rows[9]
        FullData[i,output['flag']] = rows[37]
        FullData[i,output['M*']] = rows[20]
        FullData[i,output['Zo']] = 0
        FullData[i,output['SFR']] = np.log10(rows[24]) #sfr_best
        FullData[i,output['sSFR']] = FullData[i,output['SFR']] - FullData[i,output['M*']]
        FullData[i,output['NUV-r']] = 0
        FullData[i,output['D_L']] = rows[10]
        FullData[i,10] = rows[51] # log MH2
        FullData[i,11] = rows[52] # lim log MH2 3 sig
        FullData[i,12] = rows[52] + rows[51]
        FullData[i,13] = rows[64] #weighting
        FullData[i,14] = rows[44] # LCO
        FullData[i,15] = rows[46] # rms_CO
        FullData[i,16] = rows[48] # W_CO
        FullData[i,17] = rows[35] # Metallicity
        FullData[i,18] = rows[39] # genzel
        FullData[i,19] = 4.35 #alphaCO GAL
    data = pd.DataFrame({   'group':lorh, 'groupno':lorh2, 'ID': FullData[:,output['ID']], 'S_CO': FullData[:,output['S_CO']], 'z': FullData[:,output['z']],
                            'flag': FullData[:,output['flag']], 'M*': FullData[:,output['M*']], 'Zo': FullData[:,output['Zo']], 'SFR': FullData[:,output['SFR']],
                            'sSFR': FullData[:,output['sSFR']],'NUV-r': FullData[:,output['NUV-r']],'D_L': FullData[:,output['D_L']],
                            'MH2': FullData[:,10], 'limMH2': FullData[:,11], 'MH2both': FullData[:,12], 'Weight':FullData[:,13],
                            'LCO':FullData[:,14], 'rms':FullData[:,15], 'WCO':FullData[:,16], 'Met':FullData[:,17],
                            'aCO':FullData[:,18], 'aCO_Gal':FullData[:,19]})
    LMass = (data.loc[data['group'] == 'L'])
    HMass = (data.loc[data['group'] == 'H'])
    Lz, Hz = LMass[['z']].values, HMass[['z']].values
    LMassND = (LMass.loc[LMass['flag'] == 2])
    HMassND = (HMass.loc[HMass['flag'] == 2])
    LMass1 = (LMass.loc[LMass['flag'] == 1])
    HMass1 = (HMass.loc[HMass['flag'] == 1])
    LMassNDarr = LMassND[['ID', 'S_CO', 'z', 'flag', 'M*', 'Zo', 'SFR', 'sSFR', 'NUV-r', 'D_L']].values
    HMassNDarr = HMassND[['ID', 'S_CO', 'z', 'flag', 'M*', 'Zo', 'SFR', 'sSFR', 'NUV-r', 'D_L']].values
    LMassNDarr = Vm(LMassNDarr, output['D_L'], min(Lz), max(Lz), 1)
    HMassNDarr = Vm(HMassNDarr, output['D_L'], min(Hz), max(Hz), 2)
    LMassND['L_CO'], LMassND['Vm'], LMassND['V/Vm'], LMassND['AlphaCO'], LMassND['dalpha'] = np.zeros((len(LMassND),1)), LMassNDarr[:,output['Vm']], LMassNDarr[:,output['V/Vm']], np.zeros((len(LMassND),1)), np.zeros((len(LMassND),1))
    HMassND['L_CO'], HMassND['Vm'], HMassND['V/Vm'], HMassND['AlphaCO'], HMassND['dalpha'] = np.zeros((len(HMassND),1)), HMassNDarr[:,output['Vm']], HMassNDarr[:,output['V/Vm']], np.zeros((len(HMassND),1)), np.zeros((len(HMassND),1))
    # z|1:flag|2:MH2|3:limMH2|4:MH2_both|5:Lumdist|6:M*|7:V/Vm|8:Vm
    LMass['Vm'], LMass['V/Vm'] = np.full((len(LMass),1), LMassNDarr[0,output['Vm']]), np.zeros((len(LMass),1))
    HMass['Vm'], HMass['V/Vm'] = np.full((len(HMass),1), HMassNDarr[0,output['Vm']]), np.zeros((len(HMass),1))
    LMassFull = LMass[['z', 'flag', 'MH2', 'limMH2', 'MH2both', 'D_L', 'M*', 'V/Vm', 'Vm', 'Weight']].values
    HMassFull = HMass[['z', 'flag', 'MH2', 'limMH2', 'MH2both', 'D_L', 'M*', 'V/Vm', 'Vm', 'Weight']].values
    Fulldata = np.vstack((LMassFull, HMassFull))
    # '0 z, 1 flag, 2 MH2, 3 limMH2, 4 MH2both, 5 D_L, 6 M*, 7 V/Vm, 8 Vm, 9 Weight, 10 newVm'
    a = np.zeros((len(Fulldata),1))
    a[:,0] = Fulldata[:,8]/Fulldata[:,9]
    Fulldata = np.hstack((Fulldata, a))
    LMassNDarr = LMassND[['ID', 'S_CO', 'z', 'flag', 'M*', 'Zo', 'SFR', 'sSFR', 'NUV-r', 'D_L', 'V/Vm', 'Vm', 'L_CO', 'AlphaCO', 'limMH2', 'dalpha']].values
    HMassNDarr = HMassND[['ID', 'S_CO', 'z', 'flag', 'M*', 'Zo', 'SFR', 'sSFR', 'NUV-r', 'D_L', 'V/Vm', 'Vm', 'L_CO', 'AlphaCO', 'limMH2', 'dalpha']].values
    ############
    LMassFull2 = LMass[['z', 'flag', 'MH2', 'limMH2', 'MH2both', 'D_L', 'M*', 'V/Vm', 'Vm', 'Weight', 'LCO', 'rms', 'WCO', 'S_CO', 'sSFR','ID', 'aCO_Gal', 'groupno', 'Met']].values
    HMassFull2 = HMass[['z', 'flag', 'MH2', 'limMH2', 'MH2both', 'D_L', 'M*', 'V/Vm', 'Vm', 'Weight', 'LCO', 'rms', 'WCO', 'S_CO', 'sSFR','ID', 'aCO_Gal', 'groupno', 'Met']].values
    # LCO
    #0:z|1:flag|2:MH2|3:limMH2|4:MH2both|5:D_L|6:M*|7:V/Vm|8:Vm|9:Weight|10:LCO|
    #11:rms|12:WCO|13:S_CO|14:sSFR|15:ID|16:aCO_Gal|17:groupno|18:Met|
    #19:sigSCO|20:sigLCO|21:fracTOTLCOL|22:fracTOTLCOU|23:LMIN|24:LMAX|
    #25:aGen|26:aSr|27:aGio
    for i in range(0,len(HMassFull2)):
        for j in range(0, len(XCOH)):
            if XCOH[j,0] == HMassFull2[i,15]:
                if XCOH[j,1] == 1:
                    HMassFull2[i,16] = XCOH[j,1]
    LCO = np.vstack((LMassFull2, HMassFull2))
    ###########################################################
    a = np.zeros((len(Fulldata),1))
    a[:,0] = (LCO[:,11]*LCO[:,12])/np.sqrt(LCO[:,12]/21.57) # calculating the error
    LCO = np.hstack((LCO, a))
    LCO = lCalc(LCO, 19, 0, 5, True)# calculate the error in lum from err sco
    LCOdet = LCO[LCO[:,1]==1] #det only
    LCOND = LCO[LCO[:,1]==2] # ndonly
    a = np.zeros((len(LCOND),4))
    a[:,0], a[:,1] = 0.2, 0.
    for i in range (0,len(a)):
        a[:,2] = LCOND[:,10] - LCOND[:,10]*(0.2) # min max values for each det based on errors
        a[:,3] = LCOND[:,10] + LCOND[:,10]*(0)
    LCOND = np.hstack((LCOND, a))
    a = np.zeros((len(LCOdet),4))
    for i in range (0,len(a)):
        x = np.sqrt(((LCOdet[i,20]/LCOdet[i,10])**2) + (0.1**2) + (0.15**2) + (0.021**2))
        if math.isnan(x) == True:
            x = 0.2
        a[i,0] =  x# add this err and others in quadrature to get total error ~20%
        a[i,1] = x # add this err and others in quadrature to get total error ~20%
    # a[:,1] = np.sqrt((a[:,0]**2) + (0,1**2) + (0.15**2) + (0.021**2))
    a[:,2] = LCOdet[:,10] - LCOdet[:,10]*a[:,0] # min max values for each det based on errors
    a[:,3] = LCOdet[:,10] + LCOdet[:,10]*a[:,1]
    LCOdet = np.hstack((LCOdet, a))
    ###########################################################
    bins = np.linspace(5.5,11,18)
    conversionL = LMass[['Met', 'SFR', 'M*', 'z', 'LCO', 'Vm', 'MH2both']].values
    conversionH = HMass[['Met', 'SFR', 'M*', 'z', 'LCO', 'Vm', 'MH2both']].values
    con = np.vstack((conversionL, conversionH))
    a = np.zeros((len(con),2))
    for i in range(0, len(con)):
        a[i,0] = conversion_factor_equation(con[i,0], con[i,1], con[i,2], con[i,3])
        a[i,1] = conversion_factor_equation2(con[i,0], con[i,1], con[i,2], con[i,3])
    con = np.hstack((con,a))
    np.savetxt('giometallicity.txt', con)
    LMassFull = LMass[['z', 'flag', 'SFR', 'LCO', 'MH2', 'limMH2', 'MH2both', 'D_L', 'M*', 'V/Vm', 'Vm', 'Weight', 'Met', 'aCO']].values
    HMassFull = HMass[['z', 'flag', 'SFR', 'LCO', 'MH2', 'limMH2', 'MH2both', 'D_L', 'M*', 'V/Vm', 'Vm', 'Weight', 'Met', 'aCO']].values
    LCOSch = Schechter(LCO, 10, 8, bins)
    #############################################
    LCOtot = np.vstack((LCOdet, LCOND))
    a = np.zeros((len(LCOtot),3))
    for i in range(0,len(LCOtot)):
        a[i,0] = genzel(LCOtot[i,18])
        a[i,1] = schruba(LCOtot[i,18])
        a[i,2] = conversion_factor_equation2(LCOtot[i,18], LCOtot[i,14]+LCOtot[i,6], LCOtot[i,6], LCOtot[i,0])
    LCOtot = np.hstack((LCOtot, a))
    return LMassNDarr, HMassNDarr, Fulldata, FullData, LCO, LCOdet, [LMassFull, HMassFull], LCOSch, LCOtot
# fulldata Mh2 analysis  #######################################################
def mh2calculator(data):
    #0z|1flag|2SFR|3LCO|4MH2|5limMH2|6MH2both|7D_L|8M*|9V/Vm|10Vm|11Weight|12Zo|13agen|14asch|15agio
    LMass, HMass = data
    LMassND = LMass[LMass[:,1]==2]
    LMassD = LMass[LMass[:,1]==1]
    HMassND = HMass[HMass[:,1]==2]
    HMassD = HMass[HMass[:,1]==1]
    ND = np.vstack((LMassND, HMassND))
    alldat = [LMassD, HMassD, ND]
    newdat = []
    for i in range(0, len(alldat)):
        #met, log_sfr, log_m, redshift
        a = np.zeros((len(alldat[i]),6))
        for j in range(0,len(a)):
            a[j,0] = genzel(alldat[i][j,12])
            a[j,1] = schruba(alldat[i][j,12])
            a[j,2] = conversion_factor_equation2(alldat[i][j,12], alldat[i][j,2], alldat[i][j,8], alldat[i][j,0])
            a[j,3] = a[j,0]*alldat[i][j,3]
            a[j,4] = a[j,1]*alldat[i][j,3]
            a[j,5] = a[j,2]*alldat[i][j,3]
        newdat.append(np.hstack((alldat[i],a)))
    return newdat

## Read data from tables #######################################################
highM = atpy.Table('COLDGASS_DR3_with_Z.fits')
lowM = asciidata.open('COLDGASS_LOW_29Sep15.ascii')
SAMI = asciidata.open('SAMI_IRAM_data.txt')
Full = atpy.Table('data/COLDGASS_full.fits')
# Sort Data ####################################################################
# def dict for indices #########################################################
l = {'S_CO':11, 'z':3, 'M*':4, 'Zo':5, 'SFR':6, 'flag':15, 'NUV-r': 8, 'L_CO': 12, 'ID':0}
h = {'S_CO':16, 'z':4, 'M*':5, 'Zo':12, 'SFR':7, 'flag':21, 'NUV-r': 10, 'MH2': 19, 'ID':0}
output = {  'ID':0, 'S_CO':1, 'z':2, 'flag':3, 'M*':4, 'Zo':5, 'SFR':6, 'sSFR':7,
            'NUV-r':8,'D_L':9, 'V/Vm':10, 'Vm':11, 'L_CO':12, 'AlphaCO':13,
            'MH2':14, 'dalpha':15}
compare = {'ID':0, 'flag':37, 'M*':20, 'AlphaCO':39, 'D_L':10, 'L_CO':44, 'MH2':51, 'limMH2':52}
compareoutput = {'ID':0, 'flag':1, 'M*':2, 'AlphaCO':3, 'D_L':4, 'L_CO':5, 'MH2':6, 'limMH2':7}
# New Algo #####################################################################
HMass = np.zeros((len(highM),10))
LMass = np.zeros((len(lowM[12]),9))
hmassalpha = []
# High Mass Galaxies
for i,rows in enumerate(highM):

    HMass[i,output['ID']] = rows[h['ID']]                                       #ID
    HMass[i,output['S_CO']] = rows[h['S_CO']]                                   # S_CO
    HMass[i,output['z']] = rows[h['z']]                                         # z
    HMass[i,output['flag']] = rows[h['flag']]                                   # flag
    HMass[i,output['M*']] = rows[h['M*']]                                       # Mgal
    HMass[i,output['Zo']] = rows[h['Zo']]                                       # Zo
    HMass[i,output['SFR']] = rows[h['SFR']]                                     # SFR
    HMass[i,output['sSFR']] = np.log10(HMass[i,output['SFR']]) - HMass[i,output['M*']]      # NUV-r
    HMass[i,output['NUV-r']] = rows[h['NUV-r']]      # sSFR
    HMass[i,8] = rows[17]                               # alpha CO

# Low Mass Galaxies
LMass[:,output['ID']] = list(lowM[l['ID']])                              #ID
LMass[:,output['S_CO']] = list(lowM[l['S_CO']])                         # S_CO
LMass[:,output['z']] = list(lowM[l['z']])                               # z
LMass[:,output['flag']] = list(lowM[l['flag']])                         # flag
LMass[:,output['M*']] = list(lowM[l['M*']])                             # Mgal
LMass[:,output['Zo']] = list(lowM[l['Zo']])                             # Zo
LMass[:,output['SFR']] = list(lowM[l['SFR']])                           # SFR
sSFRlist = sSFR(list(lowM[l['SFR']]), list(lowM[l['M*']]))
LMass[:,output['sSFR']] = sSFRlist                                      # sSFR
LMass[:,output['NUV-r']] = list(lowM[l['NUV-r']])      # NUV-r
#ID:0|S_CO:1|z:2|flag:3|M*:4|Zo:5|SFR:6|sSFR:7|NUV-r:8|D_L:9|V/Vm:10|Vm:11|L_CO:12|ACO:13|MH2:14|dACO:15|
################################################################################
LND, HND, FullData, weights, LCO, LCOdet, gdata, LCOSch, LCOtot = GetFull(Full, output)
plotdata = mh2calculator(gdata)
testMassLimits(LCOdet)
LND[:,output['MH2']] = 10**LND[:,output['MH2']]
HND[:,output['MH2']] = 10**HND[:,output['MH2']]
# FullData = Vm(FullData, 5, min(FullData[:,0]), max(FullData[:,0]), 3)
FullData[:,2] = 10**FullData[:,2]
FullData[:,3] = 10**FullData[:,3]
FullData[:,4] = 10**FullData[:,4]
#0:z|1:flag|2:MH2|3:limMH2|4:MH2_both|5:Lumdist|6:M*|7:V/Vm|8:Vm
FullDet = FullData[FullData[:,1] == 1]
FullND = FullData[FullData[:,1] == 2]
################################################################################


################################################################################
# Separate into non detections and detections
LMassND, HMassND = LMass, HMass
LMass = NonDetect(LMass, output['flag'], True)
HMass = NonDetect(HMass, output['flag'], True)
# get the alpha CO mass values for later but then delete from the data so it matches up
alphaCOHMass = HMass[:,8]
HMass = np.delete(HMass,8,1)
HMassND = np.delete(HMassND,8,1)
# for low mass non detections use the 5 sigma upper limit for L_CO
# store this in the 0th column
LMassND[:,1] = list(lowM[l['L_CO']])
LMassND = NonDetect(LMassND, output['flag'], False)
# for high mass non detections use the calculated upper limit for MH2
# store this in the 0th column
for i,rows in enumerate(highM):
    HMassND[i,1] = 10**rows[h['MH2']]
HMassND = NonDetect(HMassND, output['flag'], False)


# Calculate Luminosity distance for each galaxy ################################
# | S_CO | z | flag | Mgal | Zo | D_L |
LMass = lumdistance(LMass, output['z'])
HMass = lumdistance(HMass, output['z'])
LMassND = lumdistance(LMassND, output['z'])
HMassND = lumdistance(HMassND, output['z'])
# Calculate Vm #################################################################
# | S_CO | z | flag | Mgal | Zo | D_L | V/Vm | Vm |
LMass = Vm(LMass,output['D_L'], min(LMass[:,output['z']]), max(LMass[:,output['z']]), 1)
HMass = Vm(HMass,output['D_L'], min(HMass[:,output['z']]), max(HMass[:,output['z']]), 2)
LMassND = Vm(LMassND, output['D_L'], min(LMass[:,output['z']]), max(LMass[:,output['z']]), 1)
HMassND = Vm(HMassND, output['D_L'], min(HMass[:,output['z']]), max(HMass[:,output['z']]), 2)
# Calculate Luminosity Values ##################################################
# | S_CO | z | flag | Mgal | Zo | D_L | V/Vm | Vm | L_CO |
LMass = lCalc(LMass,output['S_CO'],output['z'],output['D_L'],True)
HMass = lCalc(HMass,output['S_CO'],output['z'],output['D_L'],False)
dummy = np.zeros((len(LMassND),1))
# move the stored L_CO upper limit data over to the correct column
dummy[:,0] = LMassND[:,1]
LMassND = np.hstack((LMassND, 10**dummy))
# leave dummy 0 data for L_CO as we already have MH2
dummy = np.zeros((len(HMassND),1))
HMassND = np.hstack((HMassND, dummy))
# Calculate MH2 ################################################################
# | S_CO | z | flag | Mgal | Zo | D_L | V/Vm | Vm | L_CO | AlphaCO | MH2 | dalpha |
LMass = H2Conversion(LMass, output['Zo'], output['L_CO'])
HMass = H2Conversion(HMass, output['Zo'], output['L_CO'], alphaCOHMass)
LMassND = H2Conversion(LMassND, output['Zo'], output['L_CO'])
# just add 0 axes and copy over the MH2 upper limits from the table
dummy1 = np.zeros((len(HMassND),3))
HMassND = np.hstack((HMassND, dummy1))
HMassND[:,output['MH2']] = HMassND[:,1]
Mass = np.append(LMass[:,output['M*']], HMass[:,output['M*']])
alpha = np.append(LMass[:,output['AlphaCO']], HMass[:,output['AlphaCO']])
alphaerror = np.append(LMass[:,output['dalpha']], HMass[:,output['dalpha']])
# MH2 = np.append(LMass[:,output['MH2']], HMass[:,output['MH2']])
# NH2, xH2 = sortIntoBins(MH2, 30)
# NH2L ,xH2L = sortIntoBins(LMass[:,output['MH2']], 15)
# NH2H ,xH2H = sortIntoBins(HMass[:,output['MH2']], 15)
#

# ################################################################################
# lumsL = LMass[:,output['L_CO']]
# lumsH = HMass[:,output['L_CO']]
# #lumsnew = lumsnew[:,5]
#
# lumsL = [i for i in lumsL if i > 0.0]         # remove 0 detected CO flux galaxies
# lumsH = [i for i in lumsH if i > 0.0]         # remove 0 detected CO flux galaxies
# lumsLlog, lumsHlog = np.log10(lumsL), np.log10(lumsH)
# lCombinedlog = np.append(lumsLlog, lumsHlog)
# lCombined = np.append(lumsL, lumsH)
#
# # Sort Luminosity Values into bins #############################################
# NL, midL = sortIntoBins(lumsLlog, 15)
# NH, midH = sortIntoBins(lumsHlog, 15)
# NC, midC = sortIntoBins(lCombinedlog, 20)
# #NR, midR = sortIntoBins(lumsnew, 15)
#
# # Calculations for Mass Distribution ###########################################
#
# totalMass = np.append(LMass[:,output['M*']], HMass[:,output['M*']])
# Nmass, Xmass = sortIntoBins(totalMass, 30)

# density schechter ############################################################
ND = np.vstack((LMassND, HMassND))
totaldet = np.vstack((LMass, HMass))
total = np.vstack((totaldet, ND))
#N, rho, xbins = Schechter(total, output['L_CO'], output['Vm'])
#MH2 total
# Nh2, rhoh2, xbinsh2 = Schechter(total, output['MH2'], output['Vm'])
low, high = min(np.log10(total[:,output['MH2']])), max(np.log10(total[:,output['MH2']]))
bins = np.linspace(low, high, 16) # log-spaced bins
LSch = Schechter(LMass, output['MH2'], output['Vm'], bins)
HSch = Schechter(HMass, output['MH2'], output['Vm'], bins)
NDSch = Schechter(ND, output['MH2'], output['Vm'], bins)
totSch = Schechter(total, output['MH2'], output['Vm'], bins)
detSch = Schechter(totaldet, output['MH2'], output['Vm'], bins)
FullDetSchech = Schechter(FullDet, 2, 8, bins)
FullNDSchech = Schechter(FullND, 3, 8, bins)
FullSchech = Schechter(FullData, 4, 8, bins)
total3sig = np.vstack((totaldet, LND))
total3sig = np.vstack((total3sig, HND))
totSch2 = Schechter(total3sig, output['MH2'], output['Vm'], bins)
totSch3 = Schechter(FullData, 4, 8, bins)
NDSch2 = Schechter(np.vstack((LND, HND)), output['MH2'], output['Vm'], bins)
#Nh2ND2, rhoh2ND2, xbinsh2ND2 = Schechter(HMassND, output['MH2'], output['Vm'])
# fit schechter ################################################################
# x1,x2 = xbins, xbins[4:]
# y1,y2 = rho,rho[4:]
# popt1 = schechter.log_schechter_fit(x1, y1)
# phi1, L01, alpha1 = popt1
# popt2 = schechter.log_schechter_fit(x2, y2)
# phi2, L02, alpha2 = popt2
# poptkeres = np.log10(0.00072), np.log10(9.8*math.pow(10,6)), -1.3
# xnew = np.linspace(max(xbins),min(xbins),100)
# ynew1 = schechter.log_schechter(xnew, *popt1)
# ynew2 = schechter.log_schechter(xnew, *popt2)
# ykeres = schechter.log_schechter(xnew, *poptkeres)

# Keres fit
mst=np.log10((2.81*(10**9))/(0.7**2))
alpha=-1.18
phist=np.log10(0.0089*(0.7**3))
mst1 = 10**mst
phist1 = 10**phist
xkeres = np.linspace(7,11,200)
x1 = 10**xkeres
xnew = np.linspace(7,11,24)
xnew1 = 10**xnew

# bins = list(totSch[2])
# for i in range(0,len(bins)):
#     bins[i] = 10**bins[i]

ykeres = schechter.log_schechter(xkeres, phist, mst, alpha)
ykeres2 = np.log10((phist1)*((x1/(mst1))**(alpha+1))*np.exp(-x1/mst1)*np.log(10))
yrho = ykeres2 + np.log10(x1)
# ykeres2sh = np.log10((phist1)*((bins/(mst1))**(alpha+1))*np.exp(-bins/mst1)*np.log(10))
# ykeresph2 = ykeres2sh+totSch[2]

#fit our data to a schechter function and plot
x_keres = 10**np.linspace(7, 11, 500)
##ober##########################################################################
mstcorr = (7.5*(10**8))/(0.7**2)
alphacorr = -1.07
phistcorr = 0.0243*(0.7**3)
y_ober = np.log10((phistcorr)*((x_keres/(mstcorr))**(alphacorr+1))*np.exp(-x_keres/mstcorr)*np.log(10))
################################################################################
CG_para = schechter.log_schechter_fit(totSch[2][6:], totSch[1][6:])
y_CG = schechter.log_schechter(xkeres, *CG_para)
yrhoCG = y_CG + xkeres
y_CG2 = schechter.log_schechter(np.log10(x_keres), *CG_para)
yrhoCG2 = y_CG2 + np.log10(x_keres)

yCGpts = schechter.log_schechter(totSch[2], *CG_para)
yrhoCGpts = yCGpts + totSch[2]
y_keres = np.log10((phist1)*((x_keres/(mst1))**(alpha+1))*np.exp(-x_keres/mst1)*np.log(10))
x_keres = np.log10(x_keres)
yrhokeres = y_keres + x_keres

det_para = schechter.log_schechter_fit(detSch[2][5:], detSch[1][5:])
y_det = schechter.log_schechter(xkeres, *det_para)

er = errors(total, bins, totSch[1], output)
erdet = errors(totaldet, bins, detSch[1], output)
sigma = []
for i in range(0, np.shape(er)[1]):
    eri = er[:,i]
    eri = eri[abs(eri)<99]
    sigma.append(np.std(eri))

sigmadet = []
for i in range(0, np.shape(erdet)[1]):
    eri = er[:,i]
    eri = eri[abs(eri)<99]
    sigmadet.append(np.std(eri))
# BlueRed#######################################################################
FullSchech, sdssSchech, sdssSchechAm, totSch1, totSch2, FullDatasim = BlueRed.main(bins, totSch, totSch2, totSch3, sigma, LSch, HSch, NDSch, NDSch2, FullDetSchech, FullSchech, FullDet)
################################################################################
Plotweights(weights)
fullcomparedata = comparefull(Full, compare)
PlotSchechter(LSch, HSch, NDSch, totSch, xkeres, ykeres2, y_CG, sigma, y_ober, plotdata, bins, LCOtot)
PlotSchechter2(totSch, sigma, y_CG, detSch, sigmadet, y_det, xkeres, ykeres2, )
PlotSchechter3(LSch, HSch, NDSch, totSch, xkeres, ykeres2, y_ober, bins, LCOtot, y_CG)
PlotGalactic(LSch, HSch, NDSch, totSch, xkeres, ykeres2, y_ober, bins, LCOtot, y_CG)
PlotSchruba(LSch, HSch, NDSch, totSch, xkeres, ykeres2, y_ober, bins, LCOtot, y_CG)
PlotAccurso(LSch, HSch, NDSch, totSch, xkeres, ykeres2, y_ober, bins, LCOtot, y_CG)
PlotRhoH2(LSch, HSch, NDSch, totSch, xkeres, np.log10(x1), yrho, yrhoCG, yrhoCGpts, yrhoCG2, yrhokeres, x_keres)
PlotAlphaCO(total, output)
PlotMsunvsMH2(total, output)
PlotSchechterFull(FullDetSchech, FullNDSchech, FullSchech, x_keres, y_keres, y_ober)
PlotMstarMH2(total, FullData, ND, FullND, output, FullDatasim)
compareIDforID(fullcomparedata, total, output, compareoutput)
x1 = np.log10(x1)

print 'p_H2 COLD GASS', np.sum((10**yrhoCGpts)*(totSch[2][1]-totSch[2][0]))/(10**7)
print 'p_H2 Keres', np.sum((10**yrhokeres)*(totSch[2][1]-totSch[2][0]))/(10**7)
print 'p_H2 COLD GASS', OmegaH2(x_keres, yrhoCG2), np.sqrt(np.sum(np.array(sigma)*np.array(sigma)))
print 'p_H2 Keres', OmegaH2(x_keres, yrhokeres)
print x_keres[0]
fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
ax[0,0].scatter(x_keres, yrhokeres, marker = 's', s = 100, edgecolor='blue', linewidth='2', facecolor='none', label = 'Low Mass')
ax[0,0].scatter(x_keres, yrhoCG2, marker = 's', s = 100, edgecolor='green', linewidth='2', facecolor='none', label = 'High Mass')
ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
ax[0,0].set_ylabel(r'$\mathrm{log\, \rho_{H2}\, [M_{\odot}\, Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
ax[0,0].set_ylim(4, 7.5)
ax[0,0].set_xlim(7, 11)
plt.savefig('img/schechter/pH2.pdf', format='pdf', dpi=250, transparent = False)


# # gas fractions ################################################################
# SAMI_outflows = [   567624,574200,228432,239249,31452,238125,486834,
#                     417678,106389,593680,618906,618220,383259,209807,376121]
#
# SAMI_NUV_r = [      2.34, 1.95, 2.13, 2.76, 2.14, 2.88, 2.87, 5.64, 3.30,
#                     5.05, 4.48, 3.78, 3.48, 3.25, 4.12]
#
# SAMI_SFR = [        0.39, 0.65, 0.56, 0.24, 0.72, 0.28, 0.51, 0.04, 0.84, 0.16, 0.46,
#                     1.14, 2.30, 3.66, 2.02]
#
# SAMI_Halpha = [     0.5620484135, 1.5436637384, 2.7566674648, 0.5648367555,
#                     1.0654768981, 1.612541277, 0.8870459836, 7.2135198788,
#                     1.5275349255, 2.3419404859, 28.8979144739, 1.4031710791,
#                     9.2286257903 ,4.7398961907,2.7929139441]
#
# df = pd.read_csv('SFR_cat1.csv')
# SFR_ids = df[['CATAID']].values
# SFR_meas = df[['SFR_Ha', 'SFR_W3', 'SFR_W4', 'SFR_FUV', 'SFR_NUV', 'SFR_u']].values
#
# SAMI_data = np.zeros((len(SAMI_outflows),10))
# for i in range(0, len(SAMI[0])):
#     if SAMI[0][i] in SAMI_outflows:
#         GAMAID = SAMI[0][i]
#         ind = FindIndex(GAMAID, SAMI_outflows)
#         SAMI_data[ind,0] = SAMI_outflows[ind] # GAMA ID
#         SAMI_data[ind,1] = SAMI[2][i] # Mgal
#         SAMI_data[ind,2] = SAMI[6][i] # MH2
#         SAMI_data[ind,3] = SAMI[8][i] # flag
#         SAMI_data[ind,4] = np.log10(SAMI[7][i]) # amelie's calc gas fraction
#         SAMI_data[ind,5] = fgas(SAMI_data[ind,2],SAMI_data[ind,1]) #my calc
#         SAMI_data[ind,6] = SAMI_SFR[ind]
#         SAMI_data[ind,7] = np.log10(SAMI_data[ind,6]) - SAMI_data[ind,1]
#         SAMI_data[ind,8] = np.log10(SFR_meas[ind,5]) - SAMI_data[ind,1]
#         SAMI_data[ind,9] = SAMI_NUV_r[ind]
# fgasL, fgasH = [], []
# for i in range (len(LMass)):
#     fgasL.append(fgas(LMass[i,output['MH2']], LMass[i,output['M*']]))
# for i in range (len(HMass)):
#     fgasH.append(fgas(HMass[i,output['MH2']], HMass[i,output['M*']]))
# CG_X = np.append(LMass[:,output['M*']], HMass[:,output['M*']])
# CG_Y = np.append(fgasL, fgasH)
# sSFR_X = np.append(LMass[:,output['sSFR']], HMass[:,output['sSFR']])
# SFR_X = np.log10(np.append(LMass[:,output['SFR']], HMass[:,output['SFR']]))
# CG_NUVr = np.append(LMass[:,output['NUV-r']], HMass[:,output['NUV-r']])
#
# ################################################################################
# order = 'GAMA ID \t M* \t MH2 \t flag \t fgas1 \t fgas2 \t SFR \t sSFR'
# np.savetxt('SAMI.txt', SAMI_data, delimiter='\t', fmt= '%1.2f', header = order)

################################################################################
# fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False)
# ax[0,0].scatter(CG_X, SFR_X, c='k', label = 'COLD GASS detection', alpha=0.2, s=30)
# ax[0,0].set_xlabel(r'$log_{10}(M_{*}/M_{\odot})$', fontsize = 20)
# ax[0,0].set_ylabel(r'$log_{10}(SFR)$', fontsize = 20)
# # ax[0,0].set_xlim(9,11.5)
# # ax[0,0].set_ylim(-2.5,0)
#
# fig.set_size_inches(10,6)
# plt.savefig('SFRvsM.png', transparent = False ,dpi=250)

# SAMI_data_detect = SAMI_data[SAMI_data[:,3]<2]
# SAMI_data_nondetect = SAMI_data[SAMI_data[:,3]>1]

# ###############################################################################


#fig = plt.figure(figsize=(8,6))
# fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False)
# ax[0,0].scatter(CG_X, CG_Y, c='k', label = 'COLD GASS detection', alpha=0.2, s=30)
# ax[0,0].scatter(SAMI_data_detect[:,1], SAMI_data_detect[:,5], label = 'SAMI-IRAM detection', s=100, c='g')
# ax[0,0].scatter(SAMI_data_nondetect[:,1], SAMI_data_nondetect[:,5], label = 'SAMI-IRAM no detection', s=100, c='r')
# ax[0,0].set_xlabel(r'$log_{10}(M_{*}/M_{\odot})$', fontsize = 20)
# ax[0,0].set_ylabel(r'$log_{10}(M_{H2}/M_{*})$', fontsize = 20)
# ax[0,0].set_xlim(9,11.5)
# ax[0,0].set_ylim(-2.5,0)
# # for i, txt in enumerate(SAMI_outflows):
# #     ax[0,0].annotate(str(txt), (0.03+SAMI_data[i,1],SAMI_data[i,5]))
# ax[0,0].legend(bbox_to_anchor=(1.1,1.14), loc='upper center', ncol=3)
# ax[0,1].scatter(sSFR_X, CG_Y, c='k' , label = 'COLD GASS detection', alpha=0.2, s=30)
# # ax[0,1].scatter(SAMI_data_detect[:,7], SAMI_data_detect[:,5], label = 'SAMI detection', s=100, c='r')
# # ax[0,1].scatter(SAMI_data_nondetect[:,7], SAMI_data_nondetect[:,5], label = 'SAMI no detection', s=100, c='r')
# ax[0,1].scatter(SAMI_data_detect[:,8], SAMI_data_detect[:,5], label = 'SAMI detection', s=100, c='g')
# ax[0,1].scatter(SAMI_data_nondetect[:,8], SAMI_data_nondetect[:,5], label = 'SAMI no detection', s=100, c='r')
# ax[0,1].set_xlabel(r'$log_{10}(\mathrm{sSFR})$', fontsize = 20)
# ax[0,1].set_xlim(-12,-8)
# ax[0,1].set_ylim(-2.5,0)
# #ax[0,1].set_ylabel(r'$log_{10}(M_{H2}/M_{*})$', fontsize = 20)
# #ax[0,1].legend(loc=2)
# # for i, txt in enumerate(SAMI_outflows):
# #     ax[0,1].annotate(str(int(SAMI_data[i,0])), (0.03+SAMI_data[i,7],SAMI_data[i,5]))

# ax[0,0].scatter(CG_NUVr, CG_Y, c='k' , label = 'COLD GASS detection', alpha=0.2, s=30)
# ax[0,0].scatter(SAMI_data_detect[:,9], SAMI_data_detect[:,5], label = 'SAMI-IRAM detection', s=100, c='g')
# ax[0,0].legend(bbox_to_anchor=(0.5,1.135), loc='upper center', ncol=2, fontsize = 13)
# # ax[0,0].scatter(SAMI_data_nondetect[:,9], SAMI_data_nondetect[:,5], label = 'SAMI-IRAM no detection', s=100, c='r')
# ax[0,0].set_xlabel(r'$\mathrm{NUV\minus r}$', fontsize = 20)
# ax[0,0].set_ylabel(r'$log_{10}(M_{H2}/M_{*})$', fontsize = 20)
# ax[0,0].set_xlim(1,7)
# ax[0,0].set_ylim(-2.5,0)
# fig.set_size_inches(7,6)
# for i, txt in enumerate(SAMI_data_detect):
#     ax[0,0].annotate(str(int(SAMI_data_detect[i,0])), (0.09+SAMI_data_detect[i,9],SAMI_data_detect[i,5]), fontsize=8)
# plt.savefig('IRAM_NUV-r_detections4.pdf', format='pdf', dpi=1000, transparent = False)


################################################################################
# M1 = np.append(HMass[:,output['M*']], LMass[:,output['M*']])
# SFR1 = np.log10(np.append(HMass[:,output['SFR']], LMass[:,output['SFR']]))
# plt.plot(M1, SFR1, 'ko')
# plt.show()

# # Plot Luminosity number plot ################################################
# fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
# ax[0,0].plot(midL,NL,'b-', label = 'Low mass')
# ax[0,0].plot(midH,NH,'r-', label = 'high mass')
# ax[0,0].plot(midC,NC,'g-', label = 'lCombined')
# #ax[0,0].plot(midR,NR,'k-', label = 'Pre-calc')
# ax[0,0].set_xlabel(r'$log_{10}(L_{CO})$', fontsize=20)
# ax[0,0].set_ylabel(r'$log_{10}(N)$', fontsize=20)
# ax[0,0].set_title('CO Luminosity', fontsize=20)
# ax[0,0].legend()
#
# Plot H2 mass #################################################################
# ax[0,1].plot(xH2, NH2,'b-', label = 'H2 Mass')
# #ax[0,0].plot(xH2L,NH2L,'r-', label = 'lowM MH2')
# #ax[0,0].plot(xH2H,NH2H,'g-', label = 'highM MH2')
# ax[0,1].set_xlabel(r'$log_{10}(M_{H2}/M_{\odot})$', fontsize=20)
# ax[0,1].set_ylabel(r'$log_{10}(N_{gal})$', fontsize=20)
# #ax[0,1].set_title('CO Luminosity', fontsize=20)
# ax[0,1].legend(loc=3)
#

# # # Plot V/Vm ##################################################################
# ax[0,0].plot(LMass[:,output['L_CO']], LMass[:,output['V/Vm']],'ko', label = 'low mass')
# ax[0,0].plot(HMass[:,output['L_CO']], HMass[:,output['V/Vm']],'ro', label = 'high mass')
# ax[0,0].axhline(y=np.average(LMass[:,output['V/Vm']]),color='k', label = 'average low')
# ax[0,0].axhline(y=np.average(HMass[:,output['V/Vm']]),color='r', label = 'average high')
# ax[0,0].set_xlabel(r'$log_{10}(L_{CO})$', fontsize=20)
# ax[0,0].set_ylabel(r'$\frac{V}{V_{m}}$', fontsize=20)
# #ax[1,0].set_title('Schmidt Vm', fontsize=20)
# ax[0,0].legend()
#
# # Plot alpha vs Mgal ###########################################################
# ax[0,1].errorbar(Mass, alpha, yerr=alphaerror, fmt='o')
# ax[0,1].set_xlabel(r'$log_{10}(M_{gal})$', fontsize=20)
# ax[0,1].set_ylabel(r'$\alpha_{CO}$', fontsize=20)
# #ax[1,1].set_title(r'$\alpha_{CO}$ vs $M_{gal}$', fontsize=20)
#
# # schecter #####################################################################
# # ax[1,2].plot(xbins, rho, 'bo')
# # ax[1,2].plot(xbins[4:], rho[4:], 'ro')
# # ax[1,2].set_xlabel(r'$log_{10}(L_{CO})$', fontsize=20)
# # ax[1,2].set_ylabel(r'$log_{10}{\rho(L)}$', fontsize=20)
# # ax[1,2].set_title('Schechter', fontsize=20)

# plt.show()
