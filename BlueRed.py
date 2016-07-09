import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from schechter import log_schechter_fit, log_schechter
import random
import scal_relns
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
    ax[0,0].plot(L,np.log10(yBaldry), 'k', linewidth = 3)
    ax[0,0].plot(L,np.log10(yred), 'r', linewidth = 3)
    ax[0,0].plot(L,np.log10(yblue), 'b', linewidth = 3)
    ax[0,0].set_xlabel(r'$\mathrm{log \, M_{*}\, [M_{sun}]}$', fontsize = 20)
    ax[0,0].set_ylabel(r'$\mathrm{log \, (number \, density) \,[Mpc^{-3}\, dex^{-1}]}$', fontsize = 20)
    plt.savefig('img/scal/Baldry.eps', format='eps', dpi=250, transparent = False)
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
    plt.savefig('img/scal/MSFR.eps', format='eps', dpi=250, transparent = False)
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
def PlotRhoH2(totSch, x, y):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(totSch[2], totSch[4], marker = 's', s = 100, edgecolor='blue', linewidth='2', facecolor='none', label = 'Low Mass')
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
def PlotSchechter(totSch, redSch, blueSch, x, y_scalfit, x_scal, y_keres):
    xmajorLocator   = MultipleLocator(0.5)
    xminorLocator   = MultipleLocator(0.1)
    ymajorLocator   = MultipleLocator(0.5)
    yminorLocator   = MultipleLocator(0.1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    ax[0,0].errorbar(totSch[2], totSch[1], fmt = 'o', markersize = 10, color = 'red', label = 'Scaling Relation Method')
    ax[0,0].plot(x, y_scalfit, color ='red', label = 'Scaling Relation Fit')
    ax[0,0].plot(np.log10(x_keres), y_keres, 'k--', label = 'Keres+03')
    # ax[0,0].scatter(x_scal, y_scal)
    # ax[0,0].errorbar(redSch[2], redSch[1], fmt = 'o', markersize = 10, color = 'red', label = 'Red')
    # ax[0,0].errorbar(blueSch[2], blueSch[1], fmt = 'o', markersize = 10, color = 'blue', label = 'Blue')
    ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-5, -1)
    ax[0,0].set_xlim(7.5, 10.5)
    plt.legend(fontsize = 13)
    plt.savefig('img/scal/MH2.eps', format='eps', dpi=250, transparent = False)
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
    plt.savefig('img/scal/Mstar.eps', format='eps', dpi=250, transparent = False)
    plt.savefig('img/scal/Mstar.pdf', format='pdf', dpi=250, transparent = False)
    # plt.savefig('img/MH2.png', transparent = False ,dpi=250)

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
                # print p
                # print 'helloooo'
                o += 1/(data[j,Vmaxis]**2)
                pH2 += data[j,LCOaxis]/data[j,Vmaxis]
                Num+=1
            # print p, '@@@@'
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
        print float(i)/float(eridx)
        random.shuffle(idx)
        idx1 = idx[:eridx]
        newdata = np.zeros((eridx, np.shape(data)[1]))
        for j in range(0,len(newdata)):
            newdata[j,:] = data[idx[j],:]
        newdata[:,output['Vm']] = newdata[:,output['Vm']]*0.8
        totSch = Schechter(newdata, 6, 7, x)
        drho = totSch[1] - y
        spread[i,:] = drho
    return spread
#########################################################################
V = 100000
L = np.linspace(8,11.9,24)
LKeres = np.linspace(4,8,200)

Spheroid = (3.67/10000), 10.74, -0.525
Disk = (0.855/10000), 10.70, -1.39
Keres = (7.2/10000), 7.0, -1.30
output = { 'Vm':6, 'MH2':13}

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

# add starformation rates
blues = mainSequence(blues, True, 4, False)
reds = cloud(reds)
trend = mainSequence(Baldry, False, 0, False)
data = np.zeros((len(L),2))
data[:,0] = L
data[:,1] = trend[:,5]

datagio, fit = scal_relns.fitdata2()
blues = np.hstack((blues, np.zeros((len(blues),1))))
reds = np.hstack((reds, np.zeros((len(reds),1))))
blues[:,6] = scal_relns.second2var((blues[:,4], blues[:,5]), *fit[0])
reds[:,6] = scal_relns.second2var((reds[:,4], reds[:,5]), *fit[0])

# add V
blues = np.hstack((blues, np.zeros((len(blues),1))))
reds = np.hstack((reds, np.zeros((len(reds),1))))
blues[:,7] = V
reds[:,7] = V
total = np.vstack((blues, reds))
# bins  = np.linspace(6, 10.5, 30)
bins  = np.linspace(min(total[:,6]), max(total[:,6]), 25)
massbins = np.linspace(8,11.5,25)
totSch = Schechter(total, 6, 7, bins)
redSch = Schechter(reds, 6, 7, bins)
blueSch = Schechter(blues, 6, 7, bins)
MassSchB = Schechter(blues, 4, 7, massbins)
MassSchR = Schechter(reds, 4, 7, massbins)
x = np.linspace(7.5,10.5,200)
scalfit = log_schechter_fit(totSch[2], totSch[1])
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



scal_para = log_schechter_fit(totSch[2][9:], totSch[1][9:])
y_scal = log_schechter(x_scal, *scal_para)
rhoscal = y_scal + x_scal

# er = errors(total, bins, totSch[1], output)
# sigma = []
# for i in range(0, np.shape(er)[1]):
#     eri = er[:,i]
#     eri = eri[abs(eri)<99]
#     sigma.append(np.std(eri))
# print sigma

print OmegaH2(totSch[2], totSch[1]+totSch[2])
print y_scalfit
print x3
print OmegaH2(x3, x3+y_scalfit)
PlotBaldry(L, yBaldry, yred, yblue)
PlotMSFR(blues[:,4], reds[:,4], blues[:,5], reds[:,5], data)
PlotHist(blues[:,4], reds[:,4])
PlotSimMH2(blues, reds)
# PlotRhoH2(totSch, x_scal, rhoscal)
PlotSchechter(totSch, redSch, blueSch, x3, y_scalfit, x_scal, y_keres)
PlotSchechterMass(MassSchB, MassSchR, L, yred, yblue)

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
