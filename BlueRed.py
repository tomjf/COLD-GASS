import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import schechter
import random
import scal_relns
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def log_schechter(logL, log_rho, log_Lstar, alpha):
    rholist = []
    L = 10**logL
    pstar = 10**log_rho
    Lstar = 10**log_Lstar
    log = np.log(10)
    for i in range(0,len(L)):
        frac = L[i]/Lstar
        rho = pstar*(frac**(alpha+1))*math.exp(-frac)*log
        rholist.append(rho)
    return np.log10(rholist)
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
def mainSequence(Mstar, spread):
    logSFRlist = []
    for index, M in enumerate(Mstar):
        logSFR = - (2.332*M) + (0.4156*M*M) - (0.01828*M*M*M)
        if spread == True:
            logSFR += random.gauss(0,0.3)
        logSFRlist.append(logSFR)
    return logSFRlist
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def cloud(Mstar):
    logSFRlist = []
    for index, M in enumerate(Mstar):
        logSFR = -1 + random.gauss(0,0.3)
        logSFRlist.append(logSFR)
    return logSFRlist

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
    ax[0,0].set_xlim(8,12)
    ax[0,0].set_ylim(-6,-1)
    ax[0,0].plot(L,np.log10(yBaldry), 'k', linewidth = 3)
    ax[0,0].plot(L,np.log10(yred), 'r', linewidth = 3)
    ax[0,0].plot(L,np.log10(yblue), 'b', linewidth = 3)
    ax[0,0].set_xlabel(r'$\mathrm{log \, M_{*}\, [M_{sun}]}$', fontsize = 20)
    ax[0,0].set_ylabel(r'$\mathrm{log \, N}$', fontsize = 20)
    plt.savefig('img/scal/Baldry.eps', format='eps', dpi=250, transparent = False)
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
################################################################################
d = 100
L = np.linspace(8,11.9,24)
LKeres = np.linspace(4,8,200)

Spheroid = (3.67/10000), 10.74, -0.525
Disk = (0.855/10000), 10.70, -1.39
Keres = (7.2/10000), 7.0, -1.30

#spheroid is red
ySpheroid = log_schechter(L, *Spheroid)
#disk is blue
yDisk = log_schechter(L, *Disk)
yKeres = schechter(LKeres, Keres[1], (L[1]-L[0]), Keres[0], Keres[2])

yBaldry = doubleschechter(L, 10.66, (L[1]-L[0]), 0.00396, 0.00079, -0.35, -1.47)
yred = schechter(L, 10.66, (L[1]-L[0]), 0.00396, -0.35)
yblue = schechter(L, 10.66, (L[1]-L[0]), 0.00079, -1.47)

###### make a table for the red galaxies #######################################
red = np.zeros((len(yred),4))
# the luminosity bin
red[:,0] = L
# spacing between bins
red[:,1] = 0.2
# the number of galaxies in this luminosity bin from the schechter function
red[:,2] = yred
#list of all the galaxies over all the bins
redpop = []
for i in range(0,len(yred)):
    red[i,3] = int(red[i,2]*d*d)
    for j in range(0,int(red[i,3])):
        redpop.append(random.uniform((red[i,0]-(0.5*red[i,1])), (red[i,0]+(0.5*red[i,1]))))

###### make a table for the blue galaxies ######################################
blue = np.zeros((len(yblue),4))
# the luminosity bin
blue[:,0] = L
# spacing between bins
blue[:,1] = 0.2
# the number of galaxies in this luminosity bin from the schechter function
blue[:,2] = yblue
#list of all the galaxies over all the bins
bluepop = []
for i in range(0,len(yblue)):
    blue[i,3] = int(blue[i,2]*d*d)
    for j in range(0,int(blue[i,3])):
        bluepop.append(random.uniform((blue[i,0]-(0.5*blue[i,1])), (blue[i,0]+(0.5*blue[i,1]))))
###### make a table for all the galaxies #######################################
Baldry = np.zeros((len(yBaldry),4))
Baldry[:,0] = L
Baldry[:,1] = 0.2
Baldry[:,2] = yBaldry
total = np.append(bluepop, redpop)

x = mainSequence(bluepop, True)
z = cloud(redpop)
trend = np.array(mainSequence(L,    False))
data = np.zeros((len(L),2))
data[:,0] = L
data[:,1] = trend

datagio, fit = scal_relns.fitdata2()

bluepop1 = np.zeros((len(bluepop),4))
redpop1 = np.zeros((len(redpop),4))

bluepop1[:,0], bluepop1[:,1] = bluepop, x
bluepop1[:,2] = scal_relns.second2var((bluepop1[:,0], bluepop1[:,1]), *fit[0])

redpop1[:,0], redpop1[:,1] = redpop, z
redpop1[:,2] = scal_relns.second2var((redpop1[:,0], redpop1[:,1]), *fit[0])

PlotBaldry(L, yBaldry, yred, yblue)
PlotMSFR(bluepop, redpop, x, z, data)
PlotHist(bluepop, redpop)
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
