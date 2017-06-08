import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
from main import GetFull
import atpy
import math
import random
import schechter2
import pandas as pd
import multivariate

def read_Lagos_data(fname):
     keres = pd.read_csv(fname, sep=",", header = None)
     keres.columns = ["x", "y", "erup", "erdn"]
     hobs=0.7
     Hubble_h=0.6777
     keres['lgCOB'] = keres['x'] + np.log10((hobs**2)/(Hubble_h**2))
     keres['lgdndlgLCOB'] = keres['y'] + np.log10((hobs**3)/(Hubble_h**3))
     keres['errupB'] = abs(keres['y']-keres['erup'])
     keres['errdnB'] = abs(keres['erdn']-keres['y'])
     X_CO = 2.0
     keres['MH2_x'] = keres['lgCOB'] + np.log10(580.0*X_CO) + (2*np.log10(2.6)) - np.log10(4*np.pi)
    #  print (keres)
     return keres

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
                dx = data[-1,0] - data[-2,0]
                dy = data[-1,1] - data[-2,1]
                DX = x[j] - data[-1,0]
                y = data[-1,1]
                ans = y + ((DX/dx)*dy)
                newdata[j,1] = ans
                newdata[j,0] = x[j]
    return newdata

def errors(data, x, y):
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


def PlotLum(data, uperr, lowerr, keres, lowerr2, uperr2, LCOSch, LCOdet, LCO, LCOtot, keresB, keres60):
    bins = np.linspace(5.5,11,18)
    bins2 = np.linspace(5.5,11,300)
    LCOSch = Schechter(LCOtot, 10, 8, bins)
    CG_para2, CG_para2cov = schechter2.log_schechter_fit(LCOSch[2][6:14], LCOSch[1][6:14])
    perr = np.sqrt(np.diag(CG_para2cov))
    print ('phi_star = ' + str(round(CG_para2[0],2)) + ' +/- ' + str(round(perr[0],2)))
    print ('L_0 = ' + str(round(CG_para2[1],2)) + ' +/- ' + str(round(perr[1],2)))
    print ('alpha = ' + str(round(CG_para2[2],3)) + ' +/- ' + str(round(perr[2],2)))
    boundary, minaxis, maxaxis = multivariate.throws(CG_para2, CG_para2cov, 800, bins2)
    print 'covariance'
    print CG_para2cov
    y_CG2 = schechter2.log_schechter(bins2, *CG_para2)
    LCONDl = Schechter(LCOtot, 19, 8, bins)
    LCONDu = Schechter(LCOtot, 20, 8, bins)
    a = np.zeros((len(LCOSch[1]),3))
    a[:,0] = LCOSch[1]
    a[:,1] = LCONDl[1]
    a[:,2] = LCONDu[1]
    np.savetxt('fullerrs.txt',a)
    LCOdetschech = Schechter(LCOdet, 10, 8, bins)
    LCOdetschechl = Schechter(LCOdet, 19, 8, bins)
    LCOdetschechu = Schechter(LCOdet, 20, 8, bins)
    CG_para3, CG_para2cov3 = schechter2.log_schechter_fit(LCOdetschech[2][6:14], LCOdetschech[1][6:14])
    perr = np.sqrt(np.diag(CG_para2cov3))
    print ('phi_star = ' + str(round(CG_para3[0],2)) + ' +/- ' + str(round(perr[0],2)))
    print ('L_0 = ' + str(round(CG_para3[1],2)) + ' +/- ' + str(round(perr[1],2)))
    print ('alpha = ' + str(round(CG_para3[2],3)) + ' +/- ' + str(round(perr[2],2)))
    boundary3, minaxis3, maxaxis3 = multivariate.throws(CG_para3, CG_para2cov3, 800, bins2)
    y_CG3 = schechter2.log_schechter(bins2, *CG_para3)
    sampling = errors(LCOdet, bins, LCOdetschech[1])
    samplingF = errors(LCO, bins, LCOSch[1])
    l = [0,0,0,0,0.1760912591,0.2041199827,0.1375917444,0.0137869414,0.0934900541,0,0.1259633184,0.2688453123,0.1461280357,0,0,0,0]
    u = [0,0,0,0,0.3010299957,0.3979400087,0,0.044056824,0,0.0626094827,0.1443927751,0.1035405919,0,0.4771212547,0,0,0]
    l2 =[0,0,0,0,0.3010299957,0.1918855262,0.1365819998,0.0390612296,0.2402860609,0,0.1259633184,0.2108533653,0.1461280357,0,0,0,0]
    u2 =[0,0,0,0.4771212547,0.8293037728,0.1759744873,0.0193671767,0.0026500336,0.00662909,0.0763901687,0.1345726324,0.0901766303,0,0.4771212547,0,0,0]
    lsx, lsy = np.append(LCOSch[2][7:9], LCOSch[2][10:13]), np.append(LCOSch[1][7:9], LCOSch[1][10:13]) - np.append(l[7:9], l[10:13])
    usx, usy = np.append(LCOSch[2][9], LCOSch[2][11:14]), np.append(LCOSch[1][9], LCOSch[1][11:14]) - np.append(l[9], l[11:14])
    print ('lsx', 'lsy', lsx, lsy)
    CG_para_low, CG_para_lowcov = schechter2.log_schechter_fit(lsx, lsy)
    CG_para_up, CG_para_upcov = schechter2.log_schechter_fit(usx, usy)
    for i in range(0,len(l)):
        l[i] = np.sqrt((l[i]**2)+(sampling[i]**2))
        u[i] = np.sqrt((u[i]**2)+(sampling[i]**2))
        l2[i] = np.sqrt((l2[i]**2)+(samplingF[i]**2))
        u2[i] = np.sqrt((u2[i]**2)+(samplingF[i]**2))
    # err = np.zeros((len(LCOdetschech[1]),5))
    # err[:,0] = LCOdetschech[1]
    # err[:,1] = LCOdetschechl[1]
    # err[:,2] = LCOdetschechu[1]
    # np.savetxt('errschec.txt', err)
    # for i in range(0, len(err)):
    #     bounds = np.array([err[i,1], err[i,2]])
    #     pt = err[i,0]
    #     if abs(pt)>99:
    #         llim, ulim = 0,0
    #     elif max(np.absolute(bounds))>99:
    #         llim, ulim = 0,0
    #     else:
    #         llim = bounds[bounds<pt]
    #         ulim = bounds[bounds>pt]
    #         if len(llim)>1:
    #             llim = min(llim)
    #             ulim = 0
    #         # if len(ulim)>1:
    #         #     llim = 0
    #         #     ulim = max(ulim)
    #     print llim, ulim
    #     err[i,3] = llim
    #     err[i,4] = ulim
    # print err

    xmajorLocator   = MultipleLocator(1)
    xminorLocator   = MultipleLocator(0.2)
    ymajorLocator   = MultipleLocator(1)
    yminorLocator   = MultipleLocator(0.2)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].tick_params(axis='both', which='major', labelsize=15)
    ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[0,0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[0,0].xaxis.set_major_locator(xmajorLocator)
    ax[0,0].xaxis.set_minor_locator(xminorLocator)
    ax[0,0].yaxis.set_major_locator(ymajorLocator)
    ax[0,0].yaxis.set_minor_locator(yminorLocator)
    # ax[0,0].errorbar(LCOdetschechl[2], LCOdetschechl[1], alpha = 0.1, fmt='s', markersize = 12, linewidth=2, markeredgewidth=2, capthick=3, mfc='r', mec='navy' , label='det')
    # ax[0,0].errorbar(LCOdetschechu[2], LCOdetschechu[1], alpha = 0.1, fmt='s', markersize = 12, linewidth=2, markeredgewidth=2, capthick=3, mfc='b', mec='navy' , label='det')
    ax[0,0].axvline(7.5, color = 'k', linestyle = '--', label = 'xCOLD GASS completeness', zorder = 1)
    ax[0,0].fill_between(bins2, minaxis, maxaxis, color = 'crimson', alpha = 0.3, zorder = 3)
    # ax[0,0].fill_between(bins2, minaxis3, maxaxis3, color = 'navy', alpha = 0.3, zorder = 4)
    # ax[0,0].axvline(LCOSch[2][6], color = 'r', linestyle = '-')
    # ax[0,0].axvline(LCOSch[2][14], color = 'r', linestyle = '-')
    # ax[0,0].fill_between(lowerr2[:,0], lowerr2[:,1], uperr2[:,1], color = 'k', alpha = 0.2,  zorder = 2)
    ax[0,0].plot(data[:,0], data[:,1], label = 'Vallini+16', color = 'k', linewidth=1, zorder = 2, linestyle='-')
    ax[0,0].errorbar(keres[:,0], keres[:,1], yerr=[keres[:,3], keres[:,2]], fmt='o', markersize = 10, linewidth=2, markeredgewidth=2, capthick=3, zorder = 7, mfc='darkgray', mec='gray', ecolor = 'gray', label='Keres+03')
    # ax[0,0].plot(keresB['lgCOB'], keresB['lgdndlgLCOB'], label = 'KeresLagos', color = 'green', linewidth=3)

    ax[0,0].plot(bins2, y_CG2, label = 'xCOLD GASS fit det+non-det', color = 'crimson', linestyle='-', alpha = 1, linewidth=1, zorder = 5)
    ax[0,0].plot(bins2, y_CG3, label = 'xCOLD GASS fit det', color = 'navy', linestyle='-', alpha = 1, linewidth=1, zorder = 6)

    # ax[0,0].fill_between(bins3, y_CG_low2, y_CG_up2, color = 'g', alpha = 0.3, edgecolor="g")
    # ax[0,0].fill_between(bins4, y_CG_low1, y_CG_up1, color = 'none', hatch = '//', edgecolor="g")
    # ax[0,0].scatter(LCOSch[2][6:12], LCOSch[1][6:12] - l[6:12], color = 'g')
    # ax[0,0].scatter(LCOSch[2][6:13], LCOSch[1][6:13] + u[6:13], color = 'r')
    # ax[0,0].plot(uperr[:,0], uperr[:,1], label = 'upper limit', color = 'crimson')
    # ax[0,0].plot(lowerr[:,0], lowerr[:,1], label = 'lower limit', color = 'crimson')
    # ax[0,0].scatter(lowerr2[:,0], lowerr2[:,1], label = 'uperr', color = 'g')
    # ax[0,0].scatter(uperr2[:,0], uperr2[:,1], label = 'uperr', color = 'g')

    ax[0,0].errorbar(LCOSch[2], LCOSch[1], yerr=[l2,u2], fmt='^', markersize = 10, linewidth=2, markeredgewidth=2, capthick=3, mfc='r', mec='crimson', ecolor = 'crimson', label='xCOLD GASS det+non-det', zorder = 9)
    ax[0,0].errorbar(LCOdetschech[2], LCOdetschech[1], yerr=[l,u], fmt='h', markersize = 10, linewidth=2, markeredgewidth=2, capthick=3, mfc='blue', mec='navy', ecolor='navy', label='xCOLD GASS det only', zorder = 8)

    ax[0,0].set_xlabel(r'$\mathrm{log\, L^{\prime}_{CO}\, [K \, km \,s^{-1}\, pc^{2}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, \phi\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
    ax[0,0].set_xlim(5.5, 11)
    ax[0,0].set_ylim(-7, -1)
    handles,labels = ax[0,0].get_legend_handles_labels()
    print 'cg labels', labels
    # handles = [handles[3], handles[1], handles[0], handles[5], handles[4], handles[2]]
    # labels = [labels[3], labels[1], labels[0], labels[5], labels[4], labels[2]]
    # plt.legend(handles, labels, fontsize = 13, loc=3)
    plt.legend(fontsize = 12, loc=3)
    plt.tight_layout()
    plt.savefig('img/schechter/luminosity.pdf', format='pdf', dpi=250, transparent = True)
    newdf = pd.DataFrame()
    newdf['x'] = LCOSch[2]
    newdf['y'] = LCOSch[1]
    newdf['dyu'] = u2
    newdf['dyd'] = l2
    newdf.to_csv('luminosity.txt', sep='\t')
    np.savetxt('covariance.txt', CG_para2cov)
    np.savetxt('mean.txt', CG_para2)
    print (CG_para2)

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
uperr2 = extrapolate(uperr, 5.5, 11, 200)


Full = atpy.Table('data/COLDGASS_full.fits')
#0'z', 1'flag', 2'MH2', 3'limMH2', 4'MH2both', 5'D_L', 6'M*', 7'V/Vm', 8'Vm', 9'Weight', 10'LCO'
LND, HND, FullData, weights, LCO, LCOdet, gdata, LCOSch, LCOtot = GetFull(Full, output)
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

keresB = read_Lagos_data('data/Keres03_LCOLFBband.txt')
keres60 = read_Lagos_data('data/Keres03_LCOLF60m.txt')





PlotLum(data, uperr, lowerr, keres, lowerr2, uperr2, LCOSch, LCOdet, LCO, LCOtot, keresB, keres60)
