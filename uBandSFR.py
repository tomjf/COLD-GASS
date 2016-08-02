import pandas as pd
import numpy as np
import atpy
import matplotlib.pyplot as plt
import math
from scipy import integrate

def cf_best_u():
    df = pd.read_csv('data/U_band_t1.csv')
    HiMass = df[['GASS', 'mpa_z', 'model_u', 'model_g', 'ext_u', 'ext_g']]
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Full = atpy.Table('data/COLDGASS_full.fits')
    FullData = np.zeros((len(Full),3))
    lorh = []
    for i,rows in enumerate(Full):
        #GASS|SFR_BEST|
        FullData[i,0] = rows[0]
        FullData[i,1] = np.log10(rows[24])
        FullData[i,2] = rows[10] #lumdist
        lorh.append(rows[1])
    data = pd.DataFrame({   'GASS': FullData[:,0],
                            'SFR_Best': FullData[:,1],
                            'D_L': FullData[:,2]})
    data['group'] = lorh
    HMass = (data.loc[data['group'] == 'H'])
    HMass['mpa_z'], HMass['model_u'], HMass['model_g'], HMass['ext_u'], HMass['ext_g'] = np.nan, np.nan, np.nan, np.nan, np.nan
    for index, row in HMass.iterrows():
        for index1, row1 in HiMass.iterrows():
            if row['GASS'] == row1['GASS']:
                HMass.set_value(index, 'mpa_z', HiMass.at[index1, 'mpa_z'])
                HMass.set_value(index, 'model_u', HiMass.at[index1, 'model_u'])
                HMass.set_value(index,'model_g', HiMass.at[index1, 'model_g'])
                HMass.set_value(index, 'ext_u', HiMass.at[index1, 'ext_u'])
                HMass.set_value(index, 'ext_g', HiMass.at[index1, 'ext_g'])
    HMass['u'] = HMass['model_u'] - HMass['ext_u']
    HMass['g'] = HMass['model_g'] - HMass['ext_g']
    HMass['u-g'] = HMass['u'] - HMass['g']
    HMass['C'] = (-1.1*HMass['u-g']) + 1.6
    HMass['absM'] = HMass['u'] - 5*np.log10((HMass['D_L']*1000000)/10)
    HMass['L/Lstar'] = 10**(0.4*(5.61-HMass['absM']))
    HMass['L'] = HMass['L/Lstar']*((1.86*(10**25))/(1.537*(10**14)))
    HMass['Lcorr'] = HMass['L']*HMass['C']
    HMass['SFRu'] = ((HMass['L']/(1.81*(10**21)))**1.186)*1.53
    print max(HMass[['SFRu']].values), min(HMass[['SFRu']].values)
    HMass['logSFRu'] = np.log10(HMass['SFRu'])
    HMass['logSFR_UVTIR'] = HMass['logSFRu']*0.92 + 0.014
    HMass['f_lambda_gal'] = (3.67*(10**(-9)))/(10**(0.4*HMass['u']))
    HMass['f_nu_gal'] = ((3585.0**2.0)/(3.0*(10**18)))*HMass['f_lambda_gal']
    HMass['Lv'] = (4*np.pi*((HMass['D_L']*3.086*(10**24))**2)*HMass['f_nu_gal'])*(10**(-7))
    HMass['SFRuL'] = ((HMass['Lv']/(1.81*(10**21)))**1.186)*1.53
    logSFR_U = np.zeros((len(HMass['SFRuL']),1))
    for i in range(0, len(HMass['SFRuL'])):
        logSFR_U[i] = np.log10(HMass['SFRuL'].values[i])
    HMass['logSFR_U'] = logSFR_U
    HMass['logUVTIR'] = HMass['logSFR_U']*0.92 + 0.014
    HMass.to_csv('data/test.csv')
    return HMass

def cf_best_u2():
    df = pd.read_csv('data/U_band_t1.csv')
    HiMass = df[['GASS', 'mpa_z', 'model_u', 'model_g', 'ext_u', 'ext_g']]
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Full = pd.read_csv('data/allGASS_SFRbest_simple_t1.csv')
    HMass = Full[['GASS', 'SFR_best']]
    HMass['SFR_Best'] = np.log10(HMass['SFR_best'])
    HMass['mpa_z'], HMass['model_u'], HMass['model_g'], HMass['ext_u'], HMass['ext_g'] = np.nan, np.nan, np.nan, np.nan, np.nan
    for index, row in HMass.iterrows():
        for index1, row1 in HiMass.iterrows():
            if row['GASS'] == row1['GASS']:
                print index
                HMass.set_value(index, 'mpa_z', HiMass.at[index1, 'mpa_z'])
                HMass.set_value(index, 'model_u', HiMass.at[index1, 'model_u'])
                HMass.set_value(index,'model_g', HiMass.at[index1, 'model_g'])
                HMass.set_value(index, 'ext_u', HiMass.at[index1, 'ext_u'])
                HMass.set_value(index, 'ext_g', HiMass.at[index1, 'ext_g'])
    HMass['D_L'] = lumdistance(HMass[['mpa_z']].values, 0)[:,1]
    HMass['u'] = HMass['model_u'] - HMass['ext_u']
    HMass['g'] = HMass['model_g'] - HMass['ext_g']
    HMass['u-g'] = HMass['u'] - HMass['g']
    HMass['C'] = (-1.1*HMass['u-g']) + 1.6
    HMass['absM'] = HMass['u'] - 5*np.log10((HMass['D_L']*1000000)/10)
    HMass['L/Lstar'] = 10**(0.4*(5.61-HMass['absM']))
    HMass['L'] = HMass['L/Lstar']*((1.86*(10**25))/(1.537*(10**14)))
    HMass['Lcorr'] = HMass['L']*HMass['C']
    HMass['SFRu'] = ((HMass['L']/(1.81*(10**21)))**1.186)*1.53
    print max(HMass[['SFRu']].values), min(HMass[['SFRu']].values)
    HMass['logSFRu'] = np.log10(HMass['SFRu'])
    HMass['logSFR_UVTIR'] = HMass['logSFRu']*0.92 + 0.014
    HMass['f_lambda_gal'] = (3.67*(10**(-9)))/(10**(0.4*HMass['u']))
    HMass['f_nu_gal'] = ((3585.0**2.0)/(3.0*(10**18)))*HMass['f_lambda_gal']
    HMass['Lv'] = (4*np.pi*((HMass['D_L']*3.086*(10**24))**2)*HMass['f_nu_gal'])*(10**(-7))
    HMass['SFRuL'] = ((HMass['Lv']/(1.81*(10**21)))**1.186)*1.53
    logSFR_U = np.zeros((len(HMass['SFRuL']),1))
    for i in range(0, len(HMass['SFRuL'])):
        logSFR_U[i] = np.log10(HMass['SFRuL'].values[i])
    HMass['logSFR_U'] = logSFR_U
    HMass['logUVTIR'] = HMass['logSFR_U']*0.92 + 0.014
    HMass.to_csv('data/test2.csv')
    return HMass

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

def Plot1(HMass):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(HMass['SFR_Best'], HMass['logUVTIR'], marker = 'o', s = 10, edgecolor='red', linewidth='2', facecolor='red', label = 'Low Mass')
    ax[0,0].scatter(HMass['SFR_Best'], HMass['logSFR_UVTIR'], marker = 'o', s = 10, edgecolor='green', linewidth='2', facecolor='green', label = 'Low Mass')
    ax[0,0].set_xlabel(r'$\mathrm{log\, SFR_{Best}\, [M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, SFR_{UV+TIR}\, [M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-1.5, 2.0)
    ax[0,0].set_xlim(-1.5, 2.0)
    plt.savefig('img/scal/sfr_best_UVTIR.pdf', format='pdf', dpi=250, transparent = False)

def Plot2(HMass):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(HMass['logSFR_U'], HMass['logUVTIR'], marker = 'o', s = 10, edgecolor='blue', linewidth='2', facecolor='blue', label = 'Low Mass')
    ax[0,0].set_xlabel(r'$\mathrm{log\, SFR_{u} \,[M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, SFR_{UV+TIR}\, [M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-1.5, 2.0)
    ax[0,0].set_xlim(-1.5, 2.0)
    plt.savefig('img/scal/sfr_u_UVTIR.pdf', format='pdf', dpi=250, transparent = False)

def Plot3(HMass):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(HMass['SFR_Best'], HMass['logSFR_U'], marker = 'o', s = 10, edgecolor='blue', linewidth='2', facecolor='blue', label = 'Low Mass')
    ax[0,0].set_xlabel(r'$\mathrm{log\, SFR_{Best} \,[M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, SFR_{u}\, [M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-1.5, 2.0)
    ax[0,0].set_xlim(-1.5, 2.0)
    plt.savefig('img/scal/sfr_best_u.pdf', format='pdf', dpi=250, transparent = False)

def Plot4(HMass):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(HMass['u-g'], (HMass['Lv']), marker = 'o', s = 10, edgecolor='blue', linewidth='2', facecolor='blue', label = 'Low Mass')
    ax[0,0].set_xlabel(r'$\mathrm{u-g}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\,L}$', fontsize=18)
    ax[0,0].vlines(0,20,0.55, color='k')
    # ax[0,0].set_ylim(-1.5, 2.0)
    ax[0,0].set_xlim(0.5, 2.3)
    plt.savefig('img/scal/uminusg.pdf', format='pdf', dpi=250, transparent = False)

data = pd.read_csv('data/test.csv')
# data = cf_best_u2()
Plot1(data)
Plot2(data)
Plot3(data)
Plot4(data)
