import pandas as pd
import numpy as np
import atpy
import matplotlib.pyplot as plt

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
    HMass['logSFRu'] = np.log10(HMass['SFRu'])
    HMass['logSFR_UVTIR'] = HMass['logSFRu']*0.92 + 0.014
    HMass.to_csv('data/test.csv')
    return HMass

def Plot1(HMass):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(HMass['SFR_Best'], HMass['logSFR_UVTIR'], marker = 'o', s = 10, edgecolor='red', linewidth='2', facecolor='red', label = 'Low Mass')
    ax[0,0].set_xlabel(r'$\mathrm{log\, SFR_{Best}\, [M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, SFR_{UV+TIR}\, [M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-1.5, 2.0)
    ax[0,0].set_xlim(-1.5, 2.0)
    plt.savefig('img/scal/sfr_best_UVTIR.pdf', format='pdf', dpi=250, transparent = False)

def Plot2(HMass):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(HMass['logSFRu'], HMass['logSFR_UVTIR'], marker = 'o', s = 10, edgecolor='blue', linewidth='2', facecolor='blue', label = 'Low Mass')
    ax[0,0].set_xlabel(r'$\mathrm{log\, SFR_{u} \,[M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, SFR_{UV+TIR}\, [M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-1.5, 2.0)
    ax[0,0].set_xlim(-1.5, 2.0)
    plt.savefig('img/scal/sfr_u_UVTIR.pdf', format='pdf', dpi=250, transparent = False)

def Plot3(HMass):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].scatter(HMass['SFR_Best'], HMass['logSFRu'], marker = 'o', s = 10, edgecolor='blue', linewidth='2', facecolor='blue', label = 'Low Mass')
    ax[0,0].set_xlabel(r'$\mathrm{log\, SFR_{Best} \,[M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\mathrm{log\, SFR_{u}\, [M_{\odot}\,yr^{-1}]}$', fontsize=18)
    ax[0,0].set_ylim(-1.5, 2.0)
    ax[0,0].set_xlim(-1.5, 2.0)
    plt.savefig('img/scal/sfr_best_u.pdf', format='pdf', dpi=250, transparent = False)
HMass = cf_best_u()
Plot1(HMass)
Plot2(HMass)
Plot3(HMass)
