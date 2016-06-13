import numpy as np
import atpy
import math
from scipy import integrate
import matplotlib.pyplot as plt
import asciidata
from scipy.optimize import curve_fit
import schechter
import csv
import pandas as pd

# Function to calculate the luminosity distance from z #########################
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

# Remove non-detections ########################################################
def NonDetect(data, flagrow):
    init = True
    newdata = np.zeros((1,np.shape(data)[1]))
    for i in range (0,len(data)):
        if data[i,flagrow] == 1.0:
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
def H2Conversion(data, Zindex, LCOindex):
    # alpha_CO = mZ + c (from Genzel et al)
    c = 12.0
    dc = 2.0
    m = -1.3
    dm = 0.26
    H2mass = np.zeros((len(data),3))
    for i in range(0,len(data)):
        if math.isnan(data[i,Zindex]) == True:
            alpha_CO_gal = 4.35
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
    data = np.hstack((data,H2mass))
    return data

# Vm calc ######################################################################
def Vm(data, Dlaxis, minz, maxz, N_COLDGASS):
    Omega = 0.483979888662
    N_SDSS = 12770.0
    VVmlist = np.zeros((len(data),1))
    Vmlist = np.zeros((len(data),1))
    x,y = np.zeros((1,1)), np.zeros((1,1))
    x[0,0] = minz
    y[0,0] = maxz
    D_in = float(lumdistance(x,0)[0,1])
    D_out = float(lumdistance(y,0)[0,1])
    V_in = ((4*math.pi)/3)*D_in*D_in*D_in
    V_out = ((4*math.pi)/3)*D_out*D_out*D_out
    Vm =  (N_COLDGASS/N_SDSS)*(V_out - V_in)*Omega
    for i in range(0,len(data)):
        Dl = data[i,Dlaxis]
        V = ((4*math.pi)/3)*Dl*Dl*Dl
        VVmlist[i,0] = (V/Vm)
        Vmlist[i,0] = Vm
    data = np.hstack((data,VVmlist))
    data = np.hstack((data,Vmlist))
    return data

# schechter bins ###############################################################
def Schechter(data, LCOaxis, Vmaxis, number):
    l = data[:,LCOaxis]
    l = np.log10(l)
    low, high = min(l), max(l)     # min max in logspace
    bins = np.linspace(low, high, num=number) # log-spaced bins
    rho, N, xbins = [], [], []
    for i in range (1,len(bins)):
        p, Num = 0, 0
        for j in range(0,len(l)):
            if l[j] >= bins[i-1] and l[j] < bins[i]:
                p += 1/data[j,Vmaxis]
                Num+=1
        N.append(Num)
        xbins.append((bins[i]+bins[i-1])/2)
        rho.append(p)
    # return the Number of gals, log10(density), centre pt of each bin
    #print np.sum(N), len(l)
    return N, np.log10(rho), xbins

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

## Read data from tables #######################################################
highM = atpy.Table('COLDGASS_DR3_with_Z.fits')
lowM = asciidata.open('COLDGASS_LOW_29Sep15.ascii')
SAMI = asciidata.open('SAMI_IRAM_data.txt')
# Sort Data ####################################################################
# def dict for indices #########################################################
l = {'S_CO':11, 'z':3, 'M*':4, 'Zo':5, 'SFR':6, 'flag':15, 'NUV-r': 8}
h = {'S_CO':16, 'z':4, 'M*':5, 'Zo':12, 'SFR':7, 'flag':21, 'NUV-r': 10}
output = {  'S_CO':0, 'z':1, 'flag':2, 'M*':3, 'Zo':4, 'SFR':5, 'sSFR':6,
            'NUV-r':7,'D_L':8, 'V/Vm':9, 'Vm':10, 'L_CO':11, 'AlphaCO':12,
            'MH2':13, 'dalpha':14}
# New Algo #####################################################################
HMass = np.zeros((len(highM),8))
LMass = np.zeros((len(lowM[12]),8))
# High Mass Galaxies
for i,rows in enumerate(highM):
    HMass[i,output['S_CO']] = rows[h['S_CO']]                                   # S_CO
    HMass[i,output['z']] = rows[h['z']]                                         # z
    HMass[i,output['flag']] = rows[h['flag']]                                   # flag
    HMass[i,output['M*']] = rows[h['M*']]                                       # Mgal
    HMass[i,output['Zo']] = rows[h['Zo']]                                       # Zo
    HMass[i,output['SFR']] = rows[h['SFR']]                                     # SFR
    HMass[i,output['sSFR']] = np.log10(HMass[i,output['SFR']]) - HMass[i,output['M*']]      # NUV-r
    HMass[i,output['NUV-r']] = rows[h['NUV-r']]      # sSFR

# Low Mass Galaxies
LMass[:,output['S_CO']] = list(lowM[l['S_CO']])                         # S_CO
LMass[:,output['z']] = list(lowM[l['z']])                               # z
LMass[:,output['flag']] = list(lowM[l['flag']])                         # flag
LMass[:,output['M*']] = list(lowM[l['M*']])                             # Mgal
LMass[:,output['Zo']] = list(lowM[l['Zo']])                             # Zo
LMass[:,output['SFR']] = list(lowM[l['SFR']])                           # SFR
sSFRlist = sSFR(list(lowM[l['SFR']]), list(lowM[l['M*']]))
LMass[:,output['sSFR']] = sSFRlist                                      # sSFR
LMass[:,output['NUV-r']] = list(lowM[l['NUV-r']])      # NUV-r

M = np.append(HMass[:,output['M*']], LMass[:,output['M*']])
SFR = np.log10(np.append(HMass[:,output['SFR']], LMass[:,output['SFR']]))
# plt.plot(M, SFR, 'ro')

# Attach Pre-caclulate L_CO to Low-Mass dataset
#cL_CO = np.zeros((len(list(lowM[12])),1))
#cL_CO[:,0] = list(lowM[12])
#lumsnew = np.concatenate((LMass,cL_CO),axis=1)   # [Lmass, L_CO]
# Remove non-detections from all samples
LMass = NonDetect(LMass, output['flag'])
HMass = NonDetect(HMass, output['flag'])
#lumsnew = NonDetect(lumsnew, output['flag'])

# Calculate Luminosity distance for each galaxy ################################
# | S_CO | z | flag | Mgal | Zo | D_L |
LMass = lumdistance(LMass, output['z'])
HMass = lumdistance(HMass, output['z'])
# | S_CO | z | flag | Mgal | Zo | L_CO | D_L |
#lumsnew = lumdistance(lumsnew, output['z'])

# Calculate Vm #################################################################
# | S_CO | z | flag | Mgal | Zo | D_L | V/Vm | Vm |
LMass = Vm(LMass,output['D_L'], min(LMass[:,output['z']]), max(LMass[:,output['z']]),89)
HMass = Vm(HMass,output['D_L'], min(HMass[:,output['z']]), max(HMass[:,output['z']]),215)
# | S_CO | z | flag | Mgal | Zo | L_CO | D_L | V/Vm | Vm |
#lumsnew = Vm(lumsnew,6, 0.05)

# Calculate Luminosity Values ##################################################
# | S_CO | z | flag | Mgal | Zo | D_L | V/Vm | Vm | L_CO |
LMass = lCalc(LMass,output['S_CO'],output['z'],output['D_L'],True)
HMass = lCalc(HMass,output['S_CO'],output['z'],output['D_L'],False)
# | S_CO | z | flag | Mgal | Zo | L_CO | D_L | V/Vm | Vm | L_CO |
#lumsnew = lCalc(lumsnew,0,1,6,True)

# Calculate MH2 ################################################################
# | S_CO | z | flag | Mgal | Zo | D_L | V/Vm | Vm | L_CO | AlphaCO | MH2 | dalpha |
LMass = H2Conversion(LMass, output['Zo'], output['L_CO'])
HMass = H2Conversion(HMass, output['Zo'], output['L_CO'])
Mass = np.append(LMass[:,output['M*']], HMass[:,output['M*']])
alpha = np.append(LMass[:,output['AlphaCO']], HMass[:,output['AlphaCO']])
alphaerror = np.append(LMass[:,output['dalpha']], HMass[:,output['dalpha']])
MH2 = np.append(LMass[:,output['MH2']], HMass[:,output['MH2']])
NH2, xH2 = sortIntoBins(MH2, 30)
NH2L ,xH2L = sortIntoBins(LMass[:,output['MH2']], 15)
NH2H ,xH2H = sortIntoBins(HMass[:,output['MH2']], 15)

################################################################################
lumsL = LMass[:,output['L_CO']]
lumsH = HMass[:,output['L_CO']]
#lumsnew = lumsnew[:,5]

lumsL = [i for i in lumsL if i > 0.0]         # remove 0 detected CO flux galaxies
lumsH = [i for i in lumsH if i > 0.0]         # remove 0 detected CO flux galaxies
lumsLlog, lumsHlog = np.log10(lumsL), np.log10(lumsH)
lCombinedlog = np.append(lumsLlog, lumsHlog)
lCombined = np.append(lumsL, lumsH)

# Sort Luminosity Values into bins #############################################
NL, midL = sortIntoBins(lumsLlog, 15)
NH, midH = sortIntoBins(lumsHlog, 15)
NC, midC = sortIntoBins(lCombinedlog, 20)
#NR, midR = sortIntoBins(lumsnew, 15)

# Calculations for Mass Distribution ###########################################

totalMass = np.append(LMass[:,output['M*']], HMass[:,output['M*']])
Nmass, Xmass = sortIntoBins(totalMass, 30)

# density schechter ############################################################
total = np.vstack((LMass, HMass))
N, rho, xbins = Schechter(total, output['L_CO'], output['Vm'], 20)
Nh2, rhoh2, xbinsh2 = Schechter(total, output['MH2'], output['Vm'], 12)
print total[:,output['L_CO']]
# fit schechter ################################################################
x1,x2 = xbins, xbins[4:]
y1,y2 = rho,rho[4:]
popt1 = schechter.log_schechter_fit(x1, y1)
phi1, L01, alpha1 = popt1
popt2 = schechter.log_schechter_fit(x2, y2)
phi2, L02, alpha2 = popt2
poptkeres = np.log10(0.00072), np.log10(9.8*math.pow(10,6)), -1.3
#print popt1
xnew = np.linspace(max(xbins),min(xbins),100)
ynew1 = schechter.log_schechter(xnew, *popt1)
ynew2 = schechter.log_schechter(xnew, *popt2)
ykeres = schechter.log_schechter(xnew, *poptkeres)

# Keres fit
mst=np.log10((2.81*(10**9))/(0.7**2))
alpha=-1.18
phist=np.log10(0.0089*(0.7**3))
xkeres = np.linspace(8,10.5,200)
ykeres = schechter.log_schechter(xkeres, phist, mst, alpha)


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
fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False)
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
# # schechter only ###############################################################
ax[0,0].plot(xbinsh2, rhoh2, 'bo', label = 'COLD GASS')
ax[0,0].plot(xkeres, ykeres, 'r', label = 'Keres+03')
# ax[0,0].plot(xbins[4:], rho[4:], 'ro', alpha=0.5)
# ax[0,0].plot(xnew, ynew1, 'b-')
# ax[0,0].plot(xnew, ynew2, 'r-')
# ax[0,0].plot(xnew, ykeres, 'g-')
ax[0,0].set_xlabel(r'$\mathrm{log\, M_{H2}\,[M_{sun}]}$', fontsize=18)
ax[0,0].set_ylabel(r'$\mathrm{log\, \phi_{H2}\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
ax[0,0].set_ylim(-5, -1)
# ax[0,0].set_xlim(8, 10.5)
#ax[0,1].set_title('Schechter', fontsize=20)
# ax[0,0].text(9, -5.1, (r'$\phi_{*}$ = '+str(round(phi1,2))+'\n'+ r'$L_{*}$ = '+str(round(L01,2))+'\n'+ r'$\alpha$ = '+str(round(alpha1,2))), fontsize=18, color='b')
# ax[0,0].text(9, -5.8, (r'$\phi_{*}$ = '+str(round(phi2,2))+'\n'+ r'$L_{*}$ = '+str(round(L02,2))+'\n'+ r'$\alpha$ = '+str(round(alpha2,2))), fontsize=18, color='r')
# plt.savefig('img/lum.png', dpi=1000, transparent = False)
# fig.set_size_inches(10,6)
# plt.savefig('schechter.png', transparent = False ,dpi=250)
plt.legend()
plt.show()
plt.savefig('img/MH2.png', transparent = False ,dpi=250)
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
