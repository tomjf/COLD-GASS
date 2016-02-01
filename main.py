import numpy as np
import atpy
import math
from scipy import integrate
import matplotlib.pyplot as plt
import asciidata
from scipy.optimize import curve_fit
import schechter

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
        H2mass[i,1] = np.log10(alpha_CO_gal*data[i,LCOindex])
        H2mass[i,2] = dalpha
    data = np.hstack((data,H2mass))
    return data

# Vm calc ######################################################################
def Vm(data, Dlaxis, maxz):
    VVmlist = np.zeros((len(data),1))
    Vmlist = np.zeros((len(data),1))
    x = np.zeros((1,1))
    x[0,0] = maxz
    Dm = lumdistance(x,0)[0,1]
    for i in range(0,len(data)):
        Dl = data[i,Dlaxis]
        Vm = ((4*math.pi)/3)*Dm*Dm*Dm
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
        #N.append(Num)
        #xbins.append(bins[i-1])
        #xbins.append(bins[i])
        xbins.append((bins[i]+bins[i-1])/2)
        #rho.append(p)
        rho.append(p)
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

## Read data from tables #######################################################
highM = atpy.Table('COLDGASS_DR3_with_Z.fits')
lowM = asciidata.open('COLDGASS_LOW_29Sep15.ascii')
SAMI = asciidata.open('SAMI_IRAM_data.txt')
# Sort Data ####################################################################
# def dict for indices #########################################################
l = {'S_CO':11, 'z':3, 'M*':4, 'Zo':5, 'SFR':6, 'flag':15}
h = {'S_CO':16, 'z':4, 'M*':5, 'Zo':12, 'SFR':7, 'flag':21}
output = { 'S_CO':0, 'z':1, 'flag':2, 'M*':3, 'Zo':4, 'SFR':5, 'sSFR':6, 'D_L':7, 'V/Vm':8, 'Vm':9, 'L_CO':10, 'AlphaCO':11, 'MH2':12, 'dalpha':13}
# New Algo #####################################################################
HMass = np.zeros((len(highM),7))
LMass = np.zeros((len(lowM[12]),7))
# High Mass Galaxies
for i,rows in enumerate(highM):
    HMass[i,output['S_CO']] = rows[h['S_CO']]                                   # S_CO
    HMass[i,output['z']] = rows[h['z']]                                         # z
    HMass[i,output['flag']] = rows[h['flag']]                                   # flag
    HMass[i,output['M*']] = rows[h['M*']]                                       # Mgal
    HMass[i,output['Zo']] = rows[h['Zo']]                                       # Zo
    HMass[i,output['SFR']] = rows[h['SFR']]                                     # SFR
    HMass[i,output['sSFR']] = np.log10(HMass[i,output['SFR']]) - HMass[i,output['M*']]      # sSFR

# Low Mass Galaxies
LMass[:,output['S_CO']] = list(lowM[l['S_CO']])                         # S_CO
LMass[:,output['z']] = list(lowM[l['z']])                               # z
LMass[:,output['flag']] = list(lowM[l['flag']])                         # flag
LMass[:,output['M*']] = list(lowM[l['M*']])                             # Mgal
LMass[:,output['Zo']] = list(lowM[l['Zo']])                             # Zo
LMass[:,output['SFR']] = list(lowM[l['SFR']])                           # SFR
sSFRlist = sSFR(list(lowM[l['SFR']]), list(lowM[l['M*']]))
LMass[:,output['sSFR']] = sSFRlist                                      # sSFR
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
LMass = Vm(LMass,output['D_L'], 0.02)
HMass = Vm(HMass,output['D_L'], 0.05)
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

# fit schechter ################################################################
x1,x2 = xbins, xbins[4:]
y1,y2 = rho,rho[4:]
popt1 = schechter.log_schechter_fit(x1, y1)
phi1, L01, alpha1 = popt1
popt2 = schechter.log_schechter_fit(x2, y2)
phi2, L02, alpha2 = popt2
xnew = np.linspace(max(xbins),min(xbins),100)
ynew1 = schechter.log_schechter(xnew, *popt1)
ynew2 = schechter.log_schechter(xnew, *popt2)

# gas fractions ################################################################
SAMI_outflows = [   567624,574200,228432,239249,31452,238125,486834,
                    417678,106389,593680,618906,618220,383259,209807,376121]

SAMI_SFR = [        0.39, 0.65, 0.56, 0.24, 0.72, 0.28, 0.51, 0.04, 0.84, 0.16, 0.46,
                    1.14, 2.30, 3.66, 2.02]

SAMI_data = np.zeros((len(SAMI_outflows),8))
for i in range(0, len(SAMI[0])):
    if SAMI[0][i] in SAMI_outflows:
        SAMI_data[i,0] = SAMI_outflows[i] # GAMA ID
        SAMI_data[i,1] = SAMI[2][i] # Mgal
        SAMI_data[i,2] = SAMI[6][i] # MH2
        SAMI_data[i,3] = SAMI[8][i] # flag
        SAMI_data[i,4] = np.log10(SAMI[7][i]) # amelie's calc gas fraction
        SAMI_data[i,5] = fgas(SAMI_data[i,2],SAMI_data[i,1]) #my calc
        SAMI_data[i,6] = SAMI_SFR[i]
        SAMI_data[i,7] = np.log10(SAMI_data[i,6]) - SAMI_data[i,1]
fgasL, fgasH = [], []
for i in range (len(LMass)):
    fgasL.append(fgas(LMass[i,output['MH2']], LMass[i,output['M*']]))
for i in range (len(HMass)):
    fgasH.append(fgas(HMass[i,output['MH2']], HMass[i,output['M*']]))
CG_X = np.append(LMass[:,output['M*']], HMass[:,output['M*']])
CG_Y = np.append(fgasL, fgasH)
sSFR_X = np.append(LMass[:,output['sSFR']], HMass[:,output['sSFR']])

################################################################################
order = 'GAMA ID \t M* \t MH2 \t flag \t fgas1 \t fgas2 \t SFR \t sSFR'
np.savetxt('SAMI.txt', SAMI_data, delimiter='\t', fmt= '%1.3f', header = order)

# ###############################################################################
fig, ax = plt.subplots(nrows = 1, ncols = 2, squeeze=False)
ax[0,0].plot(CG_X, CG_Y,'ko', label = 'COLD GASS', alpha=0.2)
ax[0,0].plot(SAMI_data[:,1], SAMI_data[:,5],'ro', label = 'SAMI', markersize = 10)
ax[0,0].set_xlabel(r'$log_{10}(M_{*}/M_{\odot})$', fontsize = 20)
ax[0,0].set_ylabel(r'$log_{10}(M_{H2}/M_{*})$', fontsize = 20)
for i, txt in enumerate(SAMI_outflows):
    print txt
    ax[0,0].annotate(str(txt), (0.03+SAMI_data[i,1],SAMI_data[i,5]))
ax[0,0].legend()

ax[0,1].plot(sSFR_X, CG_Y, 'ko', label = 'COLD GASS', alpha=0.2)
ax[0,1].plot(SAMI_data[:,7], SAMI_data[:,5], 'ro', label = 'SAMI', markersize = 10)
ax[0,1].set_xlabel(r'$log_{10}(sSFR)$', fontsize = 20)
ax[0,1].set_ylabel(r'$log_{10}(M_{H2}/M_{*})$', fontsize = 20)
ax[0,1].legend()
for i, txt in enumerate(SAMI_outflows):
    print txt
    ax[0,1].annotate(str(txt), (0.03+SAMI_data[i,7],SAMI_data[i,5]))
plt.show()

# # Plot Luminosity number plot ################################################
# fig, ax = plt.subplots(nrows = 2, ncols = 2, squeeze=False)
# # ax[0,0].plot(midL,NL,'b-', label = 'Low mass')
# # ax[0,0].plot(midH,NH,'r-', label = 'high mass')
# # ax[0,0].plot(midC,NC,'g-', label = 'lCombined')
# # #ax[0,0].plot(midR,NR,'k-', label = 'Pre-calc')
# # ax[0,0].set_xlabel(r'$log_{10}(L_{CO})$', fontsize=20)
# # ax[0,0].set_ylabel(r'$log_{10}(N)$', fontsize=20)
# # ax[0,0].set_title('CO Luminosity', fontsize=20)
# # ax[0,0].legend()
# # # # ax[0,0].savefig('lum1.png')
#
# # Plot H2 mass #################################################################
# ax[1,0].plot(xH2, NH2,'b-', label = 'H2 Mass')
# #ax[0,0].plot(xH2L,NH2L,'r-', label = 'lowM MH2')
# #ax[0,0].plot(xH2H,NH2H,'g-', label = 'highM MH2')
# ax[1,0].set_xlabel(r'$log_{10}(M_{H2}/M_{\odot})$', fontsize=20)
# ax[1,0].set_ylabel(r'$log_{10}(N_{gal})$', fontsize=20)
# #ax[0,1].set_title('CO Luminosity', fontsize=20)
# ax[1,0].legend(loc=3)
#
# # schechter only ###############################################################
# ax[1,1].plot(xbins, rho, 'bo')
# ax[1,1].plot(xbins[4:], rho[4:], 'ro', alpha=0.5)
# ax[1,1].plot(xnew, ynew1, 'b-')
# ax[1,1].plot(xnew, ynew2, 'r-')
# ax[1,1].set_xlabel(r'$log_{10}(L_{CO})$', fontsize=20)
# ax[1,1].set_ylabel(r'$log_{10}{\rho(L)}$', fontsize=20)
# #ax[0,1].set_title('Schechter', fontsize=20)
# ax[1,1].text(9, -5.1, (r'$\phi_{*}$ = '+str(round(phi1,2))+'\n'+ r'$L_{*}$ = '+str(round(L01,2))+'\n'+ r'$\alpha$ = '+str(round(alpha1,2))), fontsize=18, color='b')
# ax[1,1].text(9, -5.8, (r'$\phi_{*}$ = '+str(round(phi2,2))+'\n'+ r'$L_{*}$ = '+str(round(L02,2))+'\n'+ r'$\alpha$ = '+str(round(alpha2,2))), fontsize=18, color='r')
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

plt.show()
