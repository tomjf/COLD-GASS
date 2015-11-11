import numpy as np
import atpy
import math
from scipy import integrate
import matplotlib.pyplot as plt
import asciidata

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
        #DH = (c*z)/Ho                           # calculate distance from Hubble law for comparison
        Dlvals[i,0] = Dl
    data = np.hstack((data,Dlvals))
    return data

# Calculate CO Luminosity ######################################################
def lCalc(data, SCOaxis, zaxis, Dlaxis, correction):
    lums = np.zeros((len(data),1))
    for i in range(0,len(data)):                              # for each galaxy in the dataset
        if correction == True:
            SCO_cor = data[i,SCOaxis]*6.0                      # find the integrated CO flux
        else:
            SCO_cor = data[i,SCOaxis]
        C = 3.25*math.pow(10,7)                 # numerical const from eqn 4 paper 1
        freq = 111                              # observing frequency
        Dl = data[i,Dlaxis]
        SDSS_z = math.pow((1+data[i,zaxis]),-3)       # redshift component
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
    m = 12.0
    dm = 2.0
    c = -1.3
    dc = 0.26
    H2mass = np.zeros((len(data),1))
    for i in range(0,len(data)):
        alpha_CO_gal = (m*data[i,Zindex]) + c
        H2mass[i,0] = alpha_CO_gal*data[i,LCOindex]
    data = np.hstack((data,H2mass))
    return data

# Vm calc ######################################################################
def Vm(data, Dlaxis):
    VVmlist = np.zeros((len(data),1))
    Vmlist = np.zeros((len(data),1))
    x = np.zeros((1,1))
    x[0,0] = 0.0375
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

## Read data from tables #######################################################
highM = atpy.Table('COLDGASS_DR3.fits')
lowM = asciidata.open('COLDGASS_LOW_29Sep15.ascii')

# Sort Data ####################################################################
HMass = np.zeros((len(highM),4))
LMass = np.zeros((len(lowM[12]),4))
# High Mass Galaxies
for i,rows in enumerate(highM):
    HMass[i,0] = rows[15]       # S_CO
    HMass[i,1] = rows[4]        # z
    HMass[i,2] = rows[20]       # flag
    HMass[i,3] = rows[5]        # Mgal
# Low Mass Galaxies
LMass[:,0] = list(lowM[11])     # S_CO
LMass[:,1] = list(lowM[3])      # z
LMass[:,2] = list(lowM[15])     # flag
LMass[:,3] = list(lowM[4])      # Mgal
# Attach Pre-caclulate L_CO to Low-Mass dataset
cL_CO = np.zeros((len(list(lowM[12])),1))
cL_CO[:,0] = list(lowM[12])
lumsnew = np.concatenate((LMass,cL_CO),axis=1)   # [Lmass, L_CO]
# Remove non-detections from all samples
LMass = NonDetect(LMass, 2)
HMass = NonDetect(HMass, 2)
lumsnew = NonDetect(lumsnew, 2)

# Calculate Luminosity distance for each galaxy ################################
# | S_CO | z | flag | Mgal | D_L |
LMass = lumdistance(LMass, 1)
HMass = lumdistance(HMass, 1)
# | S_CO | z | flag | Mgal | L_CO | D_L |
lumsnew = lumdistance(lumsnew, 1)

# Calculate Vm #################################################################
# | S_CO | z | flag | Mgal | D_L | V/Vm | Vm |
LMass = Vm(LMass,4)
HMass = Vm(HMass,4)
# | S_CO | z | flag | Mgal | L_CO | D_L | V/Vm | Vm |
lumsnew = Vm(lumsnew,5)

# Calculate Luminosity Values ##################################################
# | S_CO | z | flag | Mgal | D_L | V/Vm | Vm | L_CO |
LMass = lCalc(LMass,0,1,4,True)
HMass = lCalc(HMass,0,1,4,False)
# | S_CO | z | flag | Mgal | L_CO | D_L | V/Vm | Vm | L_CO |
lumsnew = lCalc(lumsnew,0,1,5,True)

lumsL = LMass[:,7]
lumsH = HMass[:,7]
lumsnew = lumsnew[:,4]

lumsL = [i for i in lumsL if i > 0.0]         # remove 0 detected CO flux galaxies
lumsH = [i for i in lumsH if i > 0.0]         # remove 0 detected CO flux galaxies
lumsLlog, lumsHlog = np.log10(lumsL), np.log10(lumsH)
lCombinedlog = np.append(lumsLlog, lumsHlog)
lCombined = np.append(lumsL, lumsH)

# Sort Luminosity Values into bins #############################################
NL, midL = sortIntoBins(lumsLlog, 15)
NH, midH = sortIntoBins(lumsHlog, 15)
NC, midC = sortIntoBins(lCombinedlog, 20)
NR, midR = sortIntoBins(lumsnew, 15)

# Calculations for Mass Distribution ###########################################

totalMass = np.append(LMass[:,3], HMass[:,3])
Nmass, Xmass = sortIntoBins(totalMass, 30)

# Calculate H2 Mass fraction ###################################################

H2Mass = H2Conversion(lCombined)
H2Mass = np.log10(H2Mass)
NH2, xH2 = sortIntoBins(H2Mass, 15)

# Plot Luminosity number plot ##################################################
fig, ax = plt.subplots(nrows = 1, ncols = 3, squeeze=False)
ax[0,0].plot(midL,NL,'b-', label = 'Low mass')
ax[0,0].plot(midH,NH,'r-', label = 'high mass')
ax[0,0].plot(midC,NC,'g-', label = 'lCombined')
ax[0,0].plot(midR,NR,'k-', label = 'Pre-calc')
ax[0,0].set_xlabel(r'$log_{10}(L_{CO})$', fontsize=20)
ax[0,0].set_ylabel(r'$log_{10}(N)$', fontsize=20)
ax[0,0].set_title('CO Luminosity', fontsize=20)
ax[0,0].legend()
# ax[0,0].savefig('lum1.png')

# Plot H2 mass #################################################################
ax[0,1].plot(xH2, NH2,'b-', label = 'H2 Mass')
ax[0,1].set_xlabel(r'$log_{10}(M_{H2}/M_{\odot})$', fontsize=20)
ax[0,1].set_ylabel(r'$log_{10}(N_{gal})$', fontsize=20)
ax[0,1].set_title('CO Luminosity', fontsize=20)
ax[0,1].legend()
# plt.savefig('new.png')

# Plot V/Vm ####################################################################
ax[0,2].plot(LMass[:,7], LMass[:,5],'ko', label = r'$\frac{V}{V_{m}}$')
ax[0,2].set_xlabel(r'$log_{10}(L_{CO})$', fontsize=20)
ax[0,2].set_ylabel(r'$\frac{V}{V_{m}}$', fontsize=20)
ax[0,2].set_title('Schmidt Vm', fontsize=20)
ax[0,2].legend()
plt.show()
