import numpy as np
import atpy
import math
from scipy import integrate
import matplotlib.pyplot as plt
import asciidata

# Function to calculate the luminosity distance from z #########################
def lumdistance(z):
    omega_m = 0.31                          # from Planck
    omega_l = 0.69                          # from Planck
    c = 3*math.pow(10,5)                    # in km/s
    Ho = 75                                 # in km/(s Mpc)
    f = lambda x : (((omega_m*((1+z)**3))+omega_l)**-0.5)
    integral = integrate.quad(f, 0.0, z)    # numerically integrate to calculate luminosity distance
    Dm = (c/Ho)*integral[0]
    Dl = (1+z)*Dm                           # calculate luminosity distance
    DH = (c*z)/Ho                           # calculate distance from Hubble law for comparison
    return Dl, DH

# Calculate CO Luminosity ######################################################
def lCalc(SCO, z, correction):
    lums = []
    for i in range(0,len(SCO)):                              # for each galaxy in the dataset
        if correction == True:
            SCO_cor = SCO[i]*6.0                      # find the integrated CO flux
        else:
            SCO_cor = SCO[i]
        C = 3.25*math.pow(10,7)                 # numerical const from eqn 4 paper 1
        freq = 111                              # observing frequency
        Dl, DH = lumdistance(z[i])           # luminosity distance
        SDSS_z = math.pow((1+z[i]),-3)       # redshift component
        L_CO = C*SCO_cor*((Dl*Dl)/(freq*freq))*SDSS_z   # calculate CO luminosity
        lums.append(L_CO)
    return lums

# Remove non-detections ########################################################
def NonDetect(data, flag):
    datanew = []
    for i in range (0,len(data)):
        if flag[i] == 1:
            datanew.append(data[i])
    return datanew

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
def H2Conversion(LCO):
    molecularH = []
    alpha = 3.2
    for l in LCO:
        H2mass = alpha*l
        molecularH.append(H2mass)
    return molecularH

# Vm calc ######################################################################
def Vm(maglim,magGal,z):
    Dl = lumdistance(z)[0]
    absMagGal = magGal - (5*(np.log10(Dl)-1))
    print absMagGal
    Vm = ((4*math.pi)/3)*np.log10(0.6*(abs(maglim-25-absMagGal)))
    V = ((4*math.pi)/3)*Dl*Dl*Dl
    print '@@',Vm, V/Vm
    return Vm

## Read data from tables #######################################################
highM = atpy.Table('COLDGASS_DR3.fits')
lowM = asciidata.open('COLDGASS_LOW_29Sep15.ascii')

# Sort Data ####################################################################
SCOH, zH, totalMH, appMagH = [], [], [], []
for rows in highM:                              # for each galaxy in the dataset
    SCOH.append(rows[15])                      # find the integrated CO flux
    zH.append(rows[4])
    totalMH.append(rows[5])
    appMagH.append(rows[11])
SCOL, zL, flagL, totalML = list(lowM[11]), list(lowM[3]), list(lowM[15]), list(lowM[4])
lumsnew = list(lowM[12])
lumsnew = NonDetect(lumsnew, flagL)
SCOL, zL = NonDetect(SCOL, flagL), NonDetect(zL, flagL)

# Calculate Vm #################################################################
Vms = []
maglimit = min(appMagH)
print maglimit, appMagH[0], zH[0]
for j in range(0,len(appMagH)):
    Vma = Vm(maglimit, appMagH[j], zH[j])

# Calculate Luminosity Values ##################################################
lumsH = lCalc(SCOH,zH,False)
lumsL = lCalc(SCOL,zL,True)
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

totalMass = np.append(totalML, totalMH)
Nmass, Xmass = sortIntoBins(totalMass, 30)

# Calculate H2 Mass fraction ###################################################

H2Mass = H2Conversion(lCombined)
H2Mass = np.log10(H2Mass)
NH2, xH2 = sortIntoBins(H2Mass, 15)


# Plot Luminosity number plot ##################################################
plt.plot(midL,NL,'b-', label = 'Low mass')
plt.plot(midH,NH,'r-', label = 'high mass')
plt.plot(midC,NC,'g-', label = 'lCombined')
plt.plot(midR,NR,'k-', label = 'Pre-calc')
# plt.plot(Xmass,Nmass,'c-', label = 'Mass')
plt.xlabel('log_10(L_CO)')
plt.ylabel('log_10(N)')
plt.title('CO Luminosity')
plt.legend()
plt.savefig('lum1.png')
plt.show()

# Plot H2 mass #################################################################
# plt.plot(xH2, NH2,'b-', label = 'H2 Mass')
# plt.plot(midC,NC,'g-', label = 'lCombinedlog')
# plt.xlabel(r'$log_{10}(M_{H2}/M_{\odot})$', fontsize=20)
# plt.ylabel(r'$log_{10}(N_{gal})$', fontsize=20)
# plt.title('CO Luminosity', fontsize=20)
# plt.legend()
# plt.savefig('new.png')
# plt.show()
