import numpy as np
import atpy
import math
from scipy import integrate
import matplotlib.pyplot as plt
import asciidata
from scipy.optimize import curve_fit
import schechter

def fgas(Mgas, Mstar):
    Mgas = math.pow(10,Mgas)
    Mstar = math.pow(10,Mstar)
    m510 = 5*math.pow(10,10)
    fgas = Mgas/(Mgas + Mstar)
    #fgas510 = Mgas/(Mgas + m510)
    #print fgas, fgas510, fgas/fgas510
    return fgas

## Read data from tables #######################################################
highM = atpy.Table('COLDGASS_DR3.fits')
lowM = asciidata.open('COLDGASS_LOW_29Sep15.ascii')
SAMI = asciidata.open('SAMI_IRAM_data.txt')
################################################################################
SAMI_Mstar = SAMI[2]
SAMI_MH2 = SAMI[6]
SAMI_flag = SAMI[8]
data = np.zeros((len(SAMI_MH2),4))
for i in range (len(data)):
    data[i,0] = SAMI_Mstar[i]
    data[i,1] = SAMI_MH2[i]
    data[i,2] = SAMI_flag[i]
    data[i,3] = fgas(data[i,0],data[i,1])

################################################################################
L_Mstar = lowM[4]
L_MH2 = lowM[22]
L_flag = lowM[15]
data1 = np.zeros((len(L_MH2),4))
for i in range (len(data1)):
    data1[i,0] = L_Mstar[i]
    data1[i,1] = L_MH2[i]
    data1[i,2] = L_flag[i]
    data1[i,3] = fgas(data1[i,0],data1[i,1])

################################################################################
# H_Mstar = highM[5]
# H_MH2 = highM[22]
# H_flag = highM[15]
# data2 = np.zeros((len(H_MH2),4))
# for i in range (len(data2)):
#     data2[i,0] = H_Mstar[i]
#     data2[i,1] = H_MH2[i]
#     data2[i,2] = H_flag[i]
#     data2[i,3] = fgas(data2[i,0],data2[i,1])
#
plt.plot(data[:,0], data[:,3],'ro')
# plt.plot(data1[:,0], data1[:,3],'bo')
# plt.plot(data2[:,0], data2[:,3],'ko')
plt.show()
