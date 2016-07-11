import numpy as np
import atpy
import math
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import asciidata
from scipy.optimize import curve_fit
import csv
import pandas as pd
import random

def sdssMethod(zl, zh):
    galinfo = atpy.Table('data/sdss/gal_info_dr7_v5_2.fit')
    sfr = atpy.Table('data/sdss/gal_totsfr_dr7_v5_2.fits')
    mstar = atpy.Table('data/sdss/totlgm_dr7_v5_2.fit')
    sdssData = np.zeros((len(galinfo),3))
    for i in range(0,len(galinfo)):
        # redshift from galinfo
        sdssData[i,0] = galinfo[i][12]
        # stellar mass from totlgm
        sdssData[i,1] = mstar[i][6]
        # total sfr from totsfr
        sdssData[i,2] = sfr[i][0]
    sdssData = sdssData[sdssData[:,0] > zl]
    sdssData = sdssData[sdssData[:,0] < zh]
    sdssData = sdssData[sdssData[:,1] > 4]
    return sdssData


plt.scatter(sdssData[:,1], sdssData[:,2])
plt.xlim(8,11)
plt.ylim(-2.5,1)
plt.show()
