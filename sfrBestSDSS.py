import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import atpy
import pandas as pd

def sortTable2(info, sfrs, indices):
    newdata = np.zeros((len(info), len(indices)))
    for index, rows in enumerate(info):
        newdata[index, 0] = int(rows[indices['PLATEID']])
        newdata[index, 1] = int(rows[indices['MJD']])
        newdata[index, 2] = int(rows[indices['FIBERID']])
        newdata[index, 3] = sfrs[index][indices['SFR']]
    return newdata

def sfrbest13k():
    SDSS_index = {'PLATEID':0, 'MJD':1, 'FIBERID':2, 'SFR':0}
    df = pd.read_csv('data/allGASS_SFRbest_simple_t1.csv')
    SFRBL = df[['GASS', 'SFR_best']].values
    df = pd.read_csv('data/PS_100701.csv')
    coords_H = df[['GASS', 'PLATEID', 'MJD', 'FIBERID']].values
    df = pd.read_csv('data/LOWMASS_MASTER.csv')
    coords_L = df[['GASSID', 'PLATEID', 'MJD', 'FIBERID']].values
    galinfo = atpy.Table('data/sdss/gal_info_dr7_v5_2.fit')
    sfr = atpy.Table('data/sdss/gal_totsfr_dr7_v5_2.fits')
    S = sortTable2(galinfo, sfr, SDSS_index)
    GASS = np.vstack((coords_H, coords_L))
    IDs = np.zeros((len(SFRBL),4))
    for index, element in enumerate(SFRBL):
        print index
        for index1, element1 in enumerate(GASS):
            if element[0] == element1[0]:
                IDs[index, 0] = element1[1]
                IDs[index, 1] = element1[2]
                IDs[index, 2] = element1[3]
                for idx, el in enumerate(S):
                    if el[0]==element1[1] and el[1]==element1[2] and el[2]==element1[3]:
                        IDs[index,3] = el[3]
    SFRBL = np.hstack((SFRBL, IDs))
    np.savetxt('sfrC.txt', SFRBL)
    return SFRBL

SFRBL = sfrbest13k()
