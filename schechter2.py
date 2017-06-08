import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid.axislines import SubplotZero
from scipy.optimize import curve_fit

SFR_M = 5.96E-11
LOG_SFR_M = np.log(SFR_M)

def log_schechter(logL, log_phi, log_L0, alpha):
    schechter = log_phi
    schechter += (alpha+1)*(logL-log_L0)*np.log(10)
    schechter -= pow(10,logL-log_L0)
    return schechter

def log_schechter1(logL, log_rho, log_Lstar, alpha):
    schechter = log_rho*((logL-log_Lstar)**(alpha+1))*math.exp(-(L-Lstar))*np.log(10)
    return schechter

def log_schechter_fit(schechX, schechY):
    popt, pcov = curve_fit(log_schechter, schechX, schechY)
    return popt, pcov

def log_schechter_fit2(schechX, schechY, sigma):
    popt, pcov = curve_fit(log_schechter, schechX, schechY, sigma = sigma)
    return popt, pcov

# schechX = np.linspace(8, 12, 100)
# schechY = log_schechter(schechX, LOG_SFR_M, 11.03, -1.3)
# popt = log_schechter_fit(schechX, schechY)
# ynew = log_schechter(schechX, *popt)
#
# fig = plt.figure(1)
# ax = SubplotZero(fig, 111)
# fig.add_subplot(ax)
# ax.plot(schechX, schechY, 'k')
# ax.plot(schechX, ynew, 'ro')
# plt.show()
