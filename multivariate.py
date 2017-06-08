import matplotlib.pyplot as plt
import numpy as np
import schechter2
import pandas as pd
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def throws(mean, covariance, n, bins):
    boundary = np.zeros((n,len(bins)))
    for i in range(0,n):
        a = np.random.multivariate_normal(mean, covariance)
        a1 = schechter2.log_schechter(bins, *a)
        boundary[i,:] = a1
    minaxis = np.amin(boundary, axis = 0)
    maxaxis = np.amax(boundary, axis = 0)
    return boundary, minaxis, maxaxis


covariance = np.loadtxt('covariance.txt')
mean = np.loadtxt('mean.txt')
pts = pd.read_csv('luminosity.txt', sep = '\t')
# print (covariance)
# print (mean)
# print (pts)

xmajorLocator   = MultipleLocator(1)
xminorLocator   = MultipleLocator(0.2)
ymajorLocator   = MultipleLocator(1)
yminorLocator   = MultipleLocator(0.2)
fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
ax[0,0].tick_params(axis='both', which='major', labelsize=15)
ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0,0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0,0].xaxis.set_major_locator(xmajorLocator)
ax[0,0].xaxis.set_minor_locator(xminorLocator)
ax[0,0].yaxis.set_major_locator(ymajorLocator)
ax[0,0].yaxis.set_minor_locator(yminorLocator)
print (pts['x'][6:14].values)

new_fit, covariance = schechter2.log_schechter_fit(pts['x'][6:14].values, pts['y'][6:14].values)
perr = np.sqrt(np.diag(covariance))
print ('phi_star = ' + str(round(new_fit[0],2)) + ' +/- ' + str(round(perr[0],2)))
print ('L_0 = ' + str(round(new_fit[1],2)) + ' +/- ' + str(round(perr[1],2)))
print ('alpha = ' + str(round(new_fit[2],3)) + ' +/- ' + str(round(perr[2],2)))
new_fit2, covariance2 = schechter2.log_schechter_fit2(pts['x'][6:14].values, pts['y'][6:14].values, pts['dyu'][6:14].values)
new_fit3, covariance2 = schechter2.log_schechter_fit2(pts['x'][6:13].values, pts['y'][6:13].values, pts['dyd'][6:13].values)
n = 800
bins = np.linspace(5,12,200)
boundary = np.zeros((n,200))
# best_fit = schechter2.log_schechter(bins, *mean)
best_fit = new_fit
print (best_fit)
for i in range(0,n):
    a = np.random.multivariate_normal(mean, covariance)
    a1 = schechter2.log_schechter(bins, *a)
    ax[0,0].plot(bins, a1, color = 'k', alpha = 0.1, linewidth = 0.3)
    boundary[i,:] = a1
# # print (boundary[:,100])
# # print (np.std(boundary[:,100]))
# boundary = np.std(boundary, axis = 0)
# print (len(boundary))
# print (boundary)
minaxis = np.amin(boundary, axis = 0)
maxaxis = np.amax(boundary, axis = 0)
ax[0,0].plot(bins, schechter2.log_schechter(bins, *best_fit), color = 'crimson', linewidth = 3)
ax[0,0].fill_between(bins, minaxis, maxaxis, color = 'crimson', alpha = 0.3)
# ax[0,0].plot(bins, minaxis, color = 'r')
# ax[0,0].plot(bins, maxaxis, color = 'r')
# ax[0,0].plot(bins, schechter2.log_schechter(bins, *new_fit2), color = 'b')
# ax[0,0].plot(bins, schechter2.log_schechter(bins, *new_fit3), color = 'g')
ax[0,0].errorbar(pts['x'], pts['y'], yerr=[pts['dyd'], pts['dyu']], fmt='^', markersize = 10, linewidth=2, markeredgewidth=2, capthick=3, mfc='r', mec='crimson', ecolor = 'crimson', label='xCOLD GASS det+non-det', zorder = 7)
ax[0,0].set_xlim(5.5, 12)
ax[0,0].set_ylim(-7, -1)
ax[0,0].set_xlabel(r'$\mathrm{log\, L^{\prime}_{CO}\, [K \, km \,s^{-1}\, pc^{2}]}$', fontsize=18)
ax[0,0].set_ylabel(r'$\mathrm{log\, \phi\, [Mpc^{-3}\, dex^{-1}]}$', fontsize=18)
plt.savefig('figure1.png', dpi = 400)
# print (a)
