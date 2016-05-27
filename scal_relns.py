import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def first(x, a, b):
    return a*x + b

def second(x, a, b, c):
    return a*x*x + b*x + c

def third(x, a, b, c, d):
    return a*x*x*x + b*x*x + c*x + d

def fourth(x, a, b, c, d, e):
    return a*x*x*x*x + b*x*x*x + c*x*x + d*x + e


df = pd.read_csv('data/cold_gass_data_gio.csv')
data = df[['Log_Mh2', 'Log_SFR', 'Log_M', 'Log_LCO']].values

paramslist = []
paramslist.append(curve_fit(first, data[:,1], data[:,0]))
paramslist.append(curve_fit(second, data[:,1], data[:,0]))
paramslist.append(curve_fit(third, data[:,1], data[:,0]))
paramslist.append(curve_fit(fourth, data[:,1], data[:,0]))
print curve_fit(first, data[:,1], data[:,0])[0]

x = np.linspace(-1.5,2.0,300)
y1 = first(x, paramslist[0][0][0], paramslist[0][0][1])
y2 = second(x, paramslist[1][0][0], paramslist[1][0][1], paramslist[1][0][2])
y3 = third(x, paramslist[2][0][0], paramslist[2][0][1], paramslist[2][0][2], paramslist[2][0][3])
y4 = fourth(x, paramslist[3][0][0], paramslist[3][0][1], paramslist[3][0][2], paramslist[3][0][3], paramslist[3][0][4])

fig, ax = plt.subplots(nrows = 1, ncols = 2, squeeze=False)
s = ax[0,0].scatter(data[:,1], data[:,0], c=data[:,2])
#ax[0,0].plot(x,y1, linewidth = 1)
#ax[0,0].plot(x,y2, linewidth = 1)
ax[0,0].plot(x,y3, linewidth = 1)
#ax[0,0].plot(x,y4, linewidth = 5)
ax[0,0].set_ylabel(r'$Log(MH2)$', fontsize = 12)
ax[0,0].set_xlabel(r'$Log(SFR)$', fontsize = 12)
cb = plt.colorbar(s)
cb.set_label('Log M')
################
t = ax[0,1].scatter(data[:,1], data[:,3], c=data[:,2])
ax[0,1].set_ylabel(r'$Log(LCO)$', fontsize = 12)
ax[0,1].set_xlabel(r'$Log(SFR)$', fontsize = 12)
# cb = plt.colorbar(t)
# cb.set_label('Log M')
fig.set_size_inches(10,6)
plt.savefig('img/scalrelns.pdf', format='pdf', dpi=1000, transparent = False)
#plt.show()
