import numpy as np
from StringIO import StringIO
import re

data  = '284.801,6893.28 19.539,-1.01 19.531,-0.98 0.039,0 19.531,-0.9 19.571,-0.78 0,-0.04 19.527,-0.74 19.57,-0.71 19.532,-0.66 19.531,-0.58 19.566,-0.55 19.532,-0.55 19.531,-0.47 0.039,0 19.5,-0.43 0.039,0 19.531,-0.43 19.531,-0.35 19.489,-0.35 19.019,-0.63 19.141,-3.47 19.5,-3.48 0.031,0 19.578,-3.44 19.531,-3.35 19.532,-3.29 0.039,0 19.531,-3.24 19.527,-3.16 0.039,0 19.532,-3.09 19.531,-3.01 0.039,0 19.531,-2.97 19.528,-2.89 0.043,0 19.527,-2.81 19.531,-2.73 0.039,0 19.531,-2.7 19.54,-2.58 0.031,0 19.539,-2.53 0.039,0 19.531,-2.47 19.531,-2.38 0.039,0 19.532,-2.3 19.527,-2.27 0.039,0 19.532,-2.15 19.53,-2.11 0.04,0 19.53,-2.03 0,-0.04 19.53,-1.95 0.04,0 19.53,-1.91 19.53,-1.84 0.04,0 19.53,-1.76 19.54,-1.72 0.03,0 19.54,-1.68 58.59,-4.92 19.49,-1.72 -0.04,0 19.49,-1.87 19.42,-2.19 19.33,-2.62 19.3,-3.32 19.14,-4.22 19.1,-5.78 19.03,-7.5 18.98,-9.61 19.03,-12.07 19.02,-15.43 19.14,-18.67 19.22,-22.07 19.26,-25.7 19.33,-30 19.38,-34.49 19.37,-38.95 19.45,-43.55 19.46,-48.64 19.45,-53.86 19.49,-58.13 19.5,-61.87 19.53,-65.16 19.49,-68.71 19.53,-70.27 19.53,-69.93 0.04,-0.07 19.53,-66.57 0.04,-0.15 19.53,-60.71 0.08,-0.23 19.53,-53.2 0.08,-0.24 19.53,-47.54 0.08,-0.11 19.49,-44.85 0.04,0 19.53,-44.37 -0.04,0.04 19.53,-45.08 19.53,-46.33 19.5,-47.85 19.49,-49.42 19.49,-51.09 19.53,-52.77 0,0.07 19.53,-54.45 -0.04,0.08 19.54,-56.09 0,0.03 21.48,-63.55 11.6,3.95 -21.48,63.55 0,0.04 -19.53,56.09 -0.04,0.04 -19.53,54.46 0,0.03 -19.54,52.78 -0.03,0.08 -19.54,51.13 -0.03,0.08 -19.54,49.49 0,0.08 -19.53,47.89 -0.04,0.08 -19.53,46.4 -0.04,0.04 -19.53,45.12 -19.53,44.37 -19.49,44.77 -19.45,47.38 -19.46,52.97 -19.49,60.51 -19.49,66.48 -19.49,69.85 -19.54,70.27 -0.03,0.04 -19.54,68.79 0,0.08 -19.53,65.23 -0.04,0.08 -19.53,61.99 -0.04,0.08 -19.53,58.28 -0.04,0.12 -19.53,54.02 -19.61,49.02 -0.08,0.24 -19.53,43.79 -0.11,0.23 -19.54,39.18 -0.15,0.28 -19.53,34.76 -0.2,0.31 -19.53,30.36 -0.23,0.39 -19.53,26.01 -0.28,0.35 -19.53,22.46 -0.35,0.39 -19.53,19.03 -0.43,0.39 -19.53,15.78 -0.59,0.43 -19.49,12.42 -0.55,0.27 -19.53,9.93 -0.51,0.19 -19.53,7.74 -0.47,0.15 -19.57,5.9 -0.43,0.12 -19.53,4.33 -0.27,0.04 -19.53,3.36 -0.24,0 -19.53,2.66 -0.12,0.04 -19.53,2.19 -0.11,0 -19.54,1.87 -0.03,0 -19.54,1.72 -0.04,0 -19.53,1.68 -19.53,1.64 0,-0.04 -19.53,1.64 -19.49,1.68 -19.53,1.72 0.04,0 -19.57,1.8 0.04,0 -19.54,1.83 -19.53,1.88 0.04,0 -19.53,1.99 -19.53,2.03 0.04,0 -19.53,2.07 0.04,0 -19.531,2.19 -19.539,2.23 0.039,0 -39.058,4.68 0.039,0 -19.532,2.46 -19.527,2.54 0.039,0 -19.531,2.58 -19.539,2.7 0.039,0 -19.527,2.73 -19.532,2.81 0.039,-0.04 -19.527,2.89 0.039,0 -19.531,2.97 -19.539,3.01 0.039,0 -19.531,3.09 -19.528,3.16 0.039,0 -19.531,3.2 -19.527,3.32 0.039,0 -19.531,3.32 -19.54,3.44 -19.492,3.48 -19.527,3.55 -0.902,0.08 -19.528,0.67 -0.082,0 -39.058,0.7 -19.532,0.43 0.039,0 -19.527,0.43 -19.531,0.47 -19.539,0.5 0.039,0 -19.57,0.55 0.039,0 -39.059,1.25 0.039,0 -19.527,0.7 -19.532,0.79 0.04,0 -19.54,0.82 -19.531,0.89 0.039,0 -19.527,0.94 -25.621,1.37 -0.668,-12.27 6.129,-0.31'
data = re.split(" ", data)

# data = data.split(" ")
# a = np.genfromtxt(StringIO(data), delimiter = ",")
# print data
print len(data)
x,y = 0,0
new = np.zeros((len(data),2))
for idx, element in enumerate(data):
    element = element.split(",")
    dx,dy = element
    x += float(dx)
    y += float(dy)
    new[idx,0] = x
    new[idx,1] = y
    print x,y
    # print (i*2)-1, (i*2)
    # newdata[i,0] = data[(i*2)-1]
    # newdata[i,0] = data[(i*2)]

np.savetxt('test33.csv', new)
