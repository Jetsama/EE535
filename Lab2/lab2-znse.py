import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from PythonModule.filemanager import fileRead

from scipy.signal import find_peaks

[x,y,header] = fileRead("Lab2/data/ZnSe Trans.csv",offset=2);
ylist =y 
xlist = x

plt.plot(x, y)
plt.xlabel('Wavelength (nm)')
plt.ylabel('%T')
plt.title('Zinc Selenide (ZnSe) Transmission')

plt.savefig("Lab2/figures/znset.png")

indices = find_peaks(y, width = 4)[0]
x =[x[i] for i in indices]
y= [y[i] for i in indices]
plt.scatter(x, y)
indices = find_peaks([ -x for x in ylist], width = 4)[0]
x =[xlist[i] for i in indices]
y= [ylist[i] for i in indices]
plt.scatter(x, y)

plt.savefig("Lab2/figures/znsetp.png") #with peaks

