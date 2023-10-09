
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from PythonModule.filemanager import fileRead
from PythonModule.filemanager import fileReadCols
from scipy.interpolate import interp1d

from scipy.signal import find_peaks


def ReferenceQE():
    [x,y,header] = fileRead("Lab4/data/Reference_QE.csv",offset=2);

    plt.plot(x, y)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('QE %')
    plt.title('Reference Quantum Efficiency')
    plt.savefig("Lab4/figures/ReferenceQE.png")
def MeasuredReferenceQE():
    [cols,header] = fileReadCols("Lab4/data/EE535_Fall23_QE_Silicon.csv",offset=2);
    [wave, si, ref] = cols;
    
    plt.plot(wave, si,label = "Silicon Sample")
    plt.plot(wave, ref,label = "Reference Sample")
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reference Signal(mV)')
    plt.title('Reference and Silicon Measured Voltage')
    plt.legend()
    plt.savefig("Lab4/figures/VoltageRefSI.png")
    
    
def calcQE(file,QE_ref):
    QE = list();
    reference = list()
    [x,y,header] = fileRead(QE_ref,offset=2);
    QE_func = interp1d(x,y,bounds_error=False)
    xnew = np.arange(max(x),min(x) , 0.1)
    ynew = QE_func(xnew)
    
    [cols,header] = fileReadCols(file,offset=2);
    
    for col in zip(cols[0],cols[1],cols[2]):
        [wave, sign, ref] = col;
        QE.append( QE_func(wave)*sign/ref)
        reference.append( QE_func(wave))
    [wave, sign, ref] = cols;    
    return [wave, QE,reference]
    
def plotSi():
    [wave, QE,ref] = calcQE("Lab4/data/EE535_Fall23_QE_Silicon.csv","Lab4/data/Reference_QE.csv")
        
    plt.plot(wave, QE,label = "Silicon Sample")
    plt.plot(wave, ref,label = "Silicon Reference")
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reference Signal(mV)')
    plt.title('Reference and Silicon Calculated QE')
    plt.legend()
    plt.savefig("Lab4/figures/QESI.png")
    
    
#MeasuredReferenceQE()
plotSi()
plt.show()
#ReferenceQE()
