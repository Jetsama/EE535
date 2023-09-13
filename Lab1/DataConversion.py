import math


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
c = 299792458 #speed of light
hEV = 4.1357 * 10**-15 #plank in evs

def fileEdit(In,Out,Conv,header = None):
    with open(In) as f_in, open(Out, 'w') as f_out:
        # Transform the rest of the lines
        if header:
           f_out.write(header)
        for line in f_in:
            f_out.write(Conv(line))

def fileCombine(In1,In2,Out,notfil= True):
    with open(In1) as f_in1,open(In2) as f_in2, open(Out, 'w') as f_out:
        # Write header unchanged
        if(notfil):
            header = f_in1.readline()
            header = f_in2.readline()
        #f_out.write(header)
        header = f_in1.readline() #ignore one blank lines
        header = f_in2.readline()

        # Transform the rest of the lines
        for (line,line2) in zip(f_in1,f_in2):
            data = line.split(sep=",")
            data[1] = data[1].strip()
            data2 = line2.split(sep=",")
            newLine = ",".join([data[0],data[1],data2[1]])
            f_out.write(newLine)
            #f_out.write(Conv(line))
def absorbtion(line):
    data = line.split(sep=",")
    if(is_integer(data[0]) or isfloat(data[0])):
        [wave,R,T] = data
        
        T = float(T.strip())/100
        if(T<=0):
            T = 0.00001
        R = float(R)/100

        freq = c/(float(wave)*(10**-9))
        Ev = hEV*freq
        temp = (1-math.pow(R,2))/T
        alpha = (1/t)*math.log(temp)
        data = [str(x) for x in [Ev,alpha,"\n"]]
        newLine = ",".join(data)
        return newLine
    else:
        return line.upper()
    


def derivative(x,y):
    der = np.diff(y) / np.diff(x)
    return [x,der]
    x2 = (x[:-1] + x[1:]) / 2
    plt.plot(x2, der, 'r', x, y, 'g', x, -np.sin(x),'b')

def CleanAndRemoveDer(xLis,yLis):
    xnew = list()
    ynew = list()
    for (x,y) in zip(xLis,yLis):

        if (y > 5000000 or y < -5000000):
            return (xnew,ynew)
        xnew.append(x)
        ynew.append(y)

def fileRead(In,offset = 1):   
    x = list()
    y = list()

    with open(In) as f_in:
        for i in range(offset):
            header = f_in.readline()
        for line in f_in:
            data = line.split(sep=",")
            x.append(float(data[0]))
            y.append(float(data[1].strip()))
    return [x,y,header]
def fileWrite(Out,xList,yList,header):   
    with open(Out, 'w') as f_out:
        f_out.write(header)
        for (x,y) in zip(xList,yList):
            data = [str(i) for i in [x,y,"\n"]]
            newLine = ",".join(data)
            f_out.write(newLine)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Fit the linear regression model
def linearReg(x,y,xReg,yReg):
    model = LinearRegression()
    xReg = np.array(xReg).reshape(-1, 1)
    yReg = np.array(yReg).reshape(-1, 1)
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    model.fit(xReg, yReg)

    # Compute the predicted values
    y_pred = model.predict(x)
    x = x.flatten().tolist()
    y = y_pred.flatten().tolist()
    return[x,y]

def taucDirect(EvList, alphaList):
    ta = list()
    for (Ev,alpha) in zip(EvList,alphaList):
        ta.append(math.pow((alpha * Ev),1/2))
    return [EvList,ta]
def taucInDirect(EvList, alphaList):
    ta = list()
    for (Ev,alpha) in zip(EvList,alphaList):
        ta.append(math.pow((alpha * Ev),2))
    return [EvList,ta]
def getLinearRegion(xList,yList,xReg):
    count = 0
    x2 = list()
    y2 = list()
    for (x,y) in zip(xList,yList):
        if(x > xReg):
            count+=1
            x2.append(x)
            y2.append(y)
        if(count > 25):
            return [x2,y2]
def startRegion(xList,yList,more,less):
    x2 = list()
    y2 = list()
    for (x,y) in zip(xList,yList):
        if(y>more and y<less ):
            x2.append(x)
            y2.append(y)
    return [x2,y2]


fileCombine("Lab1/T_R_A Data/Wafer_1-refl2.csv","Lab1/T_R_A Data/Wafer_1-trans.csv","Lab1/PyData/Wafer_1-RT.csv")
t = 11.3#thickness of wafer in mils
t= t/(393.7)#(in cm)

fileEdit("Lab1/PyData/Wafer_1-RT.csv","Lab1/PyData/Wafer_1-Alpha.csv", absorbtion,header="Ev,alpha\n")
[x,alpha,header] =fileRead('Lab1/PyData/Wafer_1-Alpha.csv')
[hev,tauc] = taucDirect(x,alpha)
fileWrite("Lab1/PyData/Wafer_1-TaucDirect.csv",hev,tauc,header)

mid = .75
[xreg,yreg] = getLinearRegion(hev,tauc,mid)
[xReg,yReg] = linearReg(hev,tauc,xreg,yreg)
[xReg,yReg] = startRegion(xReg,yReg,0,30)
fileWrite("Lab1/PyData/Wafer_1-TaucDirectLINEAR.csv",xReg,yReg,header)

[hev,tauc] = taucInDirect(x,alpha)
fileWrite("Lab1/PyData/Wafer_1-TaucInDirect.csv",hev,tauc,header)
mid = .78
[xreg,yreg] = getLinearRegion(hev,tauc,mid)
[xReg,yReg] = linearReg(hev,tauc,xreg,yreg)
[xReg,yReg] = startRegion(xReg,yReg,0,1.5*10**5)
fileWrite("Lab1/PyData/Wafer_1-TaucInDirectLINEAR.csv",xReg,yReg,header)

#wafer 2
t = 7.8#thickness of wafer in mils
t= t/(393.7)#(in cm)
fileCombine("Lab1/T_R_A Data/Wafer_2-refl2.csv","Lab1/T_R_A Data/Wafer_2-trans.csv","Lab1/PyData/Wafer_2-RT.csv")
fileEdit("Lab1/PyData/Wafer_2-RT.csv","Lab1/PyData/Wafer_2-Alpha.csv", absorbtion,header="Ev,alpha\n")
[x,alpha,header] =fileRead('Lab1/PyData/Wafer_2-Alpha.csv')
[hev,tauc] = taucDirect(x,alpha)
fileWrite("Lab1/PyData/Wafer_2-TaucDirect.csv",hev,tauc,header)

mid = 1.25
[xreg,yreg] = getLinearRegion(hev,tauc,mid)
[xReg,yReg] = linearReg(hev,tauc,xreg,yreg)
[xReg,yReg] = startRegion(xReg,yReg,0,25)
fileWrite("Lab1/PyData/Wafer_2-TaucDirectLINEAR.csv",xReg,yReg,header)


[hev,tauc] = taucInDirect(x,alpha)
fileWrite("Lab1/PyData/Wafer_2-TaucInDirect.csv",hev,tauc,header)

mid = 1.3
[xreg,yreg] = getLinearRegion(hev,tauc,mid)
[xReg,yReg] = linearReg(hev,tauc,xreg,yreg)
[xReg,yReg] = startRegion(xReg,yReg,0,0.5*10**6)
fileWrite("Lab1/PyData/Wafer_2-TaucInDirectLINEAR.csv",xReg,yReg,header)



#wafer 3
t = 0.1#thickness of wafer in cm
fileCombine("Lab1/T_R_A Data/Wafer3 Reflection Filmetrics.csv","Lab1/T_R_A Data/Wafer3 Transmission Filmetrics.csv","Lab1/PyData/Wafer_3-RT.csv",notfil=False)
fileEdit("Lab1/PyData/Wafer_3-RT.csv","Lab1/PyData/Wafer_3-Alpha.csv", absorbtion,header="Ev,alpha\n")
[x,alpha,header] =fileRead('Lab1/PyData/Wafer_3-Alpha.csv')
[hev,tauc] = taucDirect(x,alpha)
fileWrite("Lab1/PyData/Wafer_3-TaucDirect.csv",hev,tauc,header)

mid = 1.65
hev.reverse()
tauc.reverse()
[xreg,yreg] = getLinearRegion(hev,tauc,mid)
[xReg,yReg] = linearReg(hev,tauc,xreg,yreg)
[xReg,yReg] = startRegion(xReg,yReg,0,15)
fileWrite("Lab1/PyData/Wafer_3-TaucDirectLINEAR.csv",xReg,yReg,header)
hev.reverse()
tauc.reverse()
[hev,tauc] = taucInDirect(x,alpha)
fileWrite("Lab1/PyData/Wafer_3-TaucInDirect.csv",hev,tauc,header)

mid = 1.65
[xreg,yreg] = getLinearRegion(hev,tauc,mid)
[xReg,yReg] = linearReg(hev,tauc,xreg,yreg)
[xReg,yReg] = startRegion(xReg,yReg,0,0.5*10**5)
fileWrite("Lab1/PyData/Wafer_3-TaucInDirectLINEAR.csv",xReg,yReg,header)

#[x,y,header] =fileRead('Lab1/PyData/Wafer_1-TaucPlot.csv')
#[xReg,yReg] = linearReg(x,y)
#[xDer,yDer] = derivative(x,y)
#[xDer,yDer] = CleanAndRemoveDer(xDer,yDer)
#fileWrite("Lab1/PyData/Wafer_1-TaucExtend.csv",xReg,yReg,header)
#fileWrite("Lab1/PyData/Wafer_1-TaucDir.csv",xDer,yDer,header)



#wafer 1
[x,T,header] = fileRead("Lab1/T_R_A Data/Wafer_1-trans.csv",offset=2)
[x,R,header] = fileRead("Lab1/T_R_A Data/Wafer_1-refl2.csv",offset=2)
A= list()
for (j,k) in zip(T,R):
    A.append(100- (j+k))
fileWrite("Lab1/PyData/Wafer_1-A.csv",x,A,"Wavelength (nm),A \n")
#wafer 2
[x,T,header] = fileRead("Lab1/T_R_A Data/Wafer_2-trans.csv",offset=2)
[x,R,header] = fileRead("Lab1/T_R_A Data/Wafer_2-refl2.csv",offset=2)
A= list()
for (j,k) in zip(T,R):
    A.append(100- (j+k))
fileWrite("Lab1/PyData/Wafer_2-A.csv",x,A,"Wavelength (nm),A \n")
#wafer 3
[x,T,header] = fileRead("Lab1/T_R_A Data/Wafer3 Transmission Filmetrics.csv",offset=2)
[x,R,header] = fileRead("Lab1/T_R_A Data/Wafer3 Reflection Filmetrics.csv",offset=2)
A= list()
for (j,k) in zip(T,R):
    A.append(100- (j+k))
fileWrite("Lab1/PyData/Wafer_3-A.csv",x,A,"Wavelength (nm),A \n")

