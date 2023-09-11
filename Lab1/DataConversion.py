import math


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

c = 299792458 #speed of light
hEV = 4.1357 * 10**-15 #plank in evs

def fileEdit(In,Out,Conv,header = None):
    with open(In) as f_in, open(Out, 'w') as f_out:
        # Transform the rest of the lines
        if header:
           f_out.write(header)
        for line in f_in:
            f_out.write(Conv(line))

def fileCombine(In1,In2,Out):
    with open(In1) as f_in1,open(In2) as f_in2, open(Out, 'w') as f_out:
        # Write header unchanged
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
    if(is_integer(data[0])):
        [wave,R,T] = data
        
        T = float(T.strip())/100
        if(T<0):
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
fileCombine("T_R_A Data/Wafer_1-refl2.csv","T_R_A Data/Wafer_1-trans.csv","PyData/Wafer_1-RT.csv")
t = 11.3#thickness of wafer in mils
t= t/(39.37) * 10**-3 #(in mm)

fileEdit("PyData/Wafer_1-RT.csv","PyData/Wafer_1-TaucPlot.csv", absorbtion,header="Ev,alpha\n")


import pandas as pd
import matplotlib.pyplot as plt

filename = 'PyData/Wafer_1-TaucPlot.csv'
dataframe = pd.read_csv(filename)

#dataframe["DATE"] = pd.to_datetime(dataframe['DATE'], format="%Y-%m-%d")

fig, ax = plt.subplots()

ax.plot(dataframe.axes[1])
fig.autofmt_xdate()
plt.show(block=True)