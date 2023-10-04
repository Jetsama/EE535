
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from PythonModule.filemanager import fileRead


from scipy.signal import find_peaks

from scipy.interpolate import interp1d








[x,y,header] = fileRead("Lab2/data/aSi_T.csv",offset=2);
[x2,n,header] = fileRead("Lab2/data/aSi_RefractiveIndexINFO.csv");

x_n = np.array(x2)*1000

curve, = plt.plot(x_n, n)
plt.scatter(x_n, n)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Refractive Index')
plt.title('Amorphous Silicon (aSi) Refractive Index')

plt.savefig("Lab2/figures/asirefractive.png")
plt.show()

N_func = interp1d(x_n,n,bounds_error=False)
xnew = np.arange(104,2200 , 0.1)

ynew = N_func(xnew)   # use interpolation function returned by `interp1d`

plt.plot(x_n, n, 'o', xnew, ynew, '-')
plt.savefig("Lab2/figures/ninterp.png")

plt.show()

X_N_data = curve.get_xdata()
N_data = curve.get_ydata()

[xerror,yerror,header] = fileRead("Lab2/data/aSi_T_instrument_error.csv",offset=2);
xerror = np.array(xerror)
yerror = np.array(yerror)

ylist = np.array(y) #-yerror
xlist = np.array(x)

min_x = int(min(min(xlist), min(xerror)))
max_x = int(max(max(xlist), max(xerror)))

# Create NumPy arrays for both lists with zeros added to the front
size = int( max_x - min_x + 1)
list1_y_array = np.zeros(size)
list2_y_array = np.zeros(size)

start_x = int(min(xlist) - min_x)
end_x = start_x + len(ylist)

start_error = len(ylist) - len(xerror)
end_error = len(ylist)

list1_y_array[start_x:end_x] = ylist
list2_y_array[start_error: end_error] = yerror

ylist_error = list1_y_array - list2_y_array

plt.plot(xerror, yerror, label="error")
plt.plot(xlist, ylist, label="no error corectio")
plt.plot(xlist, ylist_error, label="error fixed")
plt.legend()
plt.savefig("Lab2/figures/asierrorintran.png")

plt.show()

#ylist = ylist_error #update with error?


plt.plot(xlist, ylist)
plt.xlabel('Wavelength (nm)')
plt.ylabel('%T')
plt.title('Amorphous Silicon (aSi) Transmission')

plt.savefig("Lab2/figures/figure2.png")



indices = find_peaks(y, width = 4)[0]
x_max =[xlist[i] for i in indices]
y_max= [ylist[i] for i in indices]

l1 = 0
l2 = 0
n1 = 0
n2 = 0
d = 0
d_max = list()
d_error_max = list()

w_max = list()
n_error1 = 0

arr = np.array(x2)
for i in indices:
    l2 = l1 #move lamda1 to lamda2
    n2 = n1 #move refraction
    n_error2 = n_error1

    l1 = xlist[i] #get wavelength (lamda)
    difference_array = np.absolute(arr*1000 -l1)
    index = difference_array.argmin()
    n_error1 = n[index]
    n1 = N_func(l1)
    
    if(l2 != 0):
        w_max.append((l1+l2)/2)

        
        d = (l1* l2) / (2 *( n1 *l2 - n2*l1))
        print("d  is ", d)
        d_error = (l1* l2) / (2 *( n_error1 *l2 - n_error1*l1))

        print("de is ", d_error)
        
        d_error = (l1* l2) / (2 *( n_error1 *l2 - n_error1*l1))
        d_error_max.append(d_error)
        d_max.append(d)



plt.scatter(x_max, y_max) # plot maxes

l1 = 0
l2 = 0
n1 = 0
n2 = 0
d = 0
indices = find_peaks([ -x for x in ylist], width = 4)[0]
min_indices = indices
x_min =[xlist[i] for i in indices]
y_min= [ylist[i] for i in indices]
print("min")

d_min = list()
d_error_min = list()
w_min = list()
for i in indices:
    l2 = l1 #move lamda1 to lamda2
    n2 = n1 #move refraction
    n_error2 = n_error1
    l1 = xlist[i] #get wavelength (lamda)
    difference_array = np.absolute(arr*1000 -l1)
    index = difference_array.argmin()
    n_error1 = n[index]
    n1 = N_func(l1)
    if(l2 != 0):
        d = (l1* l2) / (2 *( n_error1 *l2 - n_error2*l1))
        d_min.append(d)
        w_min.append((l1+l2)/2)
        d_error = (l1* l2) / (2 *( n_error1 *l2 - n_error1*l1))
        d_error_min.append(d_error)
        print("d  is ", d)
        print("de is ", d_error)

plt.scatter(x_min, y_min) #plot mins 

plt.savefig("Lab2/figures/AmorphousSilicon.png")

plt.show()


TM_func = interp1d(x_max,y_max,bounds_error=False)
Tm_func = interp1d(x_min,y_min,bounds_error=False)
xnew = np.arange(104,2200 , 0.1)

TM_func_points = TM_func(xnew)
Tm_func_points = Tm_func(xnew)
plt.plot(xnew, TM_func_points)
plt.plot(xnew, Tm_func_points)
plt.plot(xlist, ylist)
plt.savefig("Lab2/figures/AmorphousSiliconTMTm.png")

plt.show()
l1 = 0
l2 = 0
n1 = 0
n2 = 0
d = 0
d_min = list()
w_min = list()
print("using N")
for i in min_indices:
    l2 = l1 #move lamda1 to lamda2
    n2 = n1 #move refraction
    n_error2 = n_error1
    l1 = xlist[i] #get wavelength (lamda)
    difference_array = np.absolute(arr*1000 -l1)
    index = difference_array.argmin()
    n_error1 = n[index]
    
    s = N_func(l1)
    N = 2*s*(TM_func(l1)-Tm_func(l1))/(TM_func(l1)*Tm_func(l1)) + (pow(s,2)+1)/2
    n1=pow(N + pow((pow(N,2)-pow(s,2)),1/2),1/2)
    #n1 = N_func(l1)
    if(l2 != 0):
        d = (l1* l2) / (2 *( n_error1 *l2 - n_error2*l1))
        d_min.append(d)
        w_min.append((l1+l2)/2)
        d_error = (l1* l2) / (2 *( n_error1 *l2 - n_error1*l1))
        #d_error_min.append(d_error)
        print("d  is ", d)
plt.plot(w_min, d_min,label = "minima")
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Calculate Length')
plt.title('Amorphous Silicon (aSi) Lengths')
plt.savefig("Lab2/figures/asilengthminmaN.png")
plt.show()

#make k si refence image
[x,y,header] = fileRead("Lab2/refdata/SI-k.csv");
plt.plot(x, y)
plt.xlabel('Wavelength (nm)')
plt.ylabel('k')
plt.title('Silicon extinction coefficient values')
plt.savefig("Lab2/figures/Sik.png")
plt.show()



#[min_wavel,min_length,header] = fileRead("Lab2/pydata/minlength.csv",offset=0);

#[max_wavel,max_length,header] = fileRead("Lab2/pydata/maxlength.csv",offset=0);

#plt.plot(min_wavel, min_length,label = "minima")
#plt.plot(max_wavel, max_length,label = "maxima")
#plt.xlabel('Wavelength (nm)')
#plt.ylabel('Calculate Length')
#plt.title('Amorphous Silicon (aSi) Lengths')

#plt.legend()
#plt.savefig("Lab2/figures/asilength.png")
#plt.show()

plt.plot(w_min, d_min,label = "minima")
plt.plot(w_max, d_max,label = "maxima")
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Calculate Length')
plt.title('Amorphous Silicon (aSi) Lengths')
plt.savefig("Lab2/figures/asilengthN.png")

plt.show()

plt.plot(w_min, d_min,label = "minima")
plt.plot(w_max, d_max,label = "maxima")
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Calculate Length')
plt.title('Amorphous Silicon (aSi) Lengths')
plt.savefig("Lab2/figures/asilength.png")

rows = zip(np.around(w_min,3), np.around(d_min,3))
header = ["wavelength","d (minima)"]
with open("Lab2/pydata/asilengthmin.csv", "w",newline="") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
rows = zip(np.around(w_max,3), np.around(d_max,3))
header = ["wavelength","d (maxima)"]
with open("Lab2/pydata/asilengthmax.csv", "w",newline="") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
plt.show()