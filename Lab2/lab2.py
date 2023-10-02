
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from PythonModule.filemanager import fileRead


from scipy.signal import find_peaks

[x,y,header] = fileRead("Lab2/data/aSi_T.csv",offset=2);
[x2,n,header] = fileRead("Lab2/data/aSi_RefractiveIndexINFO.csv");

[xerror,yerror,header] = fileRead("Lab2/data/aSi_T_instrument_error.csv",offset=2);
xerror = np.array(xerror)
yerror = np.array(yerror)

ylist = np.array(y) #-yerror
xlist =  np.array(x)

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
#plt.plot(xlist, ylist, label="no error corectio")
plt.plot(xlist, ylist_error, label="error fixed")
plt.legend()
plt.show()

#ylist = ylist_error
plt.plot(xlist, ylist)
plt.xlabel('Wavelength (nm)')
plt.ylabel('%T')
plt.title('Amorphous Silicon (aSi) Transmission')

plt.savefig("Lab2/figures/figure2.png")

indices = find_peaks(y, width = 4)[0]
x =[xlist[i] for i in indices]
y= [ylist[i] for i in indices]

l1 = 0
l2 = 0
n1 = 0
n2 = 0
d = 0
arr = np.array(x2)
for i in indices:
    l2 = l1 #move lamda1 to lamda2
    n2 = n1 #move refraction
    
    l1 = xlist[i] #get wavelength (lamda)
    difference_array = np.absolute(arr*1000 -l1)
    index = difference_array.argmin()
    n1 = n[index]
    if(l2 != 0):
        d = (l1* l2) / (2 *( n1 *l2 - n2*l1))
        print("d is ", d)
    



plt.scatter(x, y) # plot maxes

l1 = 0
l2 = 0
n1 = 0
n2 = 0
d = 0
indices = find_peaks([ -x for x in ylist], width = 4)[0]
x =[xlist[i] for i in indices]
y= [ylist[i] for i in indices]
print("min")
for i in indices:
    l2 = l1 #move lamda1 to lamda2
    n2 = n1 #move refraction
    
    l1 = xlist[i] #get wavelength (lamda)
    difference_array = np.absolute(arr*1000 -l1)
    index = difference_array.argmin()
    n1 = n[index]
    if(l2 != 0):
        d = (l1* l2) / (2 *( n1 *l2 - n2*l1))
        print("d is ", d)
plt.scatter(x, y) #plot mins 

plt.savefig("Lab2/figures/AmorphousSilicon.png")

plt.show()
#make k si refence image
[x,y,header] = fileRead("Lab2/refdata/SI-k.csv");
plt.plot(x, y)
plt.xlabel('Wavelength (nm)')
plt.ylabel('k')
plt.title('Silicon extinction coefficient values')
plt.savefig("Lab2/figures/Sik.png")
plt.show()



