import matplotlib.pyplot as plt
from PythonModule.filemanager import fileRead

[x,y,header] = fileRead("Lab2/data/aSi_T.csv",offset=2);



plt.plot(x, y)
plt.xlabel('Wavelength (nm)')
plt.ylabel('%T')
plt.title('Amorphous Silicon (aSi) Transmission')

plt.savefig("Lab2/figures/figure2.png")

#plt.show()