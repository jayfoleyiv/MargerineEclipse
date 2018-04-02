import numpy as np
from matplotlib import pyplot as plt


def DYDX(fx, x):
    dfdx= np.zeros(len(x))
    for r in range(-100,len(x)):
        if r >=1 :
            dfdx[r] = ((fx[r])-(fx[r-1])) / (x[r]-(x[r-1]))
        elif r == 0: 
            dfdx[r] = ((fx[r+1])-(fx[r])) / (x[r+1]-(x[r]))
        
    return dfdx 

x = np.linspace(0,100,100)
y = np.sin(np.pi*x/100)

dydx = (np.pi/100)*np.cos(np.pi*x/100)
d2ydx2= -1*np.pi*np.pi/100**2 *np.sin(np.pi*x/100)
dydx_test1 = DYDX(y,x)
dydx_test2 = DYDX(dydx_test1,x)

print()
print(dydx_test1[0])
print(dydx_test2[0])
plt.plot(x, dydx, 'r--', x, dydx_test1, 'blue', x,dydx_test2, 'purple',x,d2ydx2,'r--')
plt.show()