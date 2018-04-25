# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:57:21 2018

@author: plattelr
"""

import numpy as np
from matplotlib import pyplot as plt


def DYDX(fx, x):
    dfdx= np.zeros(len(x))
    for r in range(-100,len(x)):
        if r >=1 :
            dfdx[r] = ((fx[r])-(fx[r-1])) / (x[r]-(x[r-1]))
        elif r == 0: 
            dfdx[r] = ((fx[r])-(fx[r])) / (x[r]-(x[r]))
        
    
    
    return dfdx

x = np.linspace(0,100,100)
y = np.sin(np.pi*x/100)

dydx = (np.pi/100)*np.cos(np.pi*x/100)

dydx_test = DYDX(y,x)

print()

plt.plot(x, dydx, 'r--', x, dydx_test, 'blue')
plt.show()

