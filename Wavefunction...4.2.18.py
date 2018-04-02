import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
ax = plt.axes(xlim=(0, 500), ylim=(-0.5, .5))
line, = ax.plot([], [], lw=2)

def PIB_Func(x, n, L):
    psi_n = np.sqrt(2./L)*np.sin(2*np.pi*x/L)+np.sqrt(2./L)*np.sin(5*np.pi*x/L)+np.sqrt(2./L)*np.sin(7*np.pi*x/L)
    return psi_n
def PIB_En(n, L):
    En = (n*n * np.pi*np.pi)/(2*L*L)
    return En
def PIB_Time(n, L, t):
    E = PIB_En(n, L)
    ci = 0.+1j
    phi_n_t = np.exp(-1*ci*E*t)
    ### Write code here to define phi_n_t
    return phi_n_t

L = 500.
xt = np.linspace(0, L, 2000)
psi_exp = np.zeros(len(xt),dtype=complex)
n=1



psi = PIB_Func(x, n, L)
p=psi*psi
plt.plot(x, np.real(psi), 'purple', x,p, 'red') #P,'green', x, psi_1, 'orange', z,'blue') 
plt.show()

nt = np.linspace(1,200,200)
cn = FourierAnalysis(x, n, L)



def init():
    line.set_data([], [])
    return line,
"""
Created on Mon Apr  2 14:52:43 2018

@author: eldabaghr
"""