import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
ax = plt.axes(xlim=(0, 500), ylim=(-0.5, .5))
line, = ax.plot([], [], lw=2)

def PIB_Func(x, n, L):
    psi_n = np.sqrt(2./L)*np.sin(n*np.pi*x/L) #THIS IS OUR WAVE SKELETON
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
x = np.linspace(0, L, 2000)
psi_exp = np.zeros(len(x),dtype=complex)
#n=5


psi = PIB_Func(x, 3, L) + PIB_Func(x, 6, L) #THIS ADDS 2 FUNCTIONS TOGETHER TO MAKE A SUPER POSTIION
p=psi*psi
plt.plot(x, np.real(psi), 'purple',) #x,p, 'red') #THIS STARED OUT P IS THE PROBABLITY

#nt = np.linspace(1,200,200)
#cn = FourierAnalysis(x, n, L)





#def init():
#    line.set_data([], [])
#    return line,