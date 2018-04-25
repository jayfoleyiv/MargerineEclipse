import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation




# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 500), ylim=(-0.4, 0.4)) #makes your x and y axis, zoom in by chaning the values 
line, = ax.plot([], [], lw=2) #to make line thicker, change lw

### Function that takes in a list of x-coordinates, the quantum number, and the length L
### and returns PIB energy eigenfunction
def PIB_Func(x, n, L):
    psi_n = np.sqrt(2./L)*np.sin(n*np.pi*x/L)
    return psi_n

### Function that takes in a list of x-coordinates, a central x value, and central momentum value, 
### and a standard deviation for the position and returns complex-valued Gaussian wavepacket
def Gauss_Packet(x, x0, sig, k0):
    ci = 0.+1j
    pre = 1./(sig*np.sqrt(2.*np.pi))
    psi_x = pre*np.exp(-0.5*( (x-x0)/sig )**2)*np.exp(ci*k0*x)
    return psi_x

### Given a complex-valued wavefunction (PsiX), list of x-coordinates (x) between 0 and L, 
### and list of quantum numbers, will return a list of complex expansion coefficients
### to expand PsiX in terms of PIB energy eigenfunctions
def FourierAnalysis(x, PsiX, n, L):
    cn = np.zeros(len(n),dtype=complex)
    dx = x[1]-x[0]
    for i in range (0,len(cn)):

      som = 0+0j
      psi_i = PIB_Func(x, n[i], L)

      for j in range (0, len(x)):
        som = som + psi_i[j]*PsiX[j]*dx

      cn[i] = som

    return cn

### Give a quantum number n and a length L, return the energy 
### of an electron in a box of length L in state n
def PIB_En(n, L): #energy eigenvalues=n- only good for electron-mass = 1 
    En = (n*n * np.pi*np.pi)/(2*L*L)
    return En

### Give the quantum number and the current time, evaluate the time-dependent part of the wavefunction at current time
### and return its value
def PIB_Time(n, L, t): #returns a complex number for a PIB time dependent 
    E = PIB_En(n, L)
    ci = 0.+1j
    phi_n_t = np.exp(-1*ci*E*t) #h=1
    ### Write code here to define phi_n_t
    return phi_n_t

### Given a vector of not-necessarily-normalized complex expansion coefficients
### return the normalized version

#### Initialize some variables/arrays that will be used by the animate function
L = 500.
xt = np.linspace(0, L, 2000)
psi_exp = np.zeros(len(xt),dtype=complex)
### Imaginary unit i
ci = 0.+1j
sig = 15
k0 = 60.*np.pi/L
x0 = 200
Psi = Gauss_Packet(xt, x0, sig, k0)

nt = np.linspace(1,200,200)
cn = FourierAnalysis(xt, Psi, nt, L)

##for i in range(0,len(cn)):
##  psi_exp = psi_exp + cn[i]*PIB_Func(x, qn[i], L)




# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially to generate the animation
def animate(i): #code that creates the movie 
    
    ### Once PIB_Func and PIB_En are defined, the following
    ### code can be used to plot the time-evolution of an energy eigenfunction

    ### Define x-grid - this will be for a particle in a box of length L=30 atomic units (Bohr radii)
    ### We will represent the function with 1000 grid points (dx = 30/1000)
    L = 500.
    x = np.linspace(0, L, 2000)

    ### Imaginary unit i
    ci = 0.+1j
    fwhm = 7*np.pi/L
    k0 = 5*np.pi/L
    psi_t = np.zeros(len(x),dtype=complex)
    #psi = PIB_Func(x, 2, L)
    #ft = PIB_Time(2, L, i)
    
    for j in range(0,len(cn)):
      psi = PIB_Func(x, nt[j], L) 
      ft  = PIB_Time(nt[j], L, 4*i)
      psi_t = psi_t + cn[j]*psi*ft
   
    psi_t_star = np.conj(psi_t)

    y = np.real(psi_t)
    z = np.imag(psi_t)
    p = np.real(psi_t_star * psi_t)
    line.set_data(x, y) #to get a stationary line, write p instead of y
    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init, #calls anime multiple times it increases i, results in an updated y, x=same
                               frames=10000, interval=20, blit=True)
### uncomment to save animation as mp4 
#anim.save('pib_wp.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
plt.show()


#### Static plot
#plt.plot(x, np.real(Psi), 'b--', x, np.real(psi_exp), 'red', x, P, 'black')
#plt.show()

​

"""
Created on Mon Feb 26 16:06:43 2018

@author: eldabaghr
"""

