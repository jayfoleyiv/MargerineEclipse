import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from numpy.random import choice


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 500), ylim=(-0.3, 0.3))
line, = ax.plot([], [], lw=2)



def PIB_Func(x, n, L):
        return np.sqrt(2/L)*np.sin(n*np.pi*x/L)

def Gauss_Packet(sig,x, x0,  k0):
        ci = 0 + 1j
        pre = 1/(sig*np.sqrt(2*np.pi))
        gx = np.exp(-0.5*((x-x0)/sig)**2)
        pw = np.exp(ci*k0*x)
        return pre*gx*pw

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

def PIB_En(n, L):
    En = (n*n * np.pi*np.pi)/(2*L*L)
    return En


def PIB_Time(n, L, t):
    E = PIB_En(n, L)
    ci = 0.+1j
    phi_n_t = np.exp(-1*ci*E*t)
    return phi_n_t


def Normalize(pu):
    som=0
    for i in range(0,len(pu)):
        som=som +pu[i]

    for i in range(0,len(pu)):
        temp=pu[i]/som
        pu[i]=temp
    return pu

'''
P = (np.conj(y))*y
list_of_candidates = x
Pn = Normalize(P)
pr = np.real(Pn)
### Draw a random number using probability distribution
draw = choice(list_of_candidates, 100, p=pr)
print(draw)
x0 = draw[0]    
NewWvfxn = Pos_Eignfn(sig, xs, x0)
plt.plot(x, np.real(y), 'green', x, np.real(P), 'orange')
plt.show()
'''
L = 500.

### If you initialize in state n with energy En, then 
### measure position, yielding value x0... expand the position eigenfunction
### at x0 in terms of the energy eigenfunctions... yielding arran of expansion
### coefficiencts cn... then measure energy again, yielding energy value Em
### then you have the following three pieces of data:
### 1. n -> quantum number of initial state (for time<30)
### 2. cn -> array of expansion coefficients for position eigenfunction (for 30<t<60)
### 3. m  -> quantum number of final state (for time>60)
xt = np.linspace(0,L,500)
n = np.linspace(1, 10,10)
y = PIB_Func(xt,6,L)
P = np.real(np.conj(y)*y)
cn = FourierAnalysis(xt, y, n, L)
print(cn)


list_of_candidates=xt
Pn=Normalize(P)
pr=np.real(P)
draw=choice(list_of_candidates,15,p=pr)
print("Measurement of position yielded x0 = ",draw[0])

PosEigenfxn = Gauss_Packet(0.5, xt, draw[0], 0)
cn2 = FourierAnalysis(xt, PosEigenfxn, n, L)

#energy=PIB_En(n, L)
#list_of_candidates2=energy
#Pn=Normalize(P)
#pr=np.real(P)
#draw2=choice(list_of_candidates2, 15, p=pr)
#print("Measurement of energy yielded Ex = ", draw2[0])

psi_exp = np.zeros_like(y)

for i in range (0,len(cn)):
    psi_exp = psi_exp + cn[i]*PIB_Func(xt, i+1, L)

def init():
    line.set_data([], [])
    return line,


# animation function.  This is called sequentially to generate the animation
def animate(i):
    
    ### Once PIB_Func and PIB_En are defined, the following
    ### code can be used to plot the time-evolution of an energy eigenfunction

    ### Define x-grid - this will be for a particle in a box of length L=30 atomic units (Bohr radii)
    ### We will represent the function with 1000 grid points (dx = 30/1000)
  
    ### when i < 30, animate your initial state (some energy eigenfunction)
    L = 500
    x = np.linspace(0,L,500)
    ### Imaginary unit i
    ci = 0.+1j    
    psi_t = np.zeros(len(x),dtype=complex)

    if i<30:
        for j in range(0,len(cn)):
            psi = PIB_Func(x, n[j], L)
            ft = PIB_Time(n[j], L, i*100)
            psi_t = psi_t +cn[j]*psi*ft
        psi_t_star = np.conj(psi_t)
        #psi_t = PIB_Func(x, 10, L)*PIB_Time(10, L, 10*i)
        #print(i)
        #print(PIB_Time(10,L,10*i))
        y = np.real(psi_t)
        z = np.imag(psi_t)
        p = np.real(y*y+z*z)
         
    
    ### at t=30, a position measurement was made... yielding x0 = 22
    ### so for t>30 && t<60, animate the position eigenfunction that is centered
    ### at x0 = 22
    elif i<60:
#        print(i)
        psi_t = np.zeros(len(x),dtype=complex)
        for g in range(len(cn2)):
            psi_t = psi_t + cn2[g]*PosEigenfxn*PIB_Time(n[g], L, i*100)
#        psi_t= PIB_Func(x,10,L)*PIB_Time(10, L, i*100)
        y = np.real(psi_t)
        z = np.imag(psi_t)
        p = np.real(y*y+z*z)
        
#    elif i>60:
#        psi_t = np.zeros(len(x),dtype=complex)
#        for g in range(len(cn2)):
#        psi_t = psi_t + cn2[g]*cn2[g]*PIB_Time(n[g], L, i*100)
#        y = np.real(psi_t)
#        z = np.imag(psi_t)
#        p = np.real(y*y+z*z)
    line.set_data(x, y)
    return line,

    ### at t=60, an energy is measured, yielding E0 = ....
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=10000, interval=200, blit=True)
### uncomment to save animation as mp4 
#anim.save('pib_wp.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
plt.show()


#lt.plot(x,np.real(psi_exp),'r--', x, np.real(y), 'blue')
#lt.show()
