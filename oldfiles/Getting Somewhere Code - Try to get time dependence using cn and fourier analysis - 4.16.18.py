import numpy as np
from numpy.random import choice
from matplotlib import pyplot as plt


#where does your system live? (0,L,100)
def Gaussian(ang_array,theta0,sigma):
    return np.exp(-(ang_array-theta0)**2/sigma**2)

def PIB_Func(x, n, L):
    psi_n = np.sqrt(2./L)*np.sin(n*np.pi*x/L) #THIS IS OUR WAVE SKELETON
    return psi_n
    
def PIB_Time(n, L, t):
    E = PIB_En(n, L)
    ci = 0.+1j
    phi_n_t = np.exp(-1*ci*E*t)
    ### Write code here to define phi_n_t
    return phi_n_t

def Normalize(pu):
    som=0
    for i in range(0,len(pu)):
        som=som +pu[i]

    for i in range(0,len(pu)):
        temp=pu[i]/som
        pu[i]=temp
    return pu

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

L=50

cn=FourierAnalysis(L/2, PsiX, 3, L)
print(cn)

x= np.linspace(0,L,1000)
y= PIB_Func(x,10,50)
y2= Gaussian(x, np.pi, 0.01)
P=np.conj(y)*y
P2=np.conj(y2)*y2

psi = PIB_Func(x, 3, L)*PIB_Time(3,L,300) + PIB_Func(x, 6, L)*PIB_Time(6,L,300)

list_of_candidates=x
Pn=Normalize(P)
pr=np.real(P)
draw=choice(list_of_candidates,15,p=pr)
print(draw)
plt.xlim(0,50)
plt.plot(x,np.real(y),'red',x,np.real(P),'blue')
plt.plot(x, np.real(y2), 'purple', x, np.real(P2), 'orange')
plt.show()
