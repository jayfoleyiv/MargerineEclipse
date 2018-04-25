import numpy as np
from matplotlib import pyplot as plt

def PIB_Func(x, n, L):
    return np.sqrt(2./L)*np.sin(n*np.pi*x/L)
def Gauss_Packet(sig, x, x0 , k0):
    ci= 0+1j
    pre=1/(sig*np.sqrt(2*np.pi))
    gx=np.exp(-0.5*((x-x0)/sig)**2)
    pw=np.exp(ci*k0*x)
    return pre*gx*pw
                
#Hhat=(-(1.054e-34)/(2*9.109e-31)
#psin=((np.sqrt(2/L))*np.sin((n*np.pi*x)/L))
#z=Hhat*np.diff(psin)

#def PIB_Func(x, n, L):
    #return np.sqrt(2./L)*np.sin(n*np.pi*x/L)
    #psi_n = np.sqrt(2./L)*np.sin(n*np.pi*x/L)
    #return psi_n

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

L=500
sig=20.
k0=0.4
x0=200
x=np.linspace(0,L,1000)
y=Gauss_Packet(sig ,x, x0, k0)
phi1 = PIB_Func(x,1,L)
phi2 = PIB_Func(x,2,L)
phi3 = PIB_Func(x,3,L)
phi4 = PIB_Func(x,4,L)
phi5 = PIB_Func(x,5,L)
phi6 = PIB_Func(x,6,L)
phi7 = PIB_Func(x,7,L)

N = np.linspace(1,100,100)
#psi_exp = np.array(x,y,)
cn=FourierAnalysis(x,y,N,L)

c1 = np.sqrt(1./6)

psi_sum = c1*phi1 + c1*phi2 + c1*phi3 + c1*phi4 + c1*phi5 + c1*phi6 + c1*phi7


P=np.real(np.conj(y)*y)
                
#print(cn)

psi_exp = np.zeros(len(x))

for i in range(0, len(cn)):
    psi_exp = psi_exp + cn[i]*PIB_Func(x, i+1, L)
    plt.plot(x,y,'red', x, np.real(psi_exp), 'purple')
    plt.show()
    
    
#print(Hhat,psin,z)
plt.plot(x,y,'red', x, np.real(psi_exp), 'purple') #P,'green', x, psi_1, 'orange', z,'blue') 
plt.show()

"""
Spyder Editor

This is a temporary script file.
"""

