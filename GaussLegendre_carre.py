#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:23:31 2016

@author: 3417212
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

"""
on se place sur [-1;1]x[-1;1]
"""
def somme_de_GaussLegendre2D(N, alpha1, beta1, alpha2, beta2,func ):
    x,y = np.polynomial.legendre.leggauss(N)
    
    #homothétie de [-1;1] vers [alpha;beta]
    x1 = (beta1 - alpha1)/2 * x + (alpha1 + beta1)/2
    x2 = (beta2 - alpha2)/2 * x + (alpha2 + beta2)/2
    I_exp = (beta1-alpha1)*sum(y*func(x))/2.
    return I_exp
    
def estimateur_erreur(N, f, alpha , beta ):
    Sf = somme_de_GaussLegendre(N,alpha, beta, f)
    #print "Sf = ", Sf, " pour n = ",N
    Sf2 = somme_de_GaussLegendre(N, alpha, (alpha + beta)/2., f) + somme_de_GaussLegendre(N, (alpha + beta)/2., beta, f)
    #print "Sf2 = ", Sf2, " pour n = ",N
    return Sf - Sf2

def erreur_de_GaussLegendre2D(N, func, SS):
    X,Y = np.polynomial.legendre.leggauss(N)
    I_exp = 0
    for i1 in range(len(X)):
        for i2 in range(len(X)):
            I_exp += Y[i1] * Y[i2] * func(X[i1],X[i2])
        
    print "N:"+str(N)
    print "I_exp:"+str(I_exp)
    print "I_th:"+str(SS)
    print "abs(I_th - I_exp):"+str(abs(SS - I_exp))
    print ""   
    return abs(SS - I_exp)

def GaussLegendre2D(N, func):
    X,Y = np.polynomial.legendre.leggauss(N)
    I_exp = 0
    for i1 in range(len(X)):
        for i2 in range(len(X)):
            I_exp += Y[i1] * Y[i2] * func(X[i1],X[i2])
    print "N:"+str(N)
    print "I_exp:"+str(I_exp)
    print ""
    return I_exp
        
def f(x,y):
    return x**2 + y**2
    
def h(x,y):
    return x/(1+x*y)

N = np.linspace(1, 100, 100)
valh = errh=errg = errf = np.array([])

for n in N:
    #errf = np.append(errf, erreur_de_GaussLegendre2D(n, f, 8.0/3))
    errh = np.append(errh, erreur_de_GaussLegendre2D(n, h,0.86))

"""
fig4=plt.figure()
plt.plot(np.log(N), np.log(errf), 'r')
plt.xlabel('log(N)')
plt.ylabel('log(errf)')
fig4.suptitle('erreur de Gauss Legendre pour f(x,y)=x²+y²', fontsize=20)
fig4.savefig('f2D.jpg')
plt.show()"""

fig5=plt.figure()
plt.plot(np.log(N), np.log(errh), 'b')
plt.xlabel('log(N)')
plt.ylabel('log(errh)')
plt.show()