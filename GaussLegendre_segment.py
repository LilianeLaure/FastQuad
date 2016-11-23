#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

def somme_de_GaussLegendre(N, alpha, beta, func ):
    x,y = np.polynomial.legendre.leggauss(N)
    
    #homothétie de [-1;1] vers [alpha;beta]
    x = (beta - alpha)/2 * x + (alpha + beta)/2
    
    I_exp = (beta-alpha)*sum(y*func(x))/2.
    return I_exp
    
def estimateur_erreur(N, f, alpha , beta ):
    Sf = somme_de_GaussLegendre(N,alpha, beta, f)
    #print "Sf = ", Sf, " pour n = ",N
    Sf2 = somme_de_GaussLegendre(N, alpha, (alpha + beta)/2., f) + somme_de_GaussLegendre(N, (alpha + beta)/2., beta, f)
    #print "Sf2 = ", Sf2, " pour n = ",N
    return Sf - Sf2


def f(x):    
    return 4

def g(x):
    return x**50

def h(x):
	return np.absolute(x)


N = np.linspace(1, 100, 100)
errh=errg = errf = np.array([])


alpha=-1
beta=1
for n in N:
    errf = np.append(errf, estimateur_erreur(n,f, alpha, beta))
    errg = np.append(errg, estimateur_erreur(n,g, alpha, beta))
    if (mod(n,2) == 0):
        errh = np.append(errh, estimateur_erreur(n,h, alpha, beta))
	#print(h(n))

"""
print "erreur par rapport à f : ", errf
print "erreur par rapport à g : ", errg
print "erreur par rapport à h : ", errh


fig1=plt.figure()
plt.plot(np.log(N), np.log(errg), 'r')
plt.xlabel('log(N)')
plt.ylabel('log(errg)')
fig1.suptitle('erreur de Gauss Legendre pour g(x)=x^50', fontsize=20)
fig1.savefig('g.jpg')
plt.show()

fig2=plt.figure()
plt.plot(np.log(N), np.log(errf), 'b')
plt.xlabel('log(N)')
plt.ylabel('log(errf)')
fig2.suptitle('erreur de Gauss Legendre pour f(x)=4', fontsize=20)
fig2.savefig('f.jpg')
plt.show()
"""


fig3=plt.figure()
plt.plot(np.log(N[0:-1:2]), np.log(errh), 'b')
plt.xlabel('log(N)')
plt.ylabel('log(errh)')
fig3.suptitle('erreur de Gauss Legendre pour h(x)=|x|', fontsize=20)
plt.show()
#...
