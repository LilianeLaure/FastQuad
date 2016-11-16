#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def erreur_de_GaussLegendre(N, func, SS):
    x,y = np.polynomial.legendre.leggauss(N)
    I_exp = sum(y*func(x))
    print "N:"+str(N)
    print "I_exp:"+str(I_exp)
    print "I_th:"+str(SS)
    print "abs(I_th - I_exp):"+str(abs(SS - I_exp))
    print ""   
    return abs(SS - I_exp)

def f(x):    
    return 4

def g(x):
    return x**50

def h(x):
	return np.absolute(x)


N = np.linspace(1, 100, 100)
errh=errg = errf = np.array([])

for n in N:
    #errf = np.append(errf, erreur_de_GaussLegendre(n, f, 8))
    #errg = np.append(errg, erreur_de_GaussLegendre(n, g, 2.0/51))
    if (mod(n,2) == 0):
        errh = np.append(errh, erreur_de_GaussLegendre(n, h, 1))
	#print(h(n))
   
"""
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
plt.show()
#...
