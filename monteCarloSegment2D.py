# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:27:20 2016

@author: 3503833
"""



import random 
import numpy as np
import math
import matplotlib.pyplot as plt

N = 10
eps = 1

a1 = np.array([-eps , -1])
b1 = np.array([-eps, 1])
a2 = np.array([eps, -1])
b2 = np.array([eps, 1])


#---------------------------------------------------
# Case 1: regulare function polynom f(x) = x^2 + y^2
#---------------------------------------------------

def param(t, s, eps):

	return (-eps + 2*t*eps, -1 + 2*s)

def monteCarlo_Regul_Segment(eps, N):


	randValues = np.random.uniform(low=0,high=1,size=(N,2))

	randX = randValues[:,0]
	randY = randValues[:,1]
		
	print "randX"
	print randX
	print " " 
	print "randY"
	print randY
		
	phiX, phiY = param(randX, randY, eps)		
		
	f_moy = 4*eps(sum( phiX**2 + phiY**2 ))/N
	print "f_moy",f_moy	
	#print "f_int=", f_int
	f_moy2 = (sum((param(a1, b1, randX)**2 + param(a2, b2, randY)**2)**2))/N # f^2_moy = 1/N * sum(f^2(x))
	print "f_moy2",f_moy2
	

	error = np.sqrt(sum( (param(a1, b1, randX)**2 + param(a2, b2, randY)**2 - f_moy)**2 )/(N-1)	)/np.sqrt(N)
	#error = (b-a)*(d-c) * np.sqrt((f_moy2 - f_moy**2)/N)
	
	return error
	

figure()
errVec2 = np.array([])
print " "
print "Case regular polynom"


for n in range(1,N):
	err2 = monteCarlo_Regul_Segment(a1, b1, a2, b2, n)

	errVec2 = np.append(errVec2, err2)
	#print("n:"+str(n)+"|| error:"+str(err))


plt.plot(np.log(N), np.log(errVec2), "r", label='error')
#plt.plot( np.log(N), np.log(10/np.sqrt(subInterval)*1/N + 0.0085), "b", label='$c/\sqrt{n}x+A$')
plt.title("Case 2D triangular regular polynom $x^{2}+y^{2}$")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
plt.legend(loc=1)
plt.show()

