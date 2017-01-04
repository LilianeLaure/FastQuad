# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:50:38 2016

MonteCarlo

formula taken from:
http://mathfaculty.fullerton.edu/mathews/n2003/montecarlomod.html

Case Interval [a,b] ; 1 Dimension

"""

import random 
import numpy as np
import math
import matplotlib.pyplot as plt

N = np.linspace(1, 500000, 200)
a = 0
b = 1

# Case 1: regulare function constant f(x) = 4

def monteCarlo_Regul_Const(a, b, N):
	
	randValues = np.random.uniform(a, b, N)
	f_moy = 4*N/N
	f_int = (b-a)*f_moy # integral_a^b f(x) dx = (b-a)*f-moy
	
	f_moy2 = 4**2*N/N # f^2_moy = 1/N * sum(f^2(x))
	
	error = (b-a) * np.sqrt((f_moy2 - f_moy**2)/N)
	
	return error


print "Case regular constant"
for n in N:
	err = monteCarlo_Regul_Const(a,b,n)
	#print("n:"+str(n)+"|| error:"+str(err))




# Case 2: regulare function polynom f(x) = x^2

def monteCarlo_Regul_poly(a, b, N):
	
	randValues = np.random.uniform(a, b, N)
	f_moy = (sum(randValues**2))/N
	f_int = (b-a)*f_moy # integral_a^b f(x) dx = (b-a)*f-moy
	
	f_moy2 = (sum(randValues**4))/N # f^2_moy = 1/N * sum(f^2(x))
	
	error = (b-a) * np.sqrt((f_moy2 - f_moy**2)/N)
	
	return error
	'''
figure()
errVec2 = np.array([])
print " "
print "Case regular polynom"
for n in N:
	err2 = monteCarlo_Regul_poly(a,b,n)
	errVec2 = np.append(errVec2, err2)
	#print("n:"+str(n)+"|| error:"+str(err))

plt.plot(log(N), log(errVec2))
plt.title("Case 1D regular polynom $x^{2}$")
plt.xlabel("log(N)")
plt.ylabel("log(err)")'''

# Case 2: singular function  f(x) = ln(|x|)

def monteCarlo_Sing(a, b, N):
	
	randValues = np.random.uniform(a, b, N)
	f_moy = sum(np.log(np.abs(randValues)))/N
	f_int = (b-a)*f_moy # integral_a^b f(x) dx = (b-a)*f-moy
	
	f_moy2 = sum(np.log(np.abs(randValues))**2)/N # f^2_moy = 1/N * sum(f^2(x))
	
	error = (b-a) * np.sqrt((f_moy2 - f_moy**2)/N)
	
	return error


figure()
errVec3 = np.array([])
print " "
print "Case singular"
for n in N:
	err = monteCarlo_Sing(a,b,n)
	errVec3 = np.append(errVec3, err)

	#print("n:"+str(n)+"|| error:"+str(err))

plt.plot(log(N), log(errVec3) )
plt.title("Case 1D singular $ln(|x|)$")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
