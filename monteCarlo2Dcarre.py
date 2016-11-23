# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:50:38 2016

MonteCarlo

formula taken from:
http://mathfaculty.fullerton.edu/mathews/n2003/montecarlomod.html

Case Interval [a,b] x [c,d] ; 2D Carré

"""

import random 
import numpy as np
import math
import matplotlib.pyplot as plt

subInterval = 100
N = np.arange(0, 50000, 100)
a = 0
b = 1
c = 0
d = 1

#a = np.random.uniform(a,b, (5,2))
#print a
#print a[:,0]


#---------------------------------------------------
# Case 1: regulare function polynom f(x) = x^2 + y^2
#---------------------------------------------------

def monteCarlo_Regul_poly(a, b, c, d, N):
	
	randValuesX = np.random.uniform(a, b, N)
	randValuesY = np.random.uniform(c, d, N)
	
	f_moy = (sum(randValuesX**2 + randValuesY**2))/N
	f_int = (b-a)*(d-c)*f_moy # integral_a^b f(x) dx = (b-a)*f-moy
	
	f_moy2 = (sum((randValuesX**2 + randValuesY**2)**2))/N # f^2_moy = 1/N * sum(f^2(x))
	
	error = (b-a)*(d-c) * np.sqrt((f_moy2 - f_moy**2)/N)
	
	return error
	
	

figure()
errVec2 = np.array([])
print " "
print "Case regular polynom"

for n in N:
	err2 = monteCarlo_Regul_poly(a,b, c, d, n)
	errVec2 = np.append(errVec2, err2)
	#print("n:"+str(n)+"|| error:"+str(err))


plt.plot(np.log(N), np.log(errVec2), "r", np.log(N), -0.5*np.log(N) - 0.88, "b")
#plt.plot( np.log(N), np.log(10/np.sqrt(subInterval)*1/N + 0.0085), "b", label='$c/\sqrt{n}x+A$')
plt.title("Case 2D regular polynom $x^{2}+y^{2}$")
plt.xlabel("log(N)")
plt.ylabel("log(err)")

#plt.legend(loc='best')
plt.show()


"""
#-------------------------------- (b-a)*(d-c)*f_moy----------
# Case 2: singular function  f(x) = 1/|x-y|
#------------------------------------------

def monteCarlo_Sing(a, b, c, d, N):
	
	randValuesX = np.random.uniform(a, b, N)
	randValuesY = np.random.uniform(c, d, N)
	
	f_moy = sum(1/(np.abs(randValuesX-randValuesY)))/N
	f_int = (b-a)*(d-c)*f_moy # integral_a^b f(x) dx = (b-a)*f-moy
	
	f_moy2 = sum(1/(np.abs(randValuesX-randValuesY)**2))/N # f^2_moy = 1/N * sum(f^2(x))
	
	error = (b-a)*(d-c)*np.sqrt((f_moy2 - f_moy**2)/N)
	
	return error


figure()
errVec3 = np.array([])
print " "
print "Case singular"
for n in N:
	err = monteCarlo_Sing(a,b,c,d,n)
	errVec3 = np.append(errVec3, err)

	#print("n:"+str(1/n)+"|| error:"+str(err))

plt.plot(log(N), errVec3 )
plt.title("Case 2D singular $1/(|x-y|)$")
plt.xlabel("log(N)")
plt.ylabel("err")
plt.show()"""



#-------------------------------- (b-a)*(d-c)*f_moy----------
# Case 3: singular function  f(x) = -ln|x-y|
#------------------------------------------

eps = 1e-5

def monteCarlo_Sing(a, b, c, d, N):
	
	randValuesX = np.random.uniform(a, b, N)
	randValuesY = np.random.uniform(c, d, N)
	
	#print "n: ",n
	for i in range(1, len(randValuesX)):
		if abs(randValuesX[i] - randValuesY[i]) < eps:
			#print "i="+str(i)+" X="+str(randValuesX[i])+" Y="+str(randValuesY[i])
			np.delete(randValuesX, i)
			np.delete(randValuesY, i)
	
	nPoints = len(randValuesX)	
	
	f_moy = sum(-np.log(np.abs(randValuesX-randValuesY)))/nPoints
	print("fmoy:", f_moy)
	f_int = (b-a)*(d-c)*f_moy # integral_a^b f(x) dx = (b-a)*f-moy
	print("fint:", f_int)
	f_moy2 = sum(-np.log(np.abs(randValuesX-randValuesY)**2))/nPoints # f^2_moy = 1/N * sum(f^2(x))
	print("fmoy2:", f_moy2)
	error = (b-a)*(d-c)*np.sqrt((f_moy2 - f_moy**2)/nPoints)
	print("error:", error)
	
	return error

"""
figure()
errVec3 = np.array([])
print " "
print "Case singular"
for n in N:
	err = monteCarlo_Sing(a,b,c,d,n)
	errVec3 = np.append(errVec3, err)

	#print("n:"+str(1/n)+"|| error:"+str(err))

plt.plot(log(N), log(errVec3), "b", log(N), -0.49*log(N)-0.25, "r" )
plt.title("Case 2D singular $ln(|x-y|)$ || pente = -0.49")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
plt.show()"""