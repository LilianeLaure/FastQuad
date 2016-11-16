"""
Created on Wed Oct 09 14:50:38 2016

MonteCarlo

formula taken from:
http://mathfaculty.fullerton.edu/mathews/n2003/montecarlomod.html

Case Interval [a,b] x [c,d] ; 1D Triangle

"""

import random 
import numpy as np
import math
import matplotlib.pyplot as plt

N = 1000
a = np.array([0,0])
b = np.array([0,1])
c = np.array([0,0])
d = np.array([1,0])


# Transformation triangle Phi(x,y) = (x, (1-x)y)
def transformToTriangle(point):
	return np.array([point[0], (1 - point[0])*point[1]])


#---------------------------------------------------
# Case 1: regulare function polynom f(x) = x^2 + y^2
#---------------------------------------------------


def monteCarlo_Regul_Tria(a, b, c, d, N):

	a = transformToTriangle(a)
	b = transformToTriangle(b)
	c = transformToTriangle(c)
	d = transformToTriangle(d)
	
	a = 0
	b = 1
	c = 0
	d = 1	
	
	print "a="+str(a)+", b="+str(b)+", c="+str(c)+", d="+str(d)

	randX = np.random.uniform(0, 1, N)
	randY = np.zeros(N)
	#print randValuesX
	for i,k in enumerate(randX):
		randY[i]=np.random.uniform(0,1-k)
		
		
	f_moy = (sum((randValuesX**2 + (1-randValuesX)**2 * randValuesY**2)*np.abs(1-randValuesX)))/N
	#print "f_moy",f_moy	
	f_int = (b-a)*(d-c)*f_moy # integral_a^b f(x) dx = (b-a)*f-moy
	#print "f_int=", f_int
	f_moy2 = (sum((randValuesX**2 + (1-randValuesX)**2 * randValuesY**2)*np.abs(1-randValuesX))**2)/N # f^2_moy = 1/N * sum(f^2(x))
	#print "f_moy2",f_moy2
	error = (b-a)*(d-c) * np.sqrt((f_moy2 - f_moy**2)/N)
	
	return error
	

figure()
errVec2 = np.array([])
print " "
print "Case regular polynom"

err2 = monteCarlo_Regul_Tria(a,b, c, d, N)
errVec2 = np.append(errVec2, err2)
#print("n:"+str(n)+"|| error:"+str(err))


for n in range(1,N):
	err2 = monteCarlo_Regul_Tria(a,b, c, d, n)
	errVec2 = np.append(errVec2, err2)
	#print("n:"+str(n)+"|| error:"+str(err))


plt.plot(np.log(range(0,N)), np.log(errVec2), "r", label='error')
#plt.plot( np.log(N), np.log(10/np.sqrt(subInterval)*1/N + 0.0085), "b", label='$c/\sqrt{n}x+A$')
plt.title("Case 2D triangular regular polynom $x^{2}+y^{2}$")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
plt.legend(loc=1)
plt.show()

