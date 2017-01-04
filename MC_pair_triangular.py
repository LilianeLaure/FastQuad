# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:09:50 2016

@author: Adrian Ahne

Calculate integral of function over pair of triangular
"""


import random 
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA



# returns points chosen randomly in the triangle A, B, C
def monteCarlo_Regul_Tria(A,B,C,N):
    r1 = np.random.uniform(0, 1, N)
    r2 = np.random.uniform(0, 1, N)
    x=(1-np.sqrt(r1))*A[0]+(np.sqrt(r1)*(1-r2))*B[0]+(r2*np.sqrt(r1))*C[0]
    y=(1-np.sqrt(r1))*A[1]+(np.sqrt(r1)*(1-r2))*B[1]+(r2*np.sqrt(r1))*C[1]
    P=np.array([x,y])     
    return P

# returns error for function f(x, y) = x^2 + y^2
def monteCarlo_Tria_regul(P1,P2,N):
	
	x1 = P1[0]
	y1 = P1[1]
	
	x2 = P2[0]
	y2 = P2[1]

	f_moy=sum(x1**2 + y1**2 + x2**2 + y2**2)/N
	#print "f_moy", f_moy
	#f_moy2=sum((x1**2 + y1**2 + x2**2 + y2**2)**2)/2
	#err=np.sqrt(np.abs((f_moy2-f_moy**2)/N))
	Var=np.abs((1/N)*sum((x1**2 + y1**2 + x2**2 + y2**2)**2)-f_moy**2)
	err = np.sqrt(Var/N)*1.96
	return err


# returns error for function f(x, y) = 1/|x-y|
def monteCarlo_Tria_sing(P1,P2,N):
	
	x1 = P1[0]
	y1 = P1[1]
	
	x2 = P2[0]
	y2 = P2[1]
	f_moy=sum(1/(np.sqrt((x1-x2)**2 + (y1-y2)**2)))/N
	#print "f_moy", f_moy
	#f_moy2=sum((1/(np.abs(X-Y)))**2)/2
	#err=np.sqrt(np.abs((f_moy2-f_moy**2)/N))
	Var=np.abs((1/N)*sum((1/(np.sqrt((x1-x2)**2 + (y1-y2)**2)))**2)-f_moy**2)
	err = np.sqrt(Var/N)*1.96
	return err
'''

# calculate area in the triangle (Shoelace_formula)
def calculateTriangleArea(A, B, C):
	x1, y1 = A
	x2, y2 = B
	x3, y3 = C
	return 0.5*(x1*y2 + x2*y3 + x3*y1 - x2*y1 - x3*y2 - x1*y3)
'''
# ----------------------------------------------------------------
# Calculate and plots
#-----------------------------------------------------------------

# number of points
N=1000

# triangle 1
A1=np.array([1,1])
B1=np.array([4,1])
C1=np.array([5,3])
print("Triangle( A1="+str(A1)+", B1="+str(B1)+", C1="+str(C1)+")")

# triangle 2
A2=np.array([-3,-3])
B2=np.array([1,4])
C2=np.array([-5,0])
print("Triangle( A2="+str(A2)+", B2="+str(B2)+", C2="+str(C2)+")")
'''
# choose points randomly in triangle
Point=monteCarlo_Regul_Tria(A,B,C,N)


plt.figure()
plt.scatter(A[0], A[1])
plt.scatter(B[0], B[1])
plt.scatter(C[0], C[1])
plt.scatter(Point[0],Point[1])
#plt.show()
'''

err_Reg=np.array([])
err_Sing = np.array([])

for n in range(100,N):
	# choose points randomly in triangle
	P1=monteCarlo_Regul_Tria(A1,B1,C1,n)	# triangle 1
	P2=monteCarlo_Regul_Tria(A2,B2,C2,n)	# triangle 2

	# calculate error
	err_Reg = np.append(err_Reg,monteCarlo_Tria_regul(P1, P2, n))
	err_Sing = np.append(err_Sing,monteCarlo_Tria_sing(P1, P2, n))


# plot regular case
plt.figure()
plt.plot(log(range(100,N)),log(err_Reg),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))+3.84, "r")
plt.title("Pair of triangles regular polynom $x^{2}+y^{2}$|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")

# plot singular case
plt.figure()
plt.plot(log(range(100,N)),log(err_Sing),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))-0.989999, "r")
plt.title("Pair of triangles singular $1/(|x-y|)$ || pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")

plt.show()
