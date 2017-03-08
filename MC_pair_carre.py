# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:15:38 2016

@author: 3201955
"""

import random
import numpy as np
import math
import matplotlib.pyplot as plt

N = 1000

#random points in triangle ABC
def monteCarlo_triangle(A,B,C,N):
    r1 = np.random.uniform(0,1,N)
    r2 = np.random.uniform(0,1,N)
    x=(1-np.sqrt(r1))*A[0]+(np.sqrt(r1)*(1-r2))*B[0]+(r2*np.sqrt(r1))*C[0]
    y=(1-np.sqrt(r1))*A[1]+(np.sqrt(r1)*(1-r2))*B[1]+(r2*np.sqrt(r1))*C[1]
    P=np.array([x,y])     
    return P
    
def monteCarlo_quadrangle(A,B,C,D,N):
    P = monteCarlo_triangle(A,B,C,N)
    Q = monteCarlo_triangle(A,D,C,N)
    R = np.append(P,Q, axis=1)
    return R



# returns error for function f(x, y) = x^2 + y^2
def monteCarlo_Rec_regul(P1,P2,N):
	
	x1 = P1[0]
	y1 = P1[1]
	
	x2 = P2[0]
	y2 = P2[1]

	f_moy=sum(x1**2 + y1**2 + x2**2 + y2**2)/N

	Var=np.abs((1/N)*sum((x1**2 + y1**2 + x2**2 + y2**2)**2)-f_moy**2)
	err = np.sqrt(Var/N)*1.96
	return err


# returns error for function f(x, y) = 1/|x-y|
def monteCarlo_Rec_sing(P1,P2,N):
	
	x1 = P1[0]
	y1 = P1[1]
	
	x2 = P2[0]
	y2 = P2[1]
	f_moy=sum(1/(np.sqrt((x1-x2)**2 + (y1-y2)**2)))/N

	Var=np.abs((1/N)*sum((1/(np.sqrt((x1-x2)**2 + (y1-y2)**2)))**2)-f_moy**2)
	err = np.sqrt(Var/N)*1.96
	return err


############################################################
############################################################

# rectangle 1
A1=np.array([1,1])
B1=np.array([4,1])
C1=np.array([5,3])
D1=np.array([1,3])
print("Rectangle1:  ( A1="+str(A1)+", B1="+str(B1)+", C1="+str(C1)+", D1="+str(D1)+")")

# rectangle 2
A2=np.array([0,0])
B2=np.array([-2,1])
C2=np.array([3,2])
D2=np.array([3,0])
print("Rectangle2:  ( A2="+str(A2)+", B2="+str(B2)+", C2="+str(C2)+", D2="+str(D2)+")")

Point1=monteCarlo_quadrangle(A1,B1,C1,D1,N)
Point2=monteCarlo_quadrangle(A2,B2,C2,D2,N)

plt.figure()
plt.scatter(A1[0], A1[1])
plt.scatter(B1[0], B1[1])
plt.scatter(C1[0], C1[1])
plt.scatter(D1[0], D1[1])
plt.scatter(Point1[0],Point1[1])
plt.scatter(A2[0], A2[1],c='r')
plt.scatter(B2[0], B2[1],c='r')
plt.scatter(C2[0], C2[1],c='r')
plt.scatter(D2[0], D2[1],c='r')
plt.scatter(Point2[0],Point2[1],c='r')

############################################################

err_Reg=np.array([])
err_Sing = np.array([])

for n in range(100,N):
	P1=monteCarlo_quadrangle(A1,B1,C1,D1,n)
	P2=monteCarlo_quadrangle(A2,B2,C2,D2,n)

	# calculate error
	err_Reg = np.append(err_Reg, monteCarlo_Rec_regul(P1, P2, n))
	err_Sing = np.append(err_Sing, monteCarlo_Rec_sing(P1, P2, n))


# plot regular case
plt.figure()
plt.plot(log(range(100,N)),log(err_Reg),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))+4.2, "r")
plt.title("Pair of rectangles regular polynom $x^{2}+y^{2}$|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")

# plot singular case
plt.figure()
plt.plot(log(range(100,N)),log(err_Sing),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))+0.7, "r")
plt.title("Pair of rectangles singular $1/(|x-y|)$ || pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")


