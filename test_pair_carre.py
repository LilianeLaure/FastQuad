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
def monteCarlo_Rec_regul(P1,P2,area1,area2,N):
	
	x1 = P1[0]
	y1 = P1[1]
	
	x2 = P2[0]
	y2 = P2[1]

	f_moy=area1*area2*sum(x1**2 + y1**2 + x2**2 + y2**2)/N

	Var=np.abs((1/N)*sum((x1**2 + y1**2 + x2**2 + y2**2)**2)-f_moy**2)
	err = np.sqrt(Var/N)*1.96
	return err


# returns error for function f(x, y) = 1/|x-y|
def monteCarlo_Rec_sing(P1,P2,area1,area2,N):
	
	x1 = P1[0]
	y1 = P1[1]
	
	x2 = P2[0]
	y2 = P2[1]
	f_moy=area1*area2*sum(1/(np.sqrt((x1-x2)**2 + (y1-y2)**2)))/N

	Var=np.abs((1/N)*sum((1/(np.sqrt((x1-x2)**2 + (y1-y2)**2)))**2)-f_moy**2)
	err = np.sqrt(Var/N)*1.96
	return err



# calculate area in the triangle (Shoelace_formula)
def calculateTriangleArea(A, B, C):
	x1, y1 = A
	x2, y2 = B
	x3, y3 = C
	return 0.5*(x1*y2 + x2*y3 + x3*y1 - x2*y1 - x3*y2 - x1*y3)
	
# calculate area in rectangle
def calculateRectangleArea(A, B, C, D):
	area1 = calculateTriangleArea(A, B, C)
	area2 = calculateTriangleArea(A, C, D)
	return abs(area1) + abs(area2)


############################################################
############################################################


# rectangle 1
A1=np.array([1,1])
B1=np.array([4,1])
C1=np.array([5,3])
D1=np.array([1,3])
print("Rectangle1:  ( A1="+str(A1)+", B1="+str(B1)+", C1="+str(C1)+", D1="+str(D1)+")")



A2=np.array([-1,-4])
B2=np.array([-2,1])
C2=np.array([3,3])
D2=np.array([1,0])
print("Rectangle2:  ( A2="+str(A2)+", B2="+str(B2)+", C2="+str(C2)+", D2="+str(D2)+")")
#Point = monteCarlo_quadrangle(A,B,C,D,N)

area1 = calculateRectangleArea(A1, B1, C1, D1)
area2 = calculateRectangleArea(A2, B2, C2, D2)
print "area1", area1, ", area2", area2
#monteCarlo_Tria(Point[0],Point[1],N)

err_Reg=np.array([])
err_Sing = np.array([])


for n in range(100,N):
	P1=monteCarlo_quadrangle(A1,B1,C1,D1,n)
	P2=monteCarlo_quadrangle(A2,B2,C2,D2,n)

	# calculate error
	err_Reg = np.append(err_Reg, monteCarlo_Rec_regul(P1, P2, area1, area2, n))
	err_Sing = np.append(err_Sing, monteCarlo_Rec_sing(P1, P2, area1, area2, n))


# plot regular case
plt.figure()
plt.plot(log(range(100,N)),log(err_Reg),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))+8.81, "r")
plt.title("Pair of rectangles regular polynom $x^{2}+y^{2}$|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")

# plot singular case
plt.figure()
plt.plot(log(range(100,N)),log(err_Sing),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))+5.15, "r")
plt.title("Pair of rectangles singular $1/(|x-y|)$ || pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")


"""
figure()
plt.scatter(A[0], A[1])
plt.scatter(B[0], B[1])
plt.scatter(C[0], C[1])
plt.scatter(D[0], D[1])
plt.scatter(Point[0],Point[1])"""

plt.show()

