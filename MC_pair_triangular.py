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
    r1min=np.minimum(r1,r2)
    r2max=np.maximum(r1,r2)
    x=r1min*A[0]+(r2max-r1min)*B[0]+(1-r2max)*C[0]
    y=r1min*A[1]+(r2max-r1min)*B[1]+(1-r2max)*C[1]
    P=np.array([x,y])     
    return P

#formule de Héron
def Aire_Tria(A,B,C):
    a=np.sqrt((B[0]-C[0])**2+(B[1]-C[1])**2)
    b=np.sqrt((A[0]-C[0])**2+(A[1]-C[1])**2)
    c=np.sqrt((B[0]-A[0])**2+(B[1]-A[1])**2)
    P=a+b+c
    p=P/2
    S=np.sqrt(p*(p-a)*(p-b)*(p-c))
    return S
    
    
# returns error for function f(x, y) = exp(-|x-y|^2)
def monteCarlo_Tria_regul(Aire,P1,P2,N):
    
    x1 = P1[0]
    y1 = P1[1]
        
    x2 = P2[0]
    y2 = P2[1]
    
    f_moy=sum(np.exp(-((x1-x2)**2 + (y1-y2)**2)))/N
    f_int=Aire*f_moy
    #print "f_moy", f_moy
    f_moy2=sum(np.exp(-((x1-x2)**2 + (y1-y2)**2))**2)
    #err=np.sqrt(np.abs((f_moy2-f_moy**2)/N))
    Var=np.abs(f_moy2-N*f_moy**2)/(N-1)


    err = Aire*np.sqrt(Var/N)*1.96/f_int
    return err


# returns error for function f(x, y) = 1/|x-y|
def monteCarlo_Tria_sing(Aire,P1,P2,N):
    
    x1 = P1[0]
    y1 = P1[1]
    
    x2 = P2[0]
    y2 = P2[1]
    f_moy=sum(1/(np.sqrt((x1-x2)**2 + (y1-y2)**2)))/N
    f_int=Aire*f_moy
    #print "f_moy", f_moy
    f_moy2=sum(1/((x1-x2)**2 + (y1-y2)**2))
    #err=np.sqrt(np.abs((f_moy2-f_moy**2)/N))
    Var=np.abs(f_moy2-N*f_moy**2)/(N-1)

    err = Aire*np.sqrt(Var/N)*1.96/f_int
    return err

# ----------------------------------------------------------------
# Calculate and plots
#-----------------------------------------------------------------

# number of points
N=1000

# triangle 1
A1=np.array([0,1])
B1=np.array([1,1])
C1=np.array([0,2])
print("Triangle( A1="+str(A1)+", B1="+str(B1)+", C1="+str(C1)+")")

# triangle 2
A2=np.array([0,0.5])
B2=np.array([1.5,0.5])
C2=np.array([0.4,1.1])
print("Triangle( A2="+str(A2)+", B2="+str(B2)+", C2="+str(C2)+")")

# choose points randomly in triangle
Point=monteCarlo_Regul_Tria(A1,B1,C1,N)
Point2=monteCarlo_Regul_Tria(A2,B2,C2,N)

plt.figure()
plt.scatter(A1[0], A1[1])
plt.scatter(B1[0], B1[1])
plt.scatter(C1[0], C1[1])
plt.scatter(Point[0],Point[1])
plt.scatter(A2[0], A2[1],c='r')
plt.scatter(B2[0], B2[1],c='r')
plt.scatter(C2[0], C2[1],c='r')
plt.scatter(Point2[0],Point2[1],c='r')
#plt.show()

Aire=Aire_Tria(A1,B1,C1)*Aire_Tria(A2,B2,C2)

err_Reg=np.array([])
err_Sing = np.array([])

for n in range(100,N):
    # choose points randomly in triangle
    P1=monteCarlo_Regul_Tria(A1,B1,C1,n)    # triangle 1
    P2=monteCarlo_Regul_Tria(A2,B2,C2,n)    # triangle 2

    # calculate error
    err_Reg = np.append(err_Reg,monteCarlo_Tria_regul(Aire,P1, P2, n))
    err_Sing = np.append(err_Sing,monteCarlo_Tria_sing(Aire,P1, P2, n))
print err_Reg[100]
print err_Sing[100]
# plot regular case
plt.figure()
plt.plot(log(range(100,N)),log(err_Reg),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))-0.24, "r")
plt.title("Pair of triangles regular polynom $exp^{-|x-y|^{2}}$|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")

# plot singular case
plt.figure()
plt.plot(log(range(100,N)),log(err_Sing),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))+0.2, "r")
plt.title("Pair of triangles singular $1/(|x-y|)$ || pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")

plt.show()
