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
    z=r1min*A[2]+(r2max-r1min)*B[2]+(1-r2max)*C[2]
    P=np.array([x,y,z])     
    return P

##formule de Héron pour T1 y cst
#def Aire_Tria_T1(A,B,C):
#    a=np.sqrt((B[0]-C[0])**2+(B[2]-C[2])**2)
#    b=np.sqrt((A[0]-C[0])**2+(A[2]-C[2])**2)
#    c=np.sqrt((B[0]-A[0])**2+(B[2]-A[2])**2)
#    P=a+b+c
#    p=P/2
#    S=np.sqrt(p*(p-a)*(p-b)*(p-c))
#    return S
#
##formule de Héron pour T2 z cst
#def Aire_Tria_T2(A,B,C):
#    a=np.sqrt((B[0]-C[0])**2+(B[1]-C[1])**2)
#    b=np.sqrt((A[0]-C[0])**2+(A[1]-C[1])**2)
#    c=np.sqrt((B[0]-A[0])**2+(B[1]-A[1])**2)
#    P=a+b+c
#    p=P/2
#    S=np.sqrt(p*(p-a)*(p-b)*(p-c))
#    return S
# returns error for function f(x, y) = exp(-|x-y|^2)
def monteCarlo_Tria_regul(P1,P2,N):
    
    x1 = P1[0]
    y1 = P1[1]
    z1 = P1[2]
    
    x2 = P2[0]
    y2 = P2[1]
    z2 = P2[2]
    
    f_moy=sum(np.exp(-((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)))/N
    #    f_int = Aire*f_moy
    #print "f_moy", f_moy
    f_moy2=sum(np.exp(-((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2))**2)
    #err=np.sqrt(np.abs((f_moy2-f_moy**2)/N))
    Var=np.abs(f_moy2-N*f_moy**2)/(N-1)
    err =np.sqrt(Var/N)*1.96/f_moy
    return err


# returns error for function f(x, y) = 1/|x-y|
def monteCarlo_Tria_sing(P1,P2,N):
    
    x1 = P1[0]
    y1 = P1[1]
    z1 = P1[2]
    
    x2 = P2[0]
    y2 = P2[1]
    z2 = P2[2]
    f_moy=sum(1/(np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)))/N

    #print "f_moy", f_moy
    f_moy2=sum(1/((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2))
    #err=np.sqrt(np.abs((f_moy2-f_moy**2)/N))
    Var=np.abs(f_moy2-N*f_moy**2)/(N-1)

    err = np.sqrt(Var/N)*1.96/f_moy
    return err

# ----------------------------------------------------------------
# Calculate and plots
#-----------------------------------------------------------------

# number of points
N=1000

# triangle 1
A1=np.array([2,0,1])
B1=np.array([4,0,1])
C1=np.array([3,0,5])
print("Triangle( A1="+str(A1)+", B1="+str(B1)+", C1="+str(C1)+")")

# triangle 2
A2=np.array([2,0,1])
B2=np.array([4,0,1])
C2=np.array([3,5,1])
print("Triangle( A2="+str(A2)+", B2="+str(B2)+", C2="+str(C2)+")")

# choose points randomly in triangle
Point=monteCarlo_Regul_Tria(A1,B1,C1,N)
Point2=monteCarlo_Regul_Tria(A2,B2,C2,N)
#
#plt.figure()
#plt.scatter(A1[0], A1[2])
#plt.scatter(B1[0], B1[2])
#plt.scatter(C1[0], C1[2])
#plt.scatter(Point[0],Point[2])
#plt.scatter(A2[0], A2[1],c='r')
#plt.scatter(B2[0], B2[1],c='r')
#plt.scatter(C2[0], C2[1],c='r')
#plt.scatter(Point2[0],Point2[1],c='r')
#plt.show()
#print "Aire T1:",Aire_Tria_T1(A1,B1,C1)
#print "Aire T2:",Aire_Tria_T2(A2,B2,C2)
#Aire=Aire_Tria_T1(A1,B1,C1)*Aire_Tria_T2(A2,B2,C2)
#print "Aire:", Aire
err_Reg=np.array([])
err_Sing = np.array([])

for n in range(100,N,10):
    err1_=0
    err2_=0
    for k in range(10):
        # choose points randomly in triangle
        P1=monteCarlo_Regul_Tria(A1,B1,C1,n)    # triangle 1
        P2=monteCarlo_Regul_Tria(A2,B2,C2,n)    # triangle 2
        err1_=err1_+monteCarlo_Tria_regul(P1, P2, n)
        err2_=err2_+monteCarlo_Tria_sing(P1, P2, n)
    err1_=err1_/10
    err2_=err2_/10
    # calculate error
    err_Reg = np.append(err_Reg,err1_)
    err_Sing = np.append(err_Sing,err2_)
print err_Reg[5]
print err_Sing[5]
# plot regular case
plt.figure()
plt.plot(log(range(100,N,10)),log(err_Reg),"b")
plt.plot(log(range(100,N,10)), -0.5*log(range(100,N,10))+1.44, "r")
plt.title("Pair of triangles regular case $exp^{-|x-y|^{2}}$|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")

# plot singular case
plt.figure()
plt.plot(log(range(100,N,10)),log(err_Sing),"b")
plt.plot(log(range(100,N,10)), -0.5*log(range(100,N,10))+0.4, "r")
plt.title("Pair of triangles singular $1/(|x-y|)$ || pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")

plt.show()
