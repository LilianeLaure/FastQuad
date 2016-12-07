# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:39:42 2016

@author: 3100965
"""
import random 
import numpy as np
import math
import matplotlib.pyplot as plt

N=1000
def monteCarlo_Regul_Tria(A,B,C,N):
    r1 = np.random.uniform(0, 1, N)
    r2 = np.random.uniform(0, 1, N)
    x=(1-np.sqrt(r1))*A[0]+(np.sqrt(r1)*(1-r2))*B[0]+(r2*np.sqrt(r1))*C[0]
    y=(1-np.sqrt(r1))*A[1]+(np.sqrt(r1)*(1-r2))*B[1]+(r2*np.sqrt(r1))*C[1]
    P=np.array([x,y])     
    return P

def monteCarlo_Tria(P0,P1,N):
    f_moy=sum(P0**2+P1**2)/N
    print "f_moy", f_moy
    #f_moy2=sum((P0**2+P1**2)**2)/2
    #err=np.sqrt(np.abs((f_moy2-f_moy**2)/N))
    Var=np.abs((1/N)*sum((P0**2+P1**2)**2)-f_moy**2)
    #print "valeur",sum (P0**2+P1**2-f_moy)**2
    err = np.sqrt(Var/N)*1.96
    print "var", Var
    print "f_err", err   
    return err


figure()
A=np.array([1,1])
B=np.array([4,1])
C=np.array([5,3])
Point=monteCarlo_Regul_Tria(A,B,C,N)
print "point:",Point
print "point0",Point[0][0]
print "point1",Point[1][0]
plt.figure
'''
plt.scatter(A[0], A[1])
plt.scatter(B[0], B[1])
plt.scatter(C[0], C[1])
plt.scatter(Point[0],Point[1])
#plt.show()
'''
N=size(Point[0])
print "N",N
monteCarlo_Tria(Point[0],Point[1],N)

err=np.array([])
figure()
for n in range(100,N):
    Point=monteCarlo_Regul_Tria(A,B,C,n)
    err=np.append(err,monteCarlo_Tria(Point[0],Point[1],n))
plt.plot(log(range(100,N)),log(err),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))+3.37, "r")


plt.title("Case 2D triangular regular polynom $x^{2}+y^{2}$|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
plt.show()
