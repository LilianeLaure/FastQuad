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
    
A=np.array([1,1])
B=np.array([4,1])
C=np.array([4,3])
D=np.array([1,3])
Point=monteCarlo_quadrangle(A,B,C,D,N)

plt.figure()
plt.scatter(A[0], A[1])
plt.scatter(B[0], B[1])
plt.scatter(C[0], C[1])
plt.scatter(D[0],D[0])
plt.scatter(Point[0],Point[1])
plt.show()
#case regular f(x,y) = exp(x²+y²)
def monteCarlo_Reg(x,y,n):
    f_moy=sum(np.exp(x**2+y**2))/n
    f_moy2=sum(np.exp(x**2+y**2)**2)
    Var=np.abs(f_moy2-n*f_moy**2)/(n-1)
    err = np.sqrt(Var/n)*1.96/f_moy
    return err

#1/|x-y|
def monteCarlo_Sing(x,y,n):
    f_moy=sum(1/(x-y))/n
    f_moy2=sum(1/(x-y)**2)
    Var=np.abs(f_moy2-n*f_moy**2)/(n-1)
    err = np.sqrt(Var/n)*1.96/f_moy
    return err

#Point = monteCarlo_quadrangle(A,B,C,D,N)
##########Cas régulier
err=np.array([])
err2=np.array([])
figure()
for n in range(100,N):
    Point=monteCarlo_quadrangle(A,B,C,D,n)
    err=np.append(err,monteCarlo_Reg(Point[0],Point[1],n))
    err2=np.append(err,monteCarlo_Sing(Point[0],Point[1],n))

plt.plot(log(range(100,N)),log(err),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))+3.97, "r")
plt.title("Case 2D quadrangle regular $exp(x^{2}+y^{2})$")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
plt.show()


figure()


plt.plot(log(range(100,N)),log(err2),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))+3.97, "r")
plt.title("Case 2D quadrangle singular$")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
plt.show()

