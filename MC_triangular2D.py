# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:39:42 2016

@author: 3100965
"""
import random 
import numpy as np
from numpy.linalg import norm
import math
import matplotlib.pyplot as plt

N=1000
def Tria(A,B,C,N):
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
    
#exp(-|x-y|²)
def monteCarlo_Tria_Reg(Aire,P0,P1,N):
    f_moy=sum(np.exp(-np.abs(P0-P1)**2))/N
    f_int=Aire*f_moy
    #print "f_moy", f_moy
    f_moy2=sum(np.exp(-np.abs(P0-P1)**2)**2)
    #err=np.sqrt(np.abs((f_moy2-f_moy**2)/N))
    Var=np.abs(f_moy2-N*f_moy**2)/(N-1)
    #print "valeur",sum (P0**2+P1**2-f_moy)**2
    err = Aire*np.sqrt(Var/N)*1.96/f_int
  #  print "var", Var
 #   print "f_err", err   
    return err


#1/|x-y|
def monteCarlo_Tria_Sing(Aire,P0,P1,N):
    f_moy=sum(1/np.abs(P0-P1))/N
    #print "f_moy", f_moy
    f_int=Aire*f_moy
    f_moy2=sum(1/np.abs(P0-P1)**2)
    #err=np.sqrt(np.abs((f_moy2-f_moy**2)/N))
    Var=(f_moy2-N*f_moy**2)/(N-1)
    #print "valeur",sum (P0**2+P1**2-f_moy)**2
    err = Aire*np.sqrt(Var/N)*1.96/f_int
    #print "var", Var
   # print "f_err", err   
    return err


A=np.array([0,1])
B=np.array([0.9,1])
C=np.array([0,2])
Point=Tria(A,B,C,N)
#print "point:",Point
#print "point0",Point[0][0]
#print "point1",Point[1][0]

Aire=Aire_Tria(A,B,C)

plt.figure
plt.scatter(A[0], A[1])
plt.scatter(B[0], B[1])
plt.scatter(C[0], C[1])
plt.scatter(Point[0],Point[1])
plt.show()


N=size(Point[0])

Point=np.array([])
err=np.array([])
figure()
for n in range(100,1000,10):
    err_=0
    for k in range(10):
        Point=Tria(A,B,C,n)
        err_=err_+monteCarlo_Tria_Reg(Aire,Point[0],Point[1],n)
    err_=err_/k
    err=np.append(err,err_)
    #print err
plt.plot(log(range(100,1000,10)),log(err),"b")
plt.plot(log(range(100,1000,10)), -0.5*log(range(100,1000,10))+0.3, "r")
print "reg",err[4]

plt.title("Case 2D triangular regular $e^{-|x-y|^2}$|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
plt.show()

#------------------------case singular
err2=np.array([])
figure()
for n in range(100,1000,10):
    err_=0
    for k in range(10):
        Point=Tria(A,B,C,n)
        err_=err_+monteCarlo_Tria_Sing(Aire,Point[0],Point[1],n)
    err_=err_/k
    err2=np.append(err2,err_)
plt.plot(log(range(100,1000,10)),log(err2),"b")
plt.plot(log(range(100,1000,10)), -0.5*log(range(100,1000,10))+0.25, "r")
print "sing", err2[5]

plt.title("Case 2D triangular singular $1/|x-y|$|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
plt.show()