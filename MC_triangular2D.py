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
    
#exp(x+y)
def monteCarlo_Tria_Reg(Aire,P0,P1,N):
    f_moy=sum(np.exp(P0+P1))/N
    f_int=Aire*f_moy
    #print "f_moy", f_moy
    f_moy2=sum(np.exp(P0+P1)**2)
    #err=np.sqrt(np.abs((f_moy2-f_moy**2)/N))
    Var=np.abs(f_moy2-N*f_moy**2)/(N-1)
    #print "valeur",sum (P0**2+P1**2-f_moy)**2
    err = Aire*np.sqrt(Var/N)*1.96/f_int
  #  print "var", Var
 #   print "f_err", err   
    return err


#1/|x-y|
def monteCarlo_Tria_Sing(Aire,P0,P1,N):
    f_moy=sum(1/np.sqrt(P0**2+P1**2))/N
    #print "f_moy", f_moy
    f_int=Aire*f_moy
    f_moy2=sum(1/(P0**2+P1**2))
    #err=np.sqrt(np.abs((f_moy2-f_moy**2)/N))
    Var=np.abs((f_moy2-N*f_moy**2)/(N-1))
    #print "valeur",sum (P0**2+P1**2-f_moy)**2
    err = Aire*np.sqrt(Var/N)*1.96/f_int
    #print "var", Var
   # print "f_err", err   
    return err


A=np.array([1,1])
B=np.array([4,1])
C=np.array([5,3])
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


err=np.array([])
figure()
for n in range(100,N):
    Point=Tria(A,B,C,n)
    err=np.append(err,monteCarlo_Tria_Reg(Aire,Point[0],Point[1],n))
    #print err
plt.plot(log(range(100,N)),log(err),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))+3.37, "r")
print "reg",err[500]

plt.title("Case 2D triangular regular polynom $x^{2}+y^{2}$|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
plt.show()

#------------------------case singular
err2=np.array([])
figure()
for n in range(100,N):
    Point=Tria(A,B,C,n)
    err2=np.append(err2,monteCarlo_Tria_Sing(Aire,Point[0],Point[1],n))
plt.plot(log(range(100,N)),log(err2),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))+-0.25, "r")
print "sing", err2[500]

plt.title("Case 2D triangular singular polynom $1/x^{2}-y^{2}$|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
plt.show()