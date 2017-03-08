# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:27:20 2016

@author: 3100965
"""
import random 
import numpy as np
import math
import matplotlib.pyplot as plt

#on choisi aléatoirement des points sur chaque segment
def monteCarlo2Seg(a, b, N):
    t=np.random.uniform(0,1,N)
    
    # parametrisation
    x1 = (1-t)*a[0]+t*b[0]
    x2 = (1-t)*a[1]+t*b[1]        

    point = np.array([x1, x2])                    
   
    return point

#on mesure la taille des segments
'''
def mesure_x_y(a1,b1,a2,b2):
    dist_x=np.sqrt((a1[0]-b1[0])**2+(a1[1]-b1[1])**2)
    dist_y=np.sqrt((a2[0]-b2[0])**2+(a2[1]-b2[1])**2)
    Point=np.array([dist_x,dist_y])
    return Point
'''

#regular : e^(-|x-y|²)
def monteCarlo_err_reg(Point1, Point2):
    x1, x2 = Point1
    y1, y2 = Point2
    N = len(x1)
    
    f_moy = sum(np.exp(x1**2+x2**2+y1**2+y2**2))/N
    #f_int=f_moy
    f2=sum(np.exp(x1**2+x2**2+y1**2+y2**2)**2)
    Var = np.abs(f2 - N*f_moy**2)/(N-1)
    err = np.sqrt(Var/N)*1.96/f_moy
    
    return err
    
# singular : -ln(|x-y|)
def monteCarlo_err_sing(Point1, Point2):
    x1, y1 = Point1
    x2, y2 = Point2
    N=len(x1)
    
    f_moy = sum(-np.log(np.sqrt((x1 - x2)**2 + (y1 - y2)**2)))/N
    f2=sum(-np.log(np.sqrt((x1-x2)**2 + (y1-y2)**2))**2)
    Var = np.abs((f2-N*f_moy**2)/(N-1))
    err = np.abs(np.sqrt(Var/N)*1.96/f_moy)

    return err

'''
# singular: -ln(|x-y|)
def monteCarlo_err(Point1, Point2):
    x1, x2 = Point1
    y1, y2 = Point2
    N=len(x1)

    f_moy = sum(-np.log(np.sqrt((x1 - y1)**2 + (x2 - y2)**2)))/N

    Var=np.abs((2/N)*sum(sum(-np.log(np.sqrt((x1 - y1)**2 + (x2 - y2)**2)))**2)-f_moy**2)
    err = np.sqrt(Var*2/N)*1.96
     
    return err 
'''
# ----------------------------------------------------------------------------

N = 1000
#eps = 0.5
listEps = np.array([0,0.5, 1, 1.5, 2])

print("===== Case singular =====")
figure()
for eps in listEps:

    a1 = np.array([-eps , -1])
    b1 = np.array([-eps, 1])
    a2 = np.array([eps,- 1])
    b2 = np.array([eps, 1])
    
    err=np.array([])
    for n in range(100,N):
        Point1 = monteCarlo2Seg(a1, b1, n) # points of segment 1
        Point2 = monteCarlo2Seg(a2, b2, n) # points of segment 2
        err=np.append(err,monteCarlo_err_sing(Point1,Point2))
    
    print "eps :",eps
    print "err",err[100]
    plt.plot(log(range(100,N)),log(err), label = str(eps))
    plt.plot(log(range(100,N)), -0.5*log(range(100,N))+1.55, "k")


plt.legend(loc = 2)
plt.title("Case 1 Segments Singular polynom $-ln(|x-y|)$ more eps|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")

print("===== Case regular =====")
figure()
for eps in listEps:
    err1=np.array([])
    for n in range(100,N):
        Point1 = monteCarlo2Seg(a1, b1, n) # points of segment 1
        Point2 = monteCarlo2Seg(a2, b2, n) # points of segment 2
        err1=np.append(err1,monteCarlo_err_reg(Point1,Point2))
    
    print "eps :",eps
    print "err",err1[100]
    plt.plot(log(range(100,N)),log(err1), label = str(eps))
    #plt.plot(log(range(100,N)), -0.5*log(range(100,N))-0.55, "r")


plt.legend(loc = 2)
plt.title("Case 2 Segments Regular polynom $exp(-|x-y|^2)$ more eps|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
