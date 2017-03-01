# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:27:20 2016

@author: 3100965
"""
#f(x,y)=-ln(|x-y|)
import random 
import numpy as np
import math
import matplotlib.pyplot as plt

N = 1000
eps = 0.5

a1 = np.array([-eps , -1])
b1 = np.array([-eps, 1])
a2 = np.array([eps,- 1])
b2 = np.array([eps, 1])

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

# singular: -ln(|x-y|)
def monteCarlo_err(Point1, Point2):
    x1, x2 = Point1
    y1, y2 = Point2
    N=len(x1)
    
    f_moy = sum(-np.log(np.sqrt((x1 - y1)**2 + (x2 - y2)**2)))/N
    Var = np.abs((1/N)*sum(-np.log(np.sqrt((x1-y1)**2 + (x2-y2)**2)**2))-f_moy**2)
    err = np.sqrt(Var/N)*1.96

    return err

#--------------------------------------------------

#dist=mesure_x_y(a1,b1,a2,b2)
err=np.array([])
figure()
for n in range(100,N):
    Point1 = monteCarlo2Seg(a1, b1, n) # points of segment 1
    Point2 = monteCarlo2Seg(a2, b2, n) # points of segment 2

    err=np.append(err,monteCarlo_err(Point1,Point2))

plt.plot(np.log(range(100,N)),np.log(err),"b")
plt.plot(np.log(range(100,N)), -0.5*np.log(range(100,N))-0.87, "r")
plt.title("Case 2 Segments Singular polynom $-ln(|x-y|)$|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")