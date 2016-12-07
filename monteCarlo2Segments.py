# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:27:20 2016

@author: 3503833
"""

import random 
import numpy as np
import math
import matplotlib.pyplot as plt

N = 10
eps = 0.5

a1 = np.array([-eps , -1])
b1 = np.array([-eps, 1])
a2 = np.array([eps,- 1])
b2 = np.array([eps, 1])

def monteCarlo2Seg(a1,b1,a2,b2,N):
    t=np.random.uniform(0,1,N)
    s=np.random.uniform(0,1,N)
    x=np.array([])
    y=np.array([])
    for i in range(len(t)):
        x=np.append(x,(1-t[i])*a1+t[i]*b1)
        y=np.append(y,(1-s[i])*a2+s[i]*b2)
    Point=np.array([x,y])
    return Point

def mesure_x_y(a1,b1,a2,b2):
    dist_x=np.sqrt((a1[0]-b1[0])**2+(a1[1]-b1[1])**2)
    dist_y=np.sqrt((a2[0]-b2[0])**2+(a2[1]-b2[1])**2)
    Point=np.array([dist_x,dist_y])
    return Point
'''
def monteCarlo_err(x,y):
    N=
'''
Point=monteCarlo2Seg(a1,b1,a2,b2,2)
print "POINT :",Point
dist=mesure_x_y(a1,b1,a2,b2)
print "mesure: ",dist
print len(Point[0])