#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:23:31 2016

@author: 3417212
"""

import numpy as np
import matplotlib.pyplot as plt

"""
on se place sur [-1;1]x[-1;1]
"""


def f(x,y):
    return x[1]**2 + x[0]**2 + y[1]**2 + y[0]**2
    

def h(x,y):   
    #print(np.array_equal(x,y))
    #print("norm x : ", np.linalg.norm(x), "norm y : ", np.linalg.norm(y))
    return np.log(sqrt(x[1]**2 + y[1]**2 + (x[0]**2 + y[0]**2) ) )

def segment(a,b,t):
    s1=a[0]+t*(b[0]-a[0])
    s2=a[1]+t*(b[1]-a[1])
    return np.array([s1,s2])

# calculate length of segment
def length_Segment(a,b):
    return np.sqrt( (a[1]-b[1])**2 + (a[0]-b[0])**2 )
    
       


def calculateInt(int1, int2, N):
	
	# intervalls
	a1, b1 = int1
	a2, b2 = int2	
	
	t,w = np.polynomial.legendre.leggauss(N) # t in [-1,1], w weight, ordre 50
	
	# calculate parametrisation of vector t
	gamma = segment(a1, b1, t)
	sigma = segment(a2, b2, t)

	# calculate integral
	I = 0
	for i, gi in enumerate(gamma):
		for j, sj in enumerate(sigma):
			I = I + w[i]*w[j]*f(gi, sj)
	
	I = length_Segment(a1, b1)*length_Segment(a2, b2)*I
	print(I)
	return I
	
	
def calculateError(int1, int2, N):
	
	# intervalls
	a1, b1 = int1
	a2, b2 = int2

	I = calculateInt(int1, int2, N)
	
	# subdivise both segments to calculate error
	
	# segment [a1, b1] becomes [a1, (a1+b1)/2] and [(a1+b1)/2, b1]
	int11 = [int1[0], (int1[0]+int1[1])/2]
	int12 = [(int1[0]+int1[1])/2, int1[1]]
	
	# segment [a2, b2] becomes [a2, (a2+b2)/2] and [(a2+b2)/2, b2]
	int21 = [int2[0], (int2[0]+int2[1])/2]
	int22 = [(int2[0]+int2[1])/2, int2[1]]
	
	# calculate sum of subIntervalls
	I_sub = (calculateInt(int11, int21, N) + calculateInt(int11, int22, N) 
		    + calculateInt(int12, int21, N) + calculateInt(int12, int22, N))
						
	err = abs(I - I_sub)
	
	return err
	
	
def calculateVectorError(int1, int2, N, ordre):
	
	# composite Gauss - Legendre
	pass

###############################################################################
###############################################################################

epsilon = 0.5

a1 = np.array([-epsilon, -1])
b1 = np.array([-epsilon, +1])

a2 = np.array([epsilon, -1])
b2 = np.array([epsilon, +1])

intervalle1=[a1, b1]
intervalle2=[a2, b2]

ordre = 50 # ordre for the method gaussLegendre

N = 1000 # number of points

rangeN = range(10, 99)

errSing = np.array([])
for n in rangeN:
	errSing = np.append( errSing, calculateError(intervalle1, intervalle2, n))
	
plot(np.log(rangeN), np.log(errSing) )
plt.title("regular function, pente = -4")

