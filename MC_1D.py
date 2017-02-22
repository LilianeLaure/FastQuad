# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:06:20 2017

@author: thai
"""

import random
import numpy as np
import math
import matplotlib.pyplot as plt

N = 1000
a = 0
b = 1

# Case 1 : regular function constant f(x) = 4
def monteCarlo_Regul_Const(a, b, N):
    randValues = np.random.uniform(a, b, N)
    f_moy = (4*N)/N
    f_int = (b-a)*f_moy
    f_moy2 = (4**2*N)/N
    
    error = np.sqrt((f_moy2-f_moy**2)/N) / f_int
    return error
   
# Case 2 : regular function polynom f(x) = x^2
def monteCarlo_Regul_poly(a, b, N):
    randValues = np.random.uniform(a, b, N-1)
    f_moy = sum(randValues**2)/N
    f_int = (b-a)*f_moy
    #f_moy2 = sum((randValues**2)**2)/N
    f_moy2 = sum((randValues**2)**2)/(N-1)
    
    #error = (b-a)*np.sqrt((f_moy2 - f_moy**2)/N) / f_int
    error = (b-a)*np.sqrt((f_moy2 - N*f_moy**2/(N-1))/N)*1.96 / f_int
    return error
    
# Case 3 : singular function f(x)  =ln(|x|)
def monteCarlo_Sing(a, b, N):
    randValues = np.random.uniform(a, b, N)
    f_moy = sum(np.log(np.abs(randValues)))/N
    f_int = (b-a)*f_moy
    f_moy2 = sum((np.log(np.abs(randValues)))**2)/N
    #f_moy2 = sum((np.log(np.abs(randValues)))**2)/(N-1)
    
    error = (b-a)*np.sqrt((f_moy2 - f_moy**2)/N) / f_int
    #error = (b-a)*np.sqrt((f_moy2 - N*f_moy**2/(N-1))/N) * 1.96 / f_int
    return error
#--------------------------------------------------
'''
print("Case regular constant")
for n in range(100,N):
    err = monteCarlo_Regul_Const(a, b, n)
'''

errVec2 = np.array([])
print("Case regular polynom")
for n in range(100,N):
    err2 = monteCarlo_Regul_poly(a, b, n)
    errVec2 = np.append(errVec2, err2)

plt.plot(np.log(range(100,N)), np.log(errVec2))
plt.plot(np.log(range(100,N)), -0.5*np.log(range(100,N))+0.55, "r")
plt.title("Case 1D regular polynom $x^{2}$ || pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")

figure()
errVec3 = np.array([])
print("Case regular singular")
for n in range(100,N):
    err3 = monteCarlo_Regul_poly(a, b, n)
    errVec3 = np.append(errVec3, err3)

plt.plot(np.log(range(100,N)), np.log(errVec3))
plt.plot(np.log(range(100,N)), -0.5*np.log(range(100,N))+0.55, "r")
plt.title("Case 1D singular polynom $ln(|x|)$ || pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")