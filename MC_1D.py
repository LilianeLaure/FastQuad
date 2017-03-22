# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:06:20 2017

@author: 3201955
"""

import random
import numpy as np
import math
import matplotlib.pyplot as plt

N = 1000
a = 0
b = 1
'''
# Case 1 : regular function constant f(x) = 4
def monteCarlo_Regul_Const(a, b, N):
    randValues = np.random.uniform(a, b, N)
    f_moy = (4*N)/N
    f_int = (b-a)*f_moy
    f_moy2 = (4**2*N)/N
    
    error = np.sqrt((f_moy2-f_moy**2)/N) / f_int
    return error
   '''
# Case 1 : regular function polynom f(x) = x^2
def monteCarlo_Regul_poly(a, b, N):
    X = np.random.uniform(a, b, N)
    f_moy = sum(X**2)/N
    f_int = (b-a)*f_moy
    #f_moy2 = sum((randValues**2)**2)/N
    f_moy2 = sum((X**2)**2)
    
    #error = (b-a)*np.sqrt((f_moy2 - f_moy**2)/N) / f_int
    error = (b-a)*np.sqrt(((f_moy2 - N*f_moy**2)/(N-1))/N)*1.96 / f_int
    return error
    
#case 2: regular function exp(x)
def monteCarlo_Regul_exp(a, b, N):
    X = np.random.uniform(a, b, N)
    f_moy = sum(np.exp(X))/N
    f_int = (b-a)*f_moy
    #f_moy2 = sum((randValues**2)**2)/N
    f_moy2 = sum((np.exp(X))**2)
    error = (b-a)*np.sqrt(((f_moy2 - N*f_moy**2)/(N-1))/N)*1.96 / f_int
    return error

# Case 3 : singular function f(x)  =|x|
def monteCarlo_Sing(a, b, N):
    randValues = np.random.uniform(a, b, N)
    f_moy = sum(np.abs(randValues))/N
    f_int = (b-a)*f_moy
    f_moy2 = sum(np.abs(randValues)**2)
    #f_moy2 = sum((np.log(np.abs(randValues)))**2)/(N-1)
    
    error = (b-a)*np.sqrt(((f_moy2 - N*f_moy**2)/N-1)/N)*1.96 / f_int
    #error = (b-a)*np.sqrt((f_moy2 - N*f_moy**2/(N-1))/N) * 1.96 / f_int
    return error
#--------------------------------------------------
'''
print("Case regular constant")
for n in range(100,N):
    err = monteCarlo_Regul_Const(a, b, n)
'''
figure()
errVec1 = np.array([])
print("Case regular exp")
for n in range(100,1000,10):
    err_=0
    for k in range(10):
        err_ = err_+monteCarlo_Regul_exp(a, b, n)
    err_=err_/10
    errVec1 = np.append(errVec1, err_)

plt.plot(np.log(range(100,1000,10)), np.log(errVec1))
plt.plot(np.log(range(100,1000,10)), -0.5*np.log(range(100,1000,10))-0.55, "r")
plt.title("Case 1D regular $exp^{x}$ || pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")

figure()
errVec2 = np.array([])
print("Case regular polynom")
for n in range(100,1000,10):
    err_=0
    for k in range(10):
        err_ = err_+monteCarlo_Regul_poly(a, b, n)
    err_=err_/10
    errVec2 = np.append(errVec2, err_)

plt.plot(np.log(range(100,1000,10)), np.log(errVec2))
plt.plot(np.log(range(100,1000,10)), -0.5*np.log(range(100,1000,10))+0.55, "r")
plt.title("Case 1D regular polynom $x^{2}$ || pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")

figure()
errVec3 = np.array([])
print("Case regular singular")
for n in range(100,1000,10):
    err_=0
    for k in range(10):
        err_ = err_+monteCarlo_Regul_poly(a, b, n)
    err_=err_/10
    errVec3 = np.append(errVec3, err_)

plt.plot(np.log(range(100,1000,10)), np.log(errVec3))
plt.plot(np.log(range(100,1000,10)), -0.5*np.log(range(100,1000,10))+0.55, "r")
plt.title("Case 1D singular polynom $|x|$ || pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
