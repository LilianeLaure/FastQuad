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
eps = 0.25

a1 = np.array([-eps , -1])
b1 = np.array([-eps, 1])
a2 = np.array([eps,- 1])
b2 = np.array([eps, 1])

#on choisi aléatoirement des points sur chaque segment
def monteCarlo2Seg(a1,b1,a2,b2,N):
    t=np.random.uniform(0,1,N)
    s=np.random.uniform(0,1,N)
    x=np.array([])
    y=np.array([])
    # parametrisation on the segment
    for i in range(len(t)):
        x=np.append(x,(1-t[i])*a1+t[i]*b1)
        y=np.append(y,(1-s[i])*a2+s[i]*b2)
								
   
    Point=np.array([x,y])
    print "Point: ", Point
    return Point

#on mesure la taille des segments
#def mesure_x_y(a1,b1,a2,b2):
#    dist_x=np.sqrt((a1[0]-b1[0])**2+(a1[1]-b1[1])**2)
#    dist_y=np.sqrt((a2[0]-b2[0])**2+(a2[1]-b2[1])**2)
#    Point=np.array([dist_x,dist_y])
#    return Point

# singular: -ln(|x-y|)
def monteCarlo_err(x,y):

    x1, x2 = Point1
    y1, y2 = Point2
    N=len(x)
    Vect=np.array([])
    for i in range(N/2):
        k=i*2
        #Vect=np.append(Vect,-np.log(np.sqrt((x[k]-y[k])**2+(x[k+1]-y[k+1])**2))*dist_x*dist_y)
        Vect=np.append(Vect,-np.log(np.sqrt((x[k]-y[k])**2+(x[k+1]-y[k+1])**2)))
        #f = -np.log(np.matrix.norm(x-y))

    f_moy=sum(Vect)/(N/2)
    #print "f_moy: ",f_moy
    Var=np.abs((2/N)*sum(Vect**2)-f_moy**2)
    err = np.sqrt(Var*2/N)*1.96
    #print "var", Var
    #print "f_err", err   
    return err 


#dist=mesure_x_y(a1,b1,a2,b2)
err=np.array([])
figure()
for n in range(100,N):
	Point1 = monteCarlo2Seg(a1, b1, n)
	Point2 = monteCarlo2Seg(a2, b2, n)
    Point=monteCarlo2Seg(a1,b1,a2,b2,n)    # points of first segment
    Point1=Point[0]
    Point2=Point[1] # points of second segment 
    #err=np.append(err,monteCarlo_err(Point[0],Point[1],dist[0],dist[1]))
    err=np.append(err,monteCarlo_err(Point1,Point2))

plt.plot(log(range(100,N)),log(err),"b")
plt.plot(log(range(100,N)), -0.5*log(range(100,N))+0.5, "r")


plt.title("Case 2 Segments Singular polynom $-ln(|x-y|)$|| pente = -0.5")
plt.xlabel("log(N)")
plt.ylabel("log(err)")
plt.show()