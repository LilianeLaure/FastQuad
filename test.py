# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:49:48 2017

@author: 3503833
"""

a1 = np.array([-1.0, 0.0])
b1 = np.array([-1.0, -1.0])

a2 = np.array([0, -1.0])
b2 = np.array([-1.0, -1.0])
print "[a1, b1] = [", a1 ,",",b1, "]", "!!!! [a2, b2] = [", a2 ,",",b2, "]"

Nb_Int = 2

I = 0
for i in range(0, Nb_Int):
	print "i =", i
	for j in range(0, Nb_Int):

		# TO DO: FIND RIGHT INTERVALLS
		if i == 0:
			newInt1 = [np.array(b1), np.array(b1 + (a1-b1)/(2**(Nb_Int-i-1)))]
		elif i == Nb_Int - 1:
			newInt1 = [np.array(b1 + (a1-b1)/2), np.array(a1)]
		else:
			newInt1 = [np.array(b1 + (a1-b1)/(2**(Nb_Int-i))), np.array(b1 + (a1-b1)/(2**(Nb_Int-i-1)))]
			
		if j == 0:
			newInt2 = [np.array(b2), np.array(b2 + (a2-b2)/(2**(Nb_Int-j-1)))]
		elif j == Nb_Int - 1:
			newInt2 = [np.array(b2 + (a2-b2)/2), np.array(a2)]
		else:
			newInt2 = [np.array(b2 + (a2-b2)/(2**(Nb_Int-j))), np.array(b2 + (a2-b2)/(2**(Nb_Int-j-1)))]


		
		if ( (i == 0) and (j == 0) ):
			print "not calcul: ", " j =", j, newInt1, "x", newInt2
		else:
			print "calcul: ", " j =", j, newInt1, "x", newInt2

