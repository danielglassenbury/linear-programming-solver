# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 23:41:51 2023

@author: daniel
"""

import numpy as np

"""
Todo:
    create A, b, c, and B from adding variables and constraints
"""


"""
Matrices representing the governing linear equations---------------------------
"""
A = np.array([[2, 1, 1, 0, 0], 
              [1, 1, 0, 1, 0],
              [1, 0, 0, 0, 1]])

b = np.array([100, 80, 40]).reshape((3, 1))

c = np.array([[3, 2, 0, 0, 0]])


"""
Initial feasible solution
"""
numBasis, numVars = A.shape

# set the initial feasible solution B to all the slack variables.
# slack variables must be stored in the last columns of A
allVars = range(numVars)
basisVars = range(numVars-numBasis, numVars)

B = A[:, basisVars]

BInv = np.linalg.inv((B))

xb = np.matmul(BInv, b)

cb = c[:, basisVars]

zb = np.dot(cb, xb).item()

# find the variable entering the basis
y = np.matmul(cb, np.transpose(BInv))

# reduced costs
reducedCost = {}
for i in allVars:
    if i not in basisVars:
        print(i)
        reducedCost[i] = c[:,i].item() - np.dot(y, A[:,i]).item()

# variable enterting the basis                 
xEnter = min(reducedCost, key=reducedCost.get)


                  
# find the variable leaving the basis


