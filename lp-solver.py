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
allVars = list(range(numVars))
basisVars = list(range(numVars-numBasis, numVars))
B = A[:, basisVars]


iteration = 0
reducedCost = {1:1} #initial condition to start while loop
while (max(reducedCost.values()) > 0):
# while (iteration <3):
    iteration = iteration + 1
    print(f'iteration: {iteration}------------------------------------------')

    BInv = np.linalg.inv((B))
    xb = np.matmul(BInv, b)
    cb = c[:, basisVars]
    zb = np.dot(cb, xb).item()
    
    print(f'basis variables: {basisVars}')
    print(f'xb: {xb[:,0]}')
    print(f'cb: {cb}')
    print(f'zb = {zb}')
    
    # find the variable entering the basis
    y = np.matmul(cb, BInv)
    
    # reduced costs
    reducedCost = {}
    for i in allVars:
        if i not in basisVars:
            reducedCost[i] = c[:,i].item() - np.dot(y, A[:,i]).item()
            print(f'rc x{i} = {reducedCost[i]}')
            
    if (max(reducedCost.values()) < 0):
        break
    # variable enterting the basis                 
    xEnter = max(reducedCost, key=reducedCost.get)
    print(f'variable entering basis: {xEnter}')
                
    # find the variable leaving the basis
    alpha = np.matmul(BInv, A[:,xEnter]).reshape((numBasis, 1))
    
    # calculate theta but set theta to zero for any alpha == 0
    theta = np.divide(xb, alpha, out=np.zeros_like(xb), where=alpha!=0)
    theta = theta[:,0].tolist()
    print(f'theta: {theta}')
    thetaIndexExit = theta.index(min(j for j in theta if j > 0))
    xExit = basisVars[thetaIndexExit]
    
    print(f'variable leaving basis: {xExit}\n')
    
    # recalculate the basis
    # variables in the basis
    basisVars.remove(xExit)
    basisVars.append(xEnter)
    basisVars.sort()
    
    # recalculate B
    for i in range(len(basisVars)):
        B[:, i] = A[:, basisVars[i]]

# final iteration
print('\nfinal solution ----------------------------------------')

BInv = np.linalg.inv((B))
xb = np.matmul(BInv, b)
cb = c[:, basisVars]
zb = np.dot(cb, xb).item()

print(f'basis variables: {basisVars}')
print(f'xb: {xb[:,0]}')
print(f'cb: {cb}')
print(f'zb = {zb}')