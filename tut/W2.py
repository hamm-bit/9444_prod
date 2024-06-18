import numpy as np
import numpy.linalg as nla
import numpy.random as nra
# import cupy as np
# import cupy.linalg as nla
# import cupy.random as nra
import scipy as sp
import scipy.linalg as sla
import scipy.optimize as opt
# import matplotlib as plt
# from mpl_toolkits import mplot3d
# import torch

# Add lambda's here
inff = lambda F, x: F @ x                       # inference

# get 1st derivative of given F polynomial
def pnabla(F):
    RC = F.shape
    dF = np.zeros(RC)
    for i in range(RC[0]):
        for j in range(RC[1] - 1):
            dF[i, j] = F[i, j + 1] * (RC[1] - j - 1)
    return dF

pnabla2 = lambda F: pnabla(pnabla(F))           # get 2nd derivative of given F polynomial

# A sample function
def infF(theta):
    return theta[0]**3 + theta[1]**3

# Definitions start

'''
    F: kernel
    x_0: init param
    y: init output
    its: max iterations
    tol: tolerance
'''
def grad_desc(F, x_0, y, its, tol):
    # Find dF/dx, d^2F/dx^2 >= 0
    # Determine a direction g to decrease F(x_0)
    
    # Choose MSE as l
    for i in range(its):
        d2F = pnabla2(F)
        err = nla.norm(y - inff(F, x_0))
        # print(f"err = {err}")
        # get \nabla F and \nabla^2 F
        if (inff(d2F, x_0) >= 0 and err <= tol):
            # finish, we have found the local minimum
            print(f"its taken = {i}")
            return F

        # saddle point, not local minimum
        # we should change the direction of descent
        errF = (inff(F, x_0) - y) @ x_0.T

        s = alpha * errF
        F -= s
        
    # its reached, this (possibly) isn't the local minimum
    print("its full, descent incomplete")
    return F



# demonstrated with a random dataset
# note that the randomness will cause the model to have a large MSE
nra.seed(42)

alpha = 0.002
F = np.zeros((100, 1)).T
x_0 = np.ones((100, 1))
y = nra.randn(1, 1)

# check print correct
print(f"{inff(pnabla(F), x_0)}")

grad_desc(F, x_0, y, 1000000, 1e-2)
print(f"F = {F}")


# back propagation
def backprop(Fm, x_0, y, its, tol):
    # backprop can be conducted with gradient descent for all layers
    # For all layers, propagate from the beginning of the error
    # In the above case, it is L2 error

    # i.e. for each dZ_#
    '''
    F[0:] = grad_desc(Fm[0:], dZ_1, y, its tol)
    F[1:] = grad_desc(Fm[1:], dZ_2, y, its tol)
    F[2:] = grad_desc(Fm[0:], dZ_3, y, its tol)
    ... # all the layers will be iterated
    '''
    pass
