import numpy as np
import scipy as sp
import matplotlib as plt
from mpl_toolkits import mplot3d
import torch 
# torch.optim.adam.Adam.zero_grad


# 2)
# a)
w_1 = 1; w_2 = 2; w_0 = -2.5

x_1 = int(input())
x_2 = int(input())

s = w_1 * x_1 + w_2 * x_2 + w_0 
t = (s > 0)


# b)
# w_0 sits between either lines
w_1 = 0; w_2 = 2; w_0 = -1.5

# Recall perceptron learning algorithm (PLA)
s = w_1 * y_1 + w_2 * y_2 + w_0

# TODO


# 3)
# In the order of 
# [w_1, w_2, w_0]

x_a = [1, 1, 1, -0.5]
x_b = [1, 1, 1, -2.5]
# x_b scales with n, with w_0 = -2n + 0.5

# x_c can be simplified to only involve an AND over A, C and !E
x_c = [1, 1, -1, -2.5]

# 4)
# Drawing this neural network
# By inducing a pass-through link from input to output, depending on the truth table:
# 1 1 0
# 1 0 1
# 0 1 1
# 0 0 0

# For an OR yielding 1, and the AND must yield 0 to be true, else false
# We could describe the following network

# Intermediate layer is 2-AMD for inputs
# Outut layer is 3-OR for inputs and Intermediate
# To reject the AND = 1 solution immediately, whilst still leaving space for acceptance for OR = 1
# We shall bias AND heavier negatively.

# Note that $x_1 = 1 or -1$

x_xor = x_1 + x_2 - 2 (x_1 + x_2 - 1.5) - 0.5

