import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import math as m


# DO NOT USE LAMBDA WHEN USING CUPY
# AVOID USING PYTHON FUNCTION FOR CUDA-ASSOCIATED TASKS

# 1)
def prob(is):
    exp_is = [pow(np.e, x) for x in is]
    exp_sum = sum(exp_is)
    ans = [x / exp_sum for x in exp_is]
    return ans


# 3)
class SimpleFFN(nn.Module):
    def __init__(self) -> None:
        super(SimpleFFN, self).__init__()
        self.fc1 = nn.Linear(6, 2)
        self.fc1 = nn.Linear(2, 3)
    def forward(self, x):
        out = F.tanh(self.fc1(x))
        out = F.sigmoid(self.fc2(out))
        return out
    
