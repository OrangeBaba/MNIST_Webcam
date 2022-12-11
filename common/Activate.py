# 必要な活性化関数
import numpy as np

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

"""
よくない書き方
def ReLU(x):
    if x > 0:
        return x
    else:
        return 0
"""    

def relu():
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x>=0] = 1
    return grad