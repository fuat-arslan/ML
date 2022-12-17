import math
import numpy as np
import matplotlib.pyplot as plt

"""**utils**"""

def sigmoid(Z, dA = 0):

    exp_Z = np.exp(-Z)
    if not isinstance(dA, np.ndarray):
        A = 1/(1+exp_Z)
    else:
        s = 1/(1+exp_Z)
        A = dA * s * (1 - s)
    return A



def tanh(Z, dA = None):
    if not isinstance(dA, np.ndarray):
        A = np.tanh(Z)
    else:
        A = dA *  (1 - np.tanh(Z)**2)
    return A

def softmax(Z):
    num = np.exp(Z)
    denom = np.sum(num, axis = 0)
    A = num / denom
    return A

def stable_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    

def relu(Z, dA = 0):
    if not isinstance(dA, np.ndarray):
        A = np.maximum(0, Z)
    else:
        dZ = np.array(Z, copy=True)
        dZ[dA <= 0] = 0
        A = dZ
    return A

def Cross_Entropy_Loss(O, Y,back = False):
    if not back:
        eps = 1e-15
        L = -np.sum(np.log(O+eps) * Y)
    else:
        L = O - Y
    return L


#Network weight initilazier

def initializer(input_dim, output_dim, method):
    if method == "Xavier":
        w_i = (6 / (input_dim + output_dim)) ** 0.5
        weights = np.random.uniform(-w_i, w_i, size = (output_dim, input_dim))
    
    elif method == "Normal":
        weights = np.random.normal(size = (output_dim, input_dim))
    
    elif method == "He":
        he = 2 / (input_dim) ** 0.5
        weights = np.random.rand(output_dim, input_dim) * he
    return weights

#Random mini batch generator

def mini_batch_generator(X, Y, batch_size):
    m = X.shape[1]  
    mini_batches = []

    idx = list(np.random.permutation(m))
    shuffled_X = X[:, idx]
    shuffled_Y = Y[:, idx]

    full_batches = math.floor(m/batch_size)
    for k in range(0, full_batches):
        mini_batch_X = shuffled_X[:, k * batch_size : (k+1) * batch_size]
        mini_batch_Y = shuffled_Y[:, k * batch_size : (k+1) * batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[:, batch_size * math.floor(m / batch_size) : ]
        mini_batch_Y = shuffled_Y[:, batch_size * math.floor(m / batch_size) : ]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches