"""
This script includes neural network layers. For now there is just Dense layer.
"""
import numpy as np
from utils import softmax, sigmoid, tanh, relu, initializer


class Dense():
    def __init__(self, input_dim, output_dim, method = "Xavier", activation = "relu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.method = method
        self.weight = initializer(self.input_dim, self.output_dim, self.method)
        self.bais = np.zeros((output_dim, 1))
    
    def update_params(self, w, b):
        self.weight = w.copy()
        self.bais = b.copy()
        return 

    def forward(self, X):
        # linear forward
        self.X = X
        self.Z = self.weight @ self.X + self.bais

        # activation forward
        if self.activation == "softmax":
            A = softmax(self.Z)
        
        elif self.activation == "relu":
            A = relu(self.Z)
        
        elif self.activation == "sigmoid":
            A = sigmoid(self.Z)
        
        elif self.activation == "tanh":
            A = tanh(self.Z)

        return A

    def backward(self, dA):
        # derivative with respect to activation function        
        if self.activation == "relu":
            dZ = relu(dA, self.Z)
        
        elif self.activation == "sigmoid":
            dZ = sigmoid(self.Z, dA)
        
        elif self.activation == "tanh":
            dZ = tanh(self.Z, dA)
        else:
            dZ = dA
        m = self.X.shape[1]
        # derivative with respect to weights, bais and X(current state)
        self.dweight = 1/m * np.matmul(dZ, self.X.T)
        self.dbais = 1/m * np.sum(dZ, axis = 1, keepdims=True)
        self.dX = np.matmul(self.weight.T, dZ)
        
        return self.dX