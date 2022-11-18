"""
Optimizer scirpt. For now there is only adam optimizer.
"""

import numpy as np 


class Optimizer():
    def __init__(self,method, learning_rate, beta, beta1, beta2, epsilon = 1e-8):
        self.method = method
        self.learning_rate = learning_rate
        self.t = 1
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.flag = True

    def initializer(self, layer_list):
        if self.method == "gd":
            pass
        if self.method == "sgd":
            pass
        if self.method == "momentum":
            pass
        if self.method == "adam":
            self.v = {}
            self.s = {}
            w = []
            b = []
            for layer in layer_list: 
                w.append(np.zeros(layer.weight.shape))
                b.append(np.zeros(layer.bais.shape))
            self.v['W'] = w.copy()
            self.v['b'] = b.copy()
            self.s['W'] = w.copy()
            self.s['b'] = b.copy()


    def step(self, layer_list):
        if self.method == "gd":
            pass
        if self.method == "sgd":
            pass
        if self.method == "momentum":
            pass
        if self.method == "adam":
            if self.flag: 
                self.initializer(layer_list)
                self.flag = False
                 
            v_corrected = {'W':[],  'b':[]}                      
            s_corrected = {'W':[],  'b':[]} 
            for i, layer in enumerate(layer_list):
                self.v['W'][i] = self.beta1 * self.v['W'][i] + (1-self.beta1) * layer.dweight 
                self.v['b'][i] = self.beta1 * self.v['b'][i] + (1-self.beta1) * layer.dbais 
            
                v_corrected['W'].append((self.v["W"][i] / (1-self.beta1**self.t)))
                v_corrected['b'].append((self.v["b"][i] / (1-self.beta1**self.t)))
            
                self.s['W'][i] = self.beta2 * self.s['W'][i] + (1-self.beta2) * layer.dweight **2
                self.s['b'][i] = self.beta2 * self.s['b'][i] + (1-self.beta2) * layer.dbais **2
            
                s_corrected['W'].append((self.s["W"][i] / (1-self.beta2**self.t)))
                s_corrected['b'].append((self.s["b"][i] / (1-self.beta2**self.t)))
            
                new_W = layer.weight -(self.learning_rate * v_corrected['W'][i] / (s_corrected["W"][i]**(0.5) + self.epsilon))
                new_b = layer.bais -(self.learning_rate * v_corrected['b'][i] / (s_corrected["b"][i]**(0.5) + self.epsilon))
                layer.update_params(new_W,new_b)
        
        self.t += 1
        return