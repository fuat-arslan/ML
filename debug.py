# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from Models.NeuralNetwork.Layers import Dense
from Models.NeuralNetwork.Optimizer import Optimizer
from Models.NeuralNetwork.Trainer import NeuralNetwork
from Preprocess.Preprocess import Encoder , OutlierRemoval



data = pd.read_csv("/workspaces/ML/raw_data/archive/sdss-IV-dr16-70k.csv")

data = data.iloc[:, 1: 13]
data = data.sort_values(by = ["class"])
data = data.drop(columns=["run", "rerun", "camcol"])

# g_data = data[:30]
# s_data = data[-30:]
# q_data = data[55000: 55030]

# train_data = pd.concat([q_data, s_data, g_data])


X = data.iloc[:, :-1].to_numpy()
Y = data.iloc[:, -1].to_numpy().reshape(-1,1)

out_remove = OutlierRemoval(X, Y, 3)
clean_data = out_remove.fast(False)

X = clean_data[:, :-1].astype(float)
Y = clean_data[:, -1]

enc = Encoder()
Y = enc.fast(Y.reshape(-1,1))

layers = [Dense(8, 32), Dense(32, 8), Dense(8, 3, activation = "softmax")]
model_NN = NeuralNetwork(layers, "CrossEntopy")
adam = Optimizer("adam", 0.01, None, 0.9, 0.99)

model_NN.learn(X, Y, optimizer = adam, max_epoch = 100, batch_size = 64)

