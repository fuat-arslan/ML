"""
This scirpt includes trainer of the network.
"""

from utils import mini_batch_generator, Cross_Entropy_Loss

import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self, layers, cost_function):
        self.layers = layers
        self.cost_function = cost_function
    
    def learn(self, X, Y, optimizer, max_epoch, batch_size):

        if self.cost_function == 'CrossEntopy':
            cost = Cross_Entropy_Loss

        else:
            print('invalid cost metric!')


        train_losses = []
        val_losses = []
        
        X = X.T
        Y = Y.T
        
        epoch = 0
        loss_list = []
        
        while epoch < max_epoch:
            loss = 0
            batches_list = mini_batch_generator(X, Y, batch_size)
            for batch in batches_list:
                A, label = batch
                for layer in self.layers:
                    A = layer.forward(A)
                
                loss += cost(A, label) / A.shape[1]

                dA = cost(A, label, back = True)
                for layer in self.layers[::-1]:
                    dA = layer.backward(dA)

                optimizer.step(self.layers)

            if epoch % 10 == 0:
                print ("Cost after epoch %i: %f" %(epoch, loss/len(batches_list)))
            loss_list.append(loss/ len(batches_list))
            epoch += 1

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("NN Loss vs Epoch Graph")
        plt.plot(loss_list)
        return

    def predict(self, data):
        A = data.T.copy()
        for layer in self.layers:
            A = layer.forward(A)
        
        return A
