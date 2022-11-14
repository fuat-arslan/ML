"""
This will include measurment metrics 
"""
import copy as cp

import numpy as np


class kFold():
    def __init__(self):
        pass
    
    
    def eval(self,model,X, y, num_folds = 3):
        """
        real function to run CV alorithm
        """
        total = 0
        
        seperators = [a for a in range(0,len(X),int(len(X)/num_folds))]
        X_copy = X.copy()
        y_copy = y.copy()
        for i in range(1,num_folds):

            model_copy = cp.deepcopy(model)
            X_val = X_copy[seperators[i-1]:seperators[i]].copy()
            y_val = y_copy[seperators[i-1]:seperators[i]]
            X_train = np.delete(X_copy,range(seperators[i-1],seperators[i]),axis = 0)
            y_train = np.delete(y_copy,range(seperators[i-1],seperators[i]),axis = 0)

            model_copy.learn(X_train,y_train)
            pred = model_copy.predict(X_val)

            mse = ((pred - y_val)**2).mean(axis=0) 

            total += mse
        return total/num_folds


