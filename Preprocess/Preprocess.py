"""
This scirpt includes function for preprocessing.
"""

import numpy as np
import pandas as pd
#Pandas Numpy converter

class pd_np_Converter():

    def __init__ (self,X,col = None):
        self.X = X
        if isinstance(X, pd.DataFrame):
            self.pandas_inst = True
            self.columns = X.columns
        else :
            self.pandas_inst = False
            self.columns = col
    
    def to_nup(self):
        if self.pandas_inst:
            return self.X.to_numpy()
        else:
            return self.X
    def to_pd(self):
        if self.pandas_inst:
            return self.X
        else:
            return pd.DataFrame(self.X,self.columns)


#normalization

class Normalizer():
    """
    It calculates the mean and standard deviation of the each future.
    It subrtacts the mean and divide by std to normalize.
    """
    def __init__ (self):
        pass
    
    def learn(self,X):
        self.MEAN = np.mean(X,axis=0) #Calculate the mean 
        self.STD = np.std(X,axis=0) #Calculate the STD of each feature
        return self

    def execute(self,X):
        output = (X - self.MEAN)/self.STD
        return output

    def fast(self,X):
        return self.learn(X).execute(X)
    
#MinMax

class MinMax():
    """
    It maps the values to a ceratin interval.

    """
    def __init__(self,low = 0, high = 1):
        self.low_ = low
        self.high_ = high

    def learn(self,X):
        self.MIN = np.min(X,axis=0)
        self.MAX = np.max(X,axis=0)
        return self
    
    def execute(self,X):
        X_scaled = (X-self.MIN)/(self.MAX -self.MIN) #X features are mapped (0,1)
        X_rescaled = X_scaled * (self.high_-self.low_) + self.low_
        return X_rescaled
    
    def fast(self,X):
        return self.learn(X).execute(X)



