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


#one hot encoder 

class Encoder():
    def __init__(self,one_hot = True):
        self.one_hot = one_hot
    
    def learn(self,X):
        self.label_map_list = {}
        self.num_features = X.shape[1]
        self.one_hot_list = {}
        self.categorical_idx = []

        X_copy = X.copy()
        for col_idx in range(self.num_features):
            col = X_copy[:,col_idx]
           
            
            if not np.issubdtype(type(col[0]), np.number):
                #print(type(col[0]))
                foo_map = {}
                self.categorical_idx.append(col_idx)

                u, indices = np.unique(col, return_index=True)

                for i, uniq in enumerate(u):
                    col[col == uniq] = i
                    foo_map[uniq] = i

                col = col.astype(int)
                if self.one_hot:
                    hot = np.zeros((col.size, col.max() + 1))
                    hot[np.arange(col.size), col] = 1
                else:
                    hot = col.reshape((-1,1))

                self.one_hot_list[str(col_idx)] = hot
                self.label_map_list[str(col_idx)] = foo_map
        return self

    def execute(self,X):

        for i in self.one_hot_list.values():
            X = np.concatenate((X, i),axis = 1)
            #print(X)

        X = np.delete(X,self.categorical_idx,1)
            
        return X.astype(float).astype(int)

    def fast(self,X):
        return self.learn(X).execute(X)

#Outlier Removal by Z score


class OutlierRemoval():
    def __init__(self, X, Y, threshold):
        self.X = X
        self.Y = Y
        self.threshold = threshold
    
    def learn(self, sdv_norm = False):

        if not sdv_norm:
            self.X = (self.X - np.mean(self.X, axis = 0)) / np.std(self.X, axis = 0)

        return self

    def execute(self):
        data = np.hstack((self.X, self.Y))

        clean_data = data[np.all(abs(self.X) < self.threshold, axis=1)]
        self.outlier = data[~np.all(abs(self.X) < self.threshold, axis=1)]
        clean_X = clean_data[:,:-1]
        clean_Y = clean_data[:,-1]
        return clean_X, clean_Y.reshape(-1,1)
    
    def fast(self, sdv_norm = False):
        return self.learn(sdv_norm).execute()