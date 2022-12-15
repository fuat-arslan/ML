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
        return clean_X.astype(float), clean_Y.reshape(-1,1)
    
    def fast(self, sdv_norm = False):
        return self.learn(sdv_norm).execute()




class StratifiedTrainValTestSplit:
    def __init__(self, X, y, train_size=0.8, val_size=0.1, test_size=0.1, stratify=True, random_state=None):
        self.X = X
        self.y = y
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state

    def split(self):
        X, y = self.X, self.y
        train_size, val_size, test_size = self.train_size, self.val_size, self.test_size

        # First, check if we should stratify the data
        if self.stratify:
            # Divide the data into classes
            classes, y_indices = np.unique(y, return_inverse=True, axis = 0)
            y_counts = np.bincount(y_indices)

            # Then, divide the data into train, val, and test sets
            train_counts = (train_size * y_counts).astype(int)
            val_counts = (val_size * y_counts).astype(int)
            test_counts = (test_size * y_counts).astype(int)

            X_train, y_train = [], []
            X_val, y_val = [], []
            X_test, y_test = [], []

            for class_, count in zip(classes, y_counts):
                class_indices = np.where(y == class_)[0]

                if self.random_state is not None:
                    np.random.seed(self.random_state)
                    np.random.shuffle(class_indices)

                train_indices = class_indices[:train_counts[class_]]
                val_indices = class_indices[train_counts[class_]:train_counts[class_]+val_counts[class_]]
                test_indices = class_indices[-test_counts[class_]:]

                X_train.append(X[train_indices])
                y_train.append(y[train_indices])
                X_val.append(X[val_indices])
                y_val.append(y[val_indices])
                X_test.append(X[test_indices])
                y_test.append(y[test_indices])

            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)
            X_val = np.concatenate(X_val)
            y_val = np.concatenate(y_val)
            X_test = np.concatenate(X_test)
            y_test = np.concatenate(y_test)

            if self.random_state is not None:
                np.random.seed(self.random_state)
                
                idx1 = np.random.permutation(len(X_train)).astype(int)
                X_train = X_train[idx1]
                y_train = y_train[idx1]

                idx2 = np.random.permutation(len(X_val)).astype(int)
                X_val = X_val[idx2]
                y_val = y_val[idx2]

                idx3 = np.random.permutation(len(X_test)).astype(int)
                X_test = X_test[idx3]
                y_test = y_test[idx3]

          
        else: 
             

            # Split the data and labels into random train, validation, and test sets
            data_size = X.shape[0]
            train_count = int(data_size * self.train_size)
            val_count = int(data_size * self.val_size)
            test_count = int(data_size * self.test_size)

            # Create random indices for the train, validation, and test sets
            indices = np.random.permutation(data_size)
            train_indices = indices[:train_count]
            validation_indices = indices[train_count:train_count+val_count]
            test_indices = indices[train_size+val_count:]

            # Use the indices to create the train, validation, and test sets
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_val = X[validation_indices]
            y_val = y[validation_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]


        

        return X_train, X_val, X_test, y_train, y_val, y_test