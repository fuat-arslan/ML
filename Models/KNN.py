import numpy as np 
class KNN():
    def __init__(self, X, Y, num_neig, metric ="euclidian" ):
        self.X = X 
        self.Y = Y
        self.num_neig = num_neig
        self.metric = metric
    
    def distance_metric(self, metric, mat1, mat2):
        if metric == "euclidian":
            return np.sqrt(np.sum((mat1 - mat2) ** 2, axis = 1))

    
    def learn(self):
        return self
    
    def predict(self, sample):
        predictions = []
        for j in range(len(sample)):
            sample_hat = np.tile(sample[j], (len(self.X), 1))
            #print(len(self.X))
            #print(sample_hat.shape)
            dist = self.distance_metric(self.metric, self.X, sample_hat)
            sorted_idx = np.argsort(-dist)
            #print(sorted_idx)
            #print(self.Y[sorted_idx[:self.num_neig]])
            #print(np.bincount(int(self.Y[sorted_idx[:self.num_neig]])))
            predictions.append(np.bincount(self.Y[sorted_idx[:self.num_neig]].astype(int).reshape(-1)).argmax())
        return predictions  
