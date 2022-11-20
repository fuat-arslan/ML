import numpy as np 
class KNN():
    def __init__(self, X, Y, num_neig, metric ="euclidian" ,weighted =False):
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
        eps = 1e-13
        predictions = []
        for j in range(len(sample)):
            sample_hat = np.tile(sample[j], (len(self.X), 1))
            #print(len(self.X))
            #print(sample_hat.shape)
            dist = self.distance_metric(self.metric, self.X, sample_hat)
            sorted_idx = np.argsort(dist)
            if weighted:
                u, indicies = np.unique(self.Y[sorted_idx[:self.num_neig]],return_inverse=True)
                indexed_distances = dist[sorted_idx[:self.num_neig]]
                weighted_label = np.zeros(len(np.unique(self.Y)))
                for k in range(len(indicies)):
                    weighted_label[indicies[k]] += 1/(indexed_distances[k]+eps)

                #print('predicition',u[np.argmax(weighted_label)])
                predictions.append(u[np.argmax(weighted_label)])
            else:
                predictions.append(np.bincount(self.Y[sorted_idx[:self.num_neig]].astype(int).reshape(-1)).argmax())
        return predictions  
