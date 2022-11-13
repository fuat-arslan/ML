class KMeans():
  def __init__(self, num_cluster, state, metric, seed = 1):
    self.num_cluster = num_cluster
    self.state = state
    self.seed = seed
    self.centroids = []
    self.metric = metric

  def initialize(self, min, max, data):
    if self.state == "random":
      np.random.seed(self.seed)
      self.centroids = [np.random.uniform(min,max) for _ in range(self.num_cluster)]
      self.centroids = np.array(self.centroids)

    elif self.state == "data":
      idx = np.random.randint(0, data.shape[0], 3)
      self.centroids = data[idx]

    elif self.state == "k-means ++":
        idx = np.random.randint(0, data.shape[0])
        self.centroids.append(data[idx])
        self.centroids = np.array(self.centroids)
        print(self.centroids)

        for cent in range(self.num_cluster-1):
            print('cent is ', cent)
            min_distances=[]

            for point_idx in range(data.shape[0]):

                distances = []

                
                #look for predetermined centroids and find min distance
                #we determined cent+1 of centroid
                for pre_cid in range(cent+1):
                    #print(self.centroids[pre_cid].shape)
                    print(pre_cid)
                    dist = self.distance_metric("euclidian",data[point_idx],self.centroids[pre_cid].reshape((1,-1)))
                    distances.append(dist[0])

                # points possible minimum distances are stored. 
                #print(distances)
                foo = np.min(distances)
                min_distances.append(foo)
        
            min_distances = np.array(min_distances)
            furthest_point_idx = np.argmax(min_distances)
            new_centroid = data[furthest_point_idx]
            print(new_centroid)
            self.centroids = np.append(self.centroids, new_centroid.reshape((1,-1)),axis = 0) 
            print('new cent')


        print("a")
    else:
        print('Wrong initialization Method')

  def distance_metric(self, metric, mat1, mat2):
    if metric == "euclidian":
      return np.sqrt(np.sum((mat1 - mat2) ** 2, axis = 1))
  
  def centroid_assignment(self, sample):
    sample_hat = np.tile(sample, (self.num_cluster, 1))
    distances = self.distance_metric(self.metric, sample_hat, self.centroids)
    return np.argmin(distances)

  def learn(self, X, epoch = 100, tolerance = 1e-4):
    self.tolerance = self.num_cluster * tolerance
    self.min_, self.max_ = np.min(X, axis=0), np.max(X, axis=0)
    # random centroids are initialized
    self.initialize(self.min_, self.max_, X) 

    iter, diff = 0, 0
    new_centroids = np.zeros((self.num_cluster, X.shape[1]))
    
    while iter < epoch or diff > self.tolerance:
        # cluster list will be used to keep data indexes of each cluster
        cluster_list = [[] for i in range(self.num_cluster)]
        for idx in range(X.shape[0]):
          cluster_id = self.centroid_assignment(X[idx])
          cluster_list[cluster_id].append(idx)
        
        # centroids update and tolerans calculation
        for i, cidx in enumerate(cluster_list):
          new_centroids[i] = np.mean(X[cidx], axis = 0)
        
        diff = np.sum(self.distance_metric(self.metric, 
                                            self.centroids, new_centroids))
        self.centroids = new_centroids
        if iter % 10 == 0:
          print(iter)
        iter +=1
          
  def execute(self):
    pass

  def predict(self, X):
    label_list = []
    for idx in range(X.shape[0]):
      cluster_id = self.centroid_assignment(X[idx])
      label_list.append(cluster_id)
    return label_list

  def label_iden(self,X,y):
        label_map = {}
        # cluster list will be used to keep data indexes of each cluster
        cluster_list = [[] for i in range(self.num_cluster)]
        for idx in range(X.shape[0]):
            cluster_id = self.centroid_assignment(X[idx])
            cluster_list[cluster_id].append(idx)
        for i, cidx in enumerate(cluster_list):
            #taken from stackoverflow 
            #https://stackoverflow.com/questions/19909167/how-to-find-most-frequent-string-element-in-numpy-ndarray
            unique,pos = np.unique(y[cidx],return_inverse=True) #Finds all unique elements and their positions
            counts = np.bincount(pos)                     #Count the number of each unique element
            maxpos = counts.argmax()
            label_map[str(i)] = unique[maxpos]
            #label_map[str(i)] = np.bincount(y[cidx]).argmax()
        return label_map

  def evaluate(self, predict, ground_truth):
        acc = np.sum(predict == ground_truth) / predict.shape[0]
        return acc