# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 15:21:28 2022

@author: Melih Berk Yılmaz
"""
from tqdm import tqdm
import numpy as np

class Node:
    def __init__(self, X, y, gradient, hessian, max_depth, min_leaf_size, gamma, lmb, cover,  num_class):
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.gain = None
        self.prediction = None
        self.depth = None
        self.gamma = gamma
        self.gradient = gradient
        self.hessian = hessian
        self.lmb = lmb
        self.is_leaf = False
        self.weight = None
        self.cover = cover
        self.num_class = num_class
        #self.calculate_weight()
        self.column_subsample = np.random.permutation(8)[:round(0.8*8)]

    @staticmethod
    def loss_reduction(lhs_gradient, lhs_hessian, rhs_gradient, rhs_hessian, lmb, gamma):
        left = (lhs_gradient ** 2) / (lhs_hessian + lmb)
        right = (rhs_gradient ** 2) / (rhs_hessian + lmb)
        tot = (lhs_gradient + rhs_gradient) ** 2 / (lhs_hessian + rhs_hessian + lmb)
        gain =  1 / 2 * (left + right - tot) - gamma
        return gain

    def find_best_split(self):
        """
        Find the best split for the node
        -------------------------------
        Returns: None
        """
        gain = np.zeros(self.num_class)
        # gain = gain * float("-inf")

        for col in self.column_subsample:
            sorted_row = self.X[np.argsort(self.X[:, col])]
            for row in range(self.X.shape[0]):
                lhs = self.X[:, col] < sorted_row[row, col]
                rhs = self.X[:, col] >= sorted_row[row, col]

                if lhs.sum() < self.min_leaf_size or rhs.sum() < self.min_leaf_size:
                    continue
                lhs_gradient = self.gradient[lhs].sum(axis=0)
                lhs_hessian = self.hessian[lhs].sum(axis=0)
                rhs_gradient = self.gradient[rhs].sum(axis=0)
                rhs_hessian = self.hessian[rhs].sum(axis=0)
                temp_gain = self.loss_reduction(lhs_gradient, lhs_hessian, rhs_gradient, 
                                                rhs_hessian, self.lmb,self.gamma)
                
                if temp_gain.sum() > gain.sum():
                    #print("içerdeyim baba")
                    gain = temp_gain
                    self.split_feature = col
                    self.split_value = self.X[row, col]
        
        if gain.all()  == np.zeros(self.num_class).all():
            self.is_leaf = True
            #self.calculate_weight()
        self.gain = gain

    def split(self):
        """
        Split the node into two child nodes
        ----------------------------------
        Returns: None
        """
        
        self.is_leaf_node()
        
        if not self.is_leaf:
            
            lhs = self.X[:, self.split_feature] < self.split_value
            rhs = self.X[:, self.split_feature] >= self.split_value
             
            self.left = Node(self.X[lhs], self.y[lhs], self.gradient[lhs], self.hessian[lhs],
                             self.max_depth - 1, self.min_leaf_size,self.gamma, 
                             self.lmb, self.cover, self.num_class)
    
            self.right = Node(self.X[rhs], self.y[rhs], self.gradient[rhs], self.hessian[rhs],
                              self.max_depth - 1, self.min_leaf_size, self.gamma, 
                              self.lmb, self.cover, self.num_class)
        else:
            self.calculate_weight()

    def is_leaf_node(self):
        """
        Check if the node is a leaf node
        --------------------------------
        Returns: boolean
        """
        #or self.weight < self.cover
        if self.max_depth <= 0 :
            #self.calculate_weight()
            self.is_leaf == True
        
    def calculate_weight(self):
        """
        Calculate the weight of the node
        --------------------------------
        Returns: None

        #inf can be another condition
        """
        
        gradient = np.sum(self.y * self.gradient, axis=0)
        hessian = np.sum(self.hessian, axis=0)
        self.weight =  -gradient / (hessian + self.lmb)
        
        #self.weight = - self.gradient.sum(axis=0) / (self.hessian.sum(axis=0) + self.lmb)

    
    def predict_sample(self, sample):
        if self.is_leaf:
            return self.weight

        if sample[self.split_feature] <= self.split_value:
            node = self.left
        else:
            node = self.right
        return node.predict_sample(sample)

    def predict(self, X):
        """
        Predict the class of a given sample
        -----------------------------------
        --X: sample
        Returns: prediction
        """
        prediction = np.zeros((X.shape[0], self.num_class))
        for k, sample in enumerate(X):
          #  print(sample)
            prediction[k] = self.predict_sample(sample)
       # print(prediction[-5:-1])
        return prediction

class Tree:
    def __init__(self, X, y, gradient, hessian, max_depth, min_leaf_size, gamma, lmb, cover, num_class):
        self.root = Node(X, y, gradient, hessian, max_depth, min_leaf_size, gamma, lmb, cover, num_class)
        #self.root.calculate_weight()
        self.root.depth = 0
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.gamma = gamma
        self.lmb = lmb
        self.cover = cover
        self.num_class = num_class

    def grow_tree(self, node):
        """
        Grow the tree
        -------------
        --node: node to grow
        Returns: None
        """

        node.find_best_split()
        node.split()
        
        if node.is_leaf:
            return 

        self.grow_tree(node.left)
        self.grow_tree(node.right)


    def predict(self, X):
        """
        Predict the class of a given sample
        -----------------------------------
        --X: sample
        Returns: prediction
        """
        return self.root.predict(X)

    def learn(self):
        """
        Fit the tree
        ------------
        Returns: None
        """

        self.grow_tree(self.root)


class XGBoost_Classifier:
    def __init__(self, n_trees=10, max_depth=3, min_leaf_size=20, learning_rate=0.3, gamma=0, lmb=1, cover=1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmb = lmb
        self.cover = cover



    def gradient_hessian(self, y, y_pred):
        """
        Calculate the gradient and hessian
        ----------------------------------
        --y: true labels
        --y_pred: predicted labels
        Returns: gradient, hessian
        """
        
        gradient = y_pred - y
        hessian = y_pred * (1 - y_pred)
        return gradient, hessian

    def learn(self, X, y):
        """
        Fit the model
        -------------
        --X: training data
        --y: training labels
        Returns: None
        """
        self.trees = []
        self.X = X
        self.y = y
        self.n_classes = len(np.unique(y, axis=0))
        #self.init_prediction = np.array([1/self.n_classes] * (self.y.shape[0]*self.y.shape[1])).reshape(self.y.shape)
        self.init_prediction = np.random.rand(self.y.shape[0], self.n_classes)
        self.init_prediction = self.stable_softmax(self.init_prediction)

        self.y_pred = self.init_prediction
        for i in tqdm(range(self.n_trees)):
            gradient, hessian = self.gradient_hessian(self.y, self.y_pred)

            tree = Tree(self.X, self.y, gradient, hessian, self.max_depth, 
                        self.min_leaf_size, self.gamma, self.lmb, 
                        self.cover, self.n_classes)
            tree.learn()
            self.trees.append(tree)
            self.y_pred += self.learning_rate * tree.predict(self.X)
            self.y_pred = self.stable_softmax(self.y_pred)
            

    @staticmethod
    def stable_softmax(x):
        """
        Softmax function
        ----------------
        --x: input
        Returns: softmax(x)
        """
        e_x = np.exp(x - np.max(x, axis = 1, keepdims = True))
        return e_x / e_x.sum(axis = 1, keepdims = True)

    def predict(self, X,argmax=True):
        """
        Predict the class of a given sample
        -----------------------------------
        --X: sample
        Returns: prediction
        """
        y_pred1 = np.zeros((X.shape[0], self.n_classes))
        for tree in self.trees:
            y_pred1 += self.learning_rate * tree.predict(X)
        
        y_pred = self.stable_softmax(y_pred1)
        if argmax:
            return np.argmax(y_pred, axis=1)
        return y_pred1