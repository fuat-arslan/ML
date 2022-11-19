"""
This will include measurment metrics 
"""
import copy as cp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from utils import *

class kFold():
    def __init__(self):
        pass
    
    
    def eval(self,model,X, y, cost_metric, num_folds = 3,*argv):
        """
        real function to run CV alorithm
        """

        if cost_metric == 'CrossEntropy':
            cost = Cross_Entropy_Loss
        elif cost_metric == 'MSE':
            #mse = ((pred - y_val)**2).mean(axis=0) 
            pass
        else:
            print('Enter a valid cost metric')

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

            model_copy.learn(X_train,y_train,*argv)
            pred = model_copy.predict(X_val)

            loss = cost(pred, y_val.T)
            arg_pred = np.argmax(pred,axis=0)
            true_label = np.argmax(y_val,axis = 1)
            acc = np.sum(arg_pred == true_label)/arg_pred.shape[0]
            print(acc)
            
            total += loss
        return total/num_folds


#Score generator function

class Evaluator():
    def __init__(self, prediction, true_value, label_dict, display = True):
        self.prediction = prediction
        self.true_value = true_value
        self.label_dict = label_dict
        self.display = display

        # sort label dictionary by value
        self.label_dict = dict(sorted(self.label_dict.items(), key=lambda item: item[1]))


    def confusion_matrix(self, prediction, true_value, label_dict, display = True):
        #number of classes
        num_class = np.unique(true_value).shape[0]
        confusion_mat = np.zeros((num_class, num_class))

        for i in range(prediction.shape[0]):
            confusion_mat[prediction[i], true_value[i]] += 1

        if display:
            result = sns.heatmap(confusion_mat, annot=True ,cbar = False,fmt='g')
            result.set(xlabel='Ground Truth', ylabel='Prediction', 
                       xticklabels = list(label_dict.keys()), 
                       yticklabels = list(label_dict.keys()))
            result.xaxis.tick_top()
            result.set_title("Confusion Matrix")
        return confusion_mat
    
    def scores(self):
        plt.figure(figsize = (10,10))
        plt.subplot(2,1,1)
        confusion_mat = self.confusion_matrix(self.prediction, self.true_value, self.label_dict, self.display)
        
        acc = np.trace(confusion_mat)/np.sum(confusion_mat)
        precision = np.diag(confusion_mat) / np.sum(confusion_mat, axis = 0)
        recall = np.diag(confusion_mat) / np.sum(confusion_mat, axis = 1)
        f1_score = 2*precision * recall / (precision + recall)
        
        # recall, precision, f1 score corresponds to columns
        score = np.hstack((precision.reshape(-1,1), 
                           recall.reshape(-1,1), f1_score.reshape(-1,1)))

        plt.subplot(2,1,2)
        result = sns.heatmap(score, annot=True ,cbar = False,fmt='g')
        result.set(xticklabels=["precision", "recall", "F1 score"], 
                   ylabel='Classes', yticklabels = list(self.label_dict.keys()))
        result.xaxis.tick_top()
        result.set_title("Scores")

        print("Accuracy: ", np.round(acc,2))
        return acc, score
        