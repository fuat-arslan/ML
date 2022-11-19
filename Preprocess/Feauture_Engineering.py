import numpy as np
import time


class PCA():
    def __init__(self):
        pass

    def learn(self,X,num_component=5):
        self.MEAN = np.mean(X, axis = 0)
        self.COV = np.cov((X - self.MEAN), rowvar = False)
        self.EIG_VAL , self.EIG_VEC = np.linalg.eigh(self.COV)
        self.num_component_ = num_component
        return self
    def execute(self,X):
        indexes = np.argsort(self.EIG_VAL)[::-1]
        sorted_eigenvalue = self.EIG_VAL[indexes]
        sorted_eigenvectors = self.EIG_VEC[:,indexes]

        eig_subset = sorted_eigenvectors[:,0:self.num_component_]
        X_reduced = (X - self.MEAN) @ eig_subset

        return X_reduced
    
    def fast(self,X,num_component=5):
        return self.learn(X,num_component).execute(X)


class Correlation():
    def __init__(self):
        pass

    def learn(self,X):
        pass

    def execute(self,X):
        return np.corrcoef(X,rowvar =False)

    def fast(self,X):
        return self.learn(X).execute(X)




 #Backward Elimination

class BackwardElimination():

    def __init__(self, num_elim, stop_cond, model, col_names):
        self.num_elim = num_elim
        self.stop_cond = stop_cond
        self.model = model
        self.col_names = col_names

    def learn(self, X, Y, *argv):
        
        self.del_col_names = []
        # diff is initialized to make while conditioon true at the beginning
        eliminated = 0
        diff = self.stop_cond + 1
        while eliminated < self.num_elim or self.stop_cond < diff:
            #benchmark model
            t1 = time.time()
            self.model.fit(X)
            pred = self.model.predict(X)
            bench_acc = self.model.evaluate(X, Y, acc = True)

            temp_acc_list = np.zeros((X.shape[1], 1))

            for i in range(X.shape[1]):
                # fit model by droping one column for each column, get acc
                temp_data = np.delete(X, i, axis = 1)
                self.model.learn(temp_data, Y, *argv) 
                soft_out = self.model.predict(temp_data)
                
                pred = np.argmax(soft_out, axis = 0)
                true_label = np.argmax(Y, axis = 1)
            

                temp_acc = np.sum(pred==true_label)/pred.shape[0]
                temp_acc_list[i]= temp_acc
            
            t2 = time.time()

            # get unuseful data and drop it
            idx_col = np.argmax(temp_acc_list)
            if temp_acc_list[idx_col] > bench_acc and (temp_acc_list[idx_col] - bench_acc) > self.stop_cond:
                X = np.delete(X, idx_col, axis = 1)
                print("The column {} is dropped since accuray increased from {} to {}".format(self.col_names[idx_col], bench_acc, temp_acc_list[idx_col]))
                print("Elapsed time for droping {} column is {} mins {} secs".format(self.col_names[idx_col], (t2-t1)//60, (t2-t1) - (t2-t1)//60 * 60))
                self.col_names.remove(self.col_names[idx_col])

                eliminated += 1
                diff = temp_acc_list[idx_col] - bench_acc

            else:
                print("Stopping condition has satisfied, no more elimination")
                return self 

    def execute(self, X):
        X = np.delete(X, self.del_col_names, axis = 1)
        return X

    def fast(self, X, Y, *argv):
        return self.learn(X, Y, *argv).execute(X)
        


