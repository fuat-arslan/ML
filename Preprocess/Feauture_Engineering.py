import numpy as np
import time
import copy as cp

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
            
            self.model.learn(X,Y,*argv)
            soft_out_bench= self.model.predict(X)
                
            pred_bench = np.argmax(soft_out_bench, axis = 0)
            true_label_bench = np.argmax(Y, axis = 1)
        

            bench_acc = np.sum(pred_bench==true_label_bench)/pred_bench.shape[0]
            

            temp_acc_list = np.zeros((X.shape[1], 1))

            self.model.layers[0].__init__(self.model.layers[0].input_dim-1,self.model.layers[0].output_dim)
            for j in range(1,len(self.model.layers)):
                self.model.layers[j].__init__(self.model.layers[j].input_dim,self.model.layers[j].output_dim)
            opt = argv[0]
            opt.__init__(opt.method, opt.learning_rate, opt.beta, opt.beta1,opt.beta2)
            #self.model.__init__(self.model.layers,self.model_out)
            #self.model = cp.deepcopy(self.model)
            
            
            for i in range(X.shape[1]):
                # fit model by droping one column for each column, get acc
                print(f"\n The feature of {self.col_names[i]} is removed")
                temp_data = np.delete(X, i, axis = 1)
                for j in range(0,len(self.model.layers)):
                    self.model.layers[j].__init__(self.model.layers[j].input_dim,self.model.layers[j].output_dim)
                opt = argv[0]
                opt.__init__(opt.method, opt.learning_rate, opt.beta, opt.beta1,opt.beta2)

                self.model.learn(temp_data, Y, *argv) 
                soft_out = self.model.predict(temp_data)
                
                pred = np.argmax(soft_out, axis = 0)
                true_label = np.argmax(Y, axis = 1)
            

                temp_acc = np.sum(pred==true_label)/pred.shape[0]
                temp_acc_list[i]= temp_acc

                print('\n')
            t2 = time.time()

            # get unuseful data and drop it
            idx_col = np.argmax(temp_acc_list)
            if temp_acc_list[idx_col] > bench_acc and (temp_acc_list[idx_col] - bench_acc) > self.stop_cond:
                X = np.delete(X, idx_col, axis = 1)
                print("The column {} is dropped since accuray increased from {} to {}".format(self.col_names[idx_col], np.round(100*bench_acc,4),  np.round(100*temp_acc_list[idx_col],4)))
                print("Elapsed time for droping {} column is {} mins {} secs".format(self.col_names[idx_col], (t2-t1)//60, (t2-t1) - (t2-t1)//60 * 60))
                np.delete(self.col_names,idx_col)

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
        


