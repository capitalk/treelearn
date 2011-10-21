import copy 
import numpy as np 
import random 
import math 

from tree_helpers import * 
import randomized_tree as tree 
    
class RandomForest:
    """Random Forest ensemble of arbitrary classifiers. For more details, see:
        http://www.stat.berkeley.edu/~breiman/RandomForests/
        
    Parameters
    ----------
    base_classifier : Any classifier which obeys the fit/predict protocol. 
        Defaults to a RandomizedTree. 
        
    num_classifiers : int, optional (default = 50)
        How many base classifiers to train. 
    
    bag_percent : float, optional (default=0.7). 
        How much of the data set goes into each bootstrap sample. 
    """

    def __init__(self, 
            base_classifier=tree.RandomizedTree(), 
            num_classifiers = 50, 
            bag_percent=0.7):
        self.trees = [] 
        self.base_classifier = base_classifier
        self.num_classifiers = num_classifiers
        self.bag_percent = bag_percent
    
    def __str__(self): 
        treeStrings = ["tree " + str(i) + ": " + str(t) for i,t in enumerate(self.trees)]
        return "[RandomForest]\n" + "\n".join(treeStrings)
        
    def fit(self, X,Y):
        self.classes = np.unique(Y) 
        self.trees = [] 
            
        n = X.shape[0]
        bagsize = int(self.bag_percent * n)
        permute = np.random.permutation
        
        for i in xrange(self.num_classifiers):
            p = permute(n)
            indices = p[:bagsize] 
            data_subset = X[indices, :]
            label_subset = Y[indices] 
            t = copy.copy(self.base_classifier)
            t.fit(data_subset, label_subset)
            self.trees.append(t)
            
    def predict(self, X):
        votes = [t.predict(X) for t in self.trees]
        return majority(self.classes, votes)
