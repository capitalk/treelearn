import numpy as np 
import random 
import math 

from tree_helpers import * 
import randomized_tree as tree 
    
class RandomForest:
    def __init__(self, classes = None, bag_percent=0.7, numtrees = 50, **keywords):
        self.classes = None
        self.trees = [] 
        self.bag_percent = bag_percent
        self.numtrees = numtrees 
        self.tree_params = keywords 
    
    def __str__(self): 
        treeStrings = ["tree " + str(i) + ": " + str(t) for i,t in enumerate(self.trees)]
        return "[RandomForest]\n" + "\n".join(treeStrings)
        
    def fit(self, X,Y):
        if self.classes is None: 
            classes = np.unique(Y) 
            self.classes = classes 
        else: classes = self.classes 
        
        self.trees = [] 
            
        n = X.shape[0]
        bagsize = int(self.bag_percent * n)
        permute = np.random.permutation
        
        for i in xrange(self.numtrees):
            p = permute(n)
            indices = p[:bagsize] 
            data_bag = X[indices, :]
            label_bag = Y[indices] 
            
            t = tree.RandomizedTree(classes=classes, **self.tree_params)
            t.fit(data_bag,label_bag)
            self.trees.append(t)
            
    def predict(self, X):
        votes = [t.predict(X) for t in self.trees]
        return majority(self.classes, votes)
        
