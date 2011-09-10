import numpy as np 
import random 
import math 

import tree 

def _majority(classes, labels): 
    votes = np.zeros(len(classes))
    for i, c in enumerate(classes):
        votes[i] = np.sum(labels == c)
    majority_idx = np.argmax(votes)
    return classes[majority_idx] 
    
class RandomForest:
    def __init__(self, classes = None, bag_percent=0.7, numtrees = 50):
        self.classes = None
        self.trees = [] 
        self.bag_percent = bag_percent
        self.numtrees = numtrees 
        
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
            
            t = tree.RandomizedTree(classes=classes)
            t.fit(data_bag,label_bag)
            self.trees.append(t)
            
    def predict(self, X):
        votes = [t.predict(X) for t in self.trees]
        return majority(self.classes, votes)
        
            
            
        
