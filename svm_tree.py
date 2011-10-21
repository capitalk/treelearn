
import math 
import numpy as np
import sklearn.svm 
import sklearn.linear_model 

from tree_helpers import majority 
from constant_leaf import ConstantLeaf 

        
class svm_tree_node:
    def __init__(self, classes, C=1, depth=1, max_depth=3, min_leaf_size=10):
        self.model = None 
        self.depth = depth
        self.C = C
        self.max_depth = max_depth 
        self.min_leaf_size = min_leaf_size
        self.children = {}
        self.classes = classes
         
    def fit(self, X, Y, class_weight='auto'):
        nrows = X.shape[0]
        print "Depth", self.depth, ": Fitting model for", nrows, "feature vectors"
        self.model = sklearn.svm.LinearSVC(C=self.C)
        self.model.fit(X,Y, class_weight=class_weight)
        if self.depth < self.max_depth:
            pred = self.model.predict(X)
            for c in self.classes:
                mask = (pred == c)
                X_slice = X[mask, :]
                Y_slice = Y[mask, :]
                count = np.sum(mask)
                if count == 0:
                    child = constant_tree_node(c)
                elif count < self.min_leaf_size:
                    majority_label = majority (self.classes, Y_slice)
                    child = ConstantLeaf(majority_label)
                else: 
                    child = svm_tree_node(
                        classes = self.classes, 
                        depth = self.depth +1, 
                        max_depth = self.max_depth, 
                        min_leaf_size = self.min_leaf_size)
                    child.fit(X_slice, Y_slice, class_weight=class_weight)
                self.children[c] = child 
    
    def predict(self, X):
        nrows = X.shape[0]
        curr_labels = self.model.predict(X)
        if self.depth >= self.max_depth:
            return curr_labels
        else:
            # fill this array with sub-arrays received from the predictions of different children 
            final_labels = np.zeros(nrows)
            for c in self.classes:
                mask = (curr_labels == c)
                if np.sum(mask) > 0:
                    X_slice = X[mask, :] 
                    final_labels[mask] = self.children[c].predict(X_slice)
            return final_labels
        
class svm_tree:
    def __init__(self, C=1, max_depth=3, min_leaf_size=10):
        self.C = C
        self.max_depth = max_depth 
        self.min_leaf_size = 10
        self.root = None
        self.classes = []
        
    def fit(self, X,Y, class_weight='auto'):
        self.classes = list(np.unique(Y))
        self.root = svm_tree_node(
            classes = self.classes, 
            C = self.C,
            depth = 1, 
            max_depth = self.max_depth,
            min_leaf_size = self.min_leaf_size)
        self.root.fit(X,Y, class_weight=class_weight)
        
    def predict(self, X):
        return self.root.predict(X)
        
