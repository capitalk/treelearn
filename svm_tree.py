
import math 
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm  import LinearSVC

from tree_helpers import majority 
from constant_leaf import ConstantLeaf 

        
class SVM_Tree_Node(BaseEstimator):
    """Do not use this directly, instead train an SVM_Tree"""
    def __init__(self, 
            classes, 
            num_features_per_node = 1, 
            C=1, 
            depth=1, 
            max_depth=3, 
            min_leaf_size=10):
        self.model = None 
        self.subspace = None 
        self.depth = depth
        self.C = C
        self.max_depth = max_depth 
        self.min_leaf_size = min_leaf_size
        self.children = {}
        self.classes = classes
        self.num_features_per_node = num_features_per_node 
        
    def fit(self, X, Y, **fit_keywords):
        n_samples, n_features = X.shape
        
        #print "Depth", self.depth, ": Fitting model for", n_samples, "vectors"
        C = C = 10.0 ** (np.random.randn()-1) if self.C == 'random' else self.C
        self.model = LinearSVC(C=C)
        
        
        if self.depth >= self.max_depth:
            self.model.fit(X, Y)
        else:
            feature_indices = np.random.permutation(n_features)
            self.subspace  = feature_indices[:self.num_features_per_node]
            X_reduced = X[:, self.subspace]
            self.model.fit(X_reduced, Y, **fit_keywords)
            pred = self.model.predict(X_reduced)
            for c in self.classes:
                mask = (pred == c)
                X_slice = X[mask, :]
                Y_slice = Y[mask, :]
                count = np.sum(mask)
                if count == 0:
                    child = ConstantLeaf(c)
                elif count < self.min_leaf_size:
                    majority_label = majority (self.classes, Y_slice)
                    child = ConstantLeaf(majority_label)
                else: 
                    child = SVM_Tree_Node(
                        classes = self.classes, 
                        num_features_per_node = 1, 
                        depth = self.depth +1, 
                        max_depth = self.max_depth, 
                        min_leaf_size = self.min_leaf_size)
                    child.fit(X_slice, Y_slice, class_weight=class_weight)
                self.children[c] = child 
    
    def predict(self, X):
        nrows = X.shape[0]
        if self.subspace is not None:
            X_reduced = X[:, self.subspace] 
            curr_labels = self.model.predict(X_reduced)
        else:
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
        
class SVM_Tree(BaseEstimator):
    """A decision tree whose splits are SVM hyperplanes. 
    
    Parameters
    ----------------
    num_features_per_node : int, optional(default = sqrt of total feature count)
    
    C : float, optional (default=1). 
        Tradeoff between error and L2 regularizer of SVM optimization problem.
        Can also use value 'random' to use a random value of C for each 
        split in the tree. 
        
    max_depth : int, optional (default=3). 
        The number of SVMs will be at most 2^max_depth
        
    min_leaf_size : int, optional (default=100).
        Don't split data if it gets smaller than this number. 
    """
    
    def __init__(self, 
            num_features_per_node = None, 
            C=1, 
            max_depth=3, 
            min_leaf_size=100):
        self.num_features_per_node = num_features_per_node 
        self.C = C
        self.max_depth = max_depth 
        self.min_leaf_size = 10
        self.root = None
        self.classes = []
        
    def fit(self, X,Y, **fit_keywords):
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)
        
        n_features = X.shape[1]
        num_features_per_node = self.num_features_per_node 
        
        if num_features_per_node is None:
            num_features_per_node = int(math.ceil(math.sqrt(X.shape[0])))
        elif num_features_per_node > n_features:
            num_features_per_node = n_features 
            
        self.classes = list(np.unique(Y))
        
        self.root = SVM_Tree_Node(
            classes = self.classes, 
            num_features_per_node = num_features_per_node, 
            C = self.C,
            depth = 1, 
            max_depth = self.max_depth,
            min_leaf_size = self.min_leaf_size)
        self.root.fit(X, Y, **fit_keywords)
        
    def predict(self, X):
        return self.root.predict(X)
        
