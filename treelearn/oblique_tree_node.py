from copy import deepcopy 
import numpy as np 
from sklearn.base import BaseEstimator
from tree_helpers import majority, clear_sklearn_fields
from constant_leaf import ConstantLeaf 

class _ObliqueTreeNode(BaseEstimator):
    """
        Do not use this directly, instead train an ObliqueTree"""
    def __init__(self, 
            split_classifier, 
            leaf_model, 
            num_features_per_node, 
            classes, 
            depth, 
            max_depth, 
            min_leaf_size, 
            randomize_split_params, 
            randomize_leaf_params, 
            verbose):
        
        self.split_classifier = split_classifier 
        self.leaf_model = leaf_model 
        self.num_features_per_node = num_features_per_node 
        self.classes = classes
        self.depth = depth
        self.max_depth = max_depth 
        self.min_leaf_size = min_leaf_size
        self.randomize_split_params = randomize_split_params
        self.randomize_leaf_params = randomize_leaf_params
        self.verbose = verbose 
        
        self.children = {}
        self.model = None 
        self.subspace = None 
    
    def _fit_leaf(self, X, Y, fit_keywords):
        model = deepcopy(self.leaf_model)
        for field, gen in self.randomize_leaf_params.items():
            setattr(model, field,  gen())
        model.fit(X, Y, **fit_keywords) 
        clear_sklearn_fields(model)
        return model 
        
    def _fit_child(self, X_slice, Y_slice, fit_keywords):
        count = X_slice.shape[0] 
        unique_ys = np.unique(Y_slice)
        if len(unique_ys) == 1:
            child = ConstantLeaf(int(unique_ys[0]))
        elif count < self.min_leaf_size:
            child = self._fit_leaf(X_slice, Y_slice, fit_keywords)
        else: 
            child = _ObliqueTreeNode(
                split_classifier = self.split_classifier, 
                leaf_model = self.leaf_model, 
                num_features_per_node = self.num_features_per_node, 
                classes = self.classes, 
                depth = self.depth +1, 
                max_depth = self.max_depth, 
                min_leaf_size = self.min_leaf_size, 
                randomize_split_params = self.randomize_split_params, 
                randomize_leaf_params = self.randomize_leaf_params, 
                verbose = self.verbose 
            )
            child.fit(X_slice, Y_slice, **fit_keywords)
        return child 
    
   

    def fit(self, X, Y, **fit_keywords):
        n_samples, n_features = X.shape
        
        if self.verbose: 
            print "Depth", self.depth, ": Fitting model for", n_samples, "vectors"
            
        if self.depth >= self.max_depth or n_samples <= self.min_leaf_size:
            self.model = self._fit_leaf(X, Y, fit_keywords)
        else:
            
            # if we've been passed a limit to the number of features 
            # then train the current model on a random subspace of that size
            if self.num_features_per_node:
                feature_indices = np.random.permutation(n_features)
                self.subspace  = feature_indices[:self.num_features_per_node]
                X_reduced = X[:, self.subspace]
            else:
                X_reduced = X 
            
            self.model = deepcopy(self.split_classifier)
            for field, gen in self.randomize_split_params.items():
                setattr(self.model, field,  gen())
                
            self.model.fit(X_reduced, Y, **fit_keywords)
            clear_sklearn_fields(self.model)
            pred = self.model.predict(X_reduced)
            
            for c in self.classes:
                mask = (pred == c)
                count = np.sum(mask)
                if count == 0:
                    self.children[c] = ConstantLeaf(int(c))
                else:
                    X_slice = X[mask, :] 
                    Y_slice = Y[mask]
                    self.children[c] = self._fit_child(X_slice, Y_slice, fit_keywords)
                    
    def predict(self, X):
        nrows = X.shape[0]
        if self.subspace is not None:
            X_reduced = X[:, self.subspace] 
            pred = self.model.predict(X_reduced)
        else:
            pred = self.model.predict(X) 
        
        if len(self.children) == 0:
            return pred 
        else:
            # fill this array with sub-arrays received from the predictions of children 
            outputs = pred.copy()
            for c in self.classes:
                mask = (pred == c)
                X_slice = X[mask, :]
                count = X_slice.shape[0]
                if count > 0:
                    outputs[mask] = self.children[c].predict(X_slice)
            return outputs 
