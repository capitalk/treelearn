from copy import deepcopy 
import numpy as np 
from sklearn.base import BaseEstimator
from tree_helpers import majority 
from constant_leaf import ConstantLeaf 

class _ObliqueTreeNode(BaseEstimator):
    """
        Set classes = None if doing regression. 
        Do not use this directly, instead train an ObliqueTree"""
    def __init__(self, 
            split_classifier, 
            leaf_model, 
            num_features_per_node = None, 
            classes = None, 
            depth=1, 
            max_depth=3, 
            min_leaf_size=10, 
            error_tol = 0, 
            randomize_split_params={}, 
            randomize_leaf_params={}, 
            verbose = False):
        
        self.split_classifier = split_classifier 
        self.leaf_model = leaf_model 
        self.num_features_per_node = num_features_per_node 
        self.classes = classes
        self.depth = depth
        self.max_depth = max_depth 
        self.min_leaf_size = min_leaf_size
        self.error_tol = error_tol 
        self.randomize_split_params = randomize_split_params
        self.randomize_leaf_params = randomize_leaf_params
        self.verbose = verbose 
        
        self.left = None 
        self.right = None 
        self.model = None 
        self.subspace = None 

    
    def _classifier_split(self, n_samples, Y):
        """split data into biggest predicted class and all others"""
        class_counts = [np.sum(Y == c) for c in self.classes]
        biggest_class_index = np.argmax(class_counts)
        biggest_count = class_counts[biggest_class_index]
        biggest_class = self.classes[biggest_class_index]
        target = np.zeros(n_samples)
        target[Y == biggest_class] = 1
        left_count = n_samples - biggest_count 
        return target, left_count, biggest_count
    
    def _regression_split_target(self, n_samples, Y): 
        """
        Let's try to find a threshold which minimizes variance in 
        two groups of actual outputs 
        """
        thresholds = np.unique(Y)
        best_target = None 
        best_left_count = None 
        best_right_count = None 
        best_score = np.inf 
        for t in thresholds:
            left_mask = (Y <= t)
            left_count = np.sum(left_mask)
            right_mask = ~left_mask
            right_count = n_samples - left_count 
            if left_count > 0 and right_count > 0:
                left_variance = np.var(Y[left_mask])
                right_variance = np.var(Y[right_mask])
                left_weight = float(left_count) / n_samples
                right_weight = float(right_count) / n_samples
                combined_score = \
                    left_weight * left_variance + right_weight * right_variance
                if combined_score < best_score:
                    best_score = combined_score
                    best_target = np.array(left_mask, dtype='int')
                    best_left_count = left_count
                    best_right_count = right_count 
        return best_target, best_left_count, best_right_count 
                    
    def _fit_child(self, count, X_slice, Y_slice, fit_keywords):
        if count < self.min_leaf_size:
            child = ConstantLeaf(majority (Y_slice))
        else: 
            child = _ObliqueTreeNode(
                split_classifier = self.split_classifier, 
                leaf_model = self.leaf_model, 
                num_features_per_node = self.num_features_per_node, 
                classes = self.classes, 
                depth = self.depth +1, 
                max_depth = self.max_depth, 
                min_leaf_size = self.min_leaf_size, 
                error_tol = self.error_tol, 
                randomize_split_params = self.randomize_split_params, 
                randomize_leaf_params = self.randomize_leaf_params, 
                verbose = self.verbose 
            )
            child.fit(X_slice, Y_slice, **fit_keywords)
        return child 
    
    def _fit_as_leaf(self, X, Y, fit_keywords):
        self.model = deepcopy(self.leaf_model)
        for field, gen in self.randomize_leaf_params.items():
            setattr(self.model, field,  gen())
        self.model.fit(X, Y, **fit_keywords) 

    def fit(self, X, Y, **fit_keywords):
        n_samples, n_features = X.shape
        
        if self.verbose: 
            print "Depth", self.depth, ": Fitting model for", n_samples, "vectors"
        if self.depth >= self.max_depth or n_samples <= self.min_leaf_size:
            self._fit_as_leaf(X,Y,fit_keywords)
        else:
            self.model = deepcopy(self.split_classifier)
            for field, gen in self.randomize_split_params.items():
                setattr(self.model, field,  gen())
            # if we've been passed a limit to the number of features 
            # then train the current model on a random subspace of that size
            if self.num_features_per_node:
                feature_indices = np.random.permutation(n_features)
                self.subspace  = feature_indices[:self.num_features_per_node]
                X_reduced = X[:, self.subspace]
            else:
                X_reduced = X 
            
            if self.classes:
                target, left_count, right_count = self._classifier_split(n_samples, Y)
            else:
                target, left_count, right_count = self._regression_split(n_samples, Y)
            
            if self.verbose:
                print "[oblique_tree_node] Target: Left count =", left_count, "Right count =", right_count 
            
            # if we don't have any data for one of our branches, then
            # turn this node into a leaf 
            
            if left_count == 0 or right_count == 0:
                self._fit_as_leaf(X, Y, fit_keywords)
            else:
                self.model.fit(X_reduced, target, **fit_keywords)
                pred = self.model.predict(X_reduced)
                error = np.sum( (Y - pred) ** 2)
                # if the splitter makes no mistakes, then turn this into a leaf
                if error <= self.error_tol:
                    self._fit_as_leaf(X, Y, fit_keywords)
                else: 
                    right_mask = np.array(target, dtype='bool')
                    left_mask = ~right_mask 
                    X_left = X[left_mask, :] 
                    Y_left = Y[left_mask]
                    self.left = self._fit_child(left_count, X_left, Y_left, fit_keywords)
            
                    X_right = X[right_mask, :]
                    Y_right = Y[right_mask] 
                    self.right = self._fit_child(right_count, X_right, Y_right, fit_keywords)
            
    def predict(self, X):
        nrows = X.shape[0]
        if self.subspace is not None:
            X_reduced = X[:, self.subspace] 
            pred = self.model.predict(X_reduced)
        else:
            pred = self.model.predict(X) 
            
        if self.left is None:
            return pred
        else:
            # fill this array with sub-arrays received from the predictions of children 
            outputs = np.zeros(nrows)
            left_mask = pred == 0
            X_left = X[left_mask, :] 
            if X_left.shape[0] > 0:
                outputs[left_mask] = self.left.predict(X_left)
            right_mask = ~left_mask 
            X_right = X[right_mask, :] 
            if X_right.shape[0] > 0:
                outputs[right_mask] = self.right.predict(X_right)
            
