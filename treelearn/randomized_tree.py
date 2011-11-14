import numpy as np 
import scipy 
import scipy.weave 
import scipy.stats 
import random 
import math 

from sklearn.base import BaseEstimator 
from constant_leaf import ConstantLeaf
from tree_node import TreeNode 
import random 
from tree_helpers import * 

class RandomizedTree(BaseEstimator):
    """Decision tree which only inspects a random subset of the features
       at each split. Uses Gini impurity to compare possible data splits. 

    Parameters
    ----------
    num_features_per_node : int, optional (default = None).
        At each split, how many features should we consider splitting. 
        If None, then use log(total number of features). 

    min_leaf_size : int, optional (default=15). 
        Stop splitting when the data gets this small. 
    
    max_height : int, optional (default = 100). 
        Stop growing tree at this height. 
    
    max_thresholds : int, optional (default = None). 
        At each split, generate at most this number of evenly spaced thresholds
        between the min and max feature values. The default behavior is
        to consider all midpoints between unique feature values. 
    
    classes : sequence of int labels, optional (default = None). 
        If None, then use the unique values of the classes given to 'fit'. 
    
    feature_names : string list (default = None). 
        Names to use for pretty printing. 
        
    verbose : bool (default = False).
        Print debugging output. 
    """

    def __init__(self, 
            num_features_per_node=None, 
            min_leaf_size=10, 
            max_height = 20, 
            max_thresholds=None, 
            regression =  False, 
            feature_names = None, 
            verbose = False):
        self.root = None 
        self.num_features_per_node = num_features_per_node 
        self.min_leaf_size = min_leaf_size
        self.max_height = max_height 
        self.classes = None 
        self.feature_names = feature_names 
        self.max_thresholds = max_thresholds 
        if max_thresholds is None:
            self.get_thresholds = self.all_thresholds
        else:
            self.get_thresholds = self.random_threshold_subset 
        self.regression = regression 
        if regression: 
            self.leaf_dtype = 'float' 
        else:
            self.leaf_dtype = 'int'
            
        self.verbose = verbose 

    
    def all_thresholds(self, x): 
        """get midpoints between all unique values"""
        if len(x) > 1: 
            return midpoints(np.unique(x))
        else: 
            return x 
    
    def random_threshold_subset(self, x): 
        total = len(x)
        k = self.max_thresholds 
        nsamples = min(total, k)
        rand_subset = random.sample(x, nsamples)
        return self.all_thresholds(rand_subset)
    
    def _split(self, data, labels, m, height):
        n_samples = data.shape[0]
        if n_samples <= self.min_leaf_size or height > self.max_height:
            self.nleaves += 1
            if self.regression:
                return ConstantLeaf(np.mean(labels))
            else: 
                return ConstantLeaf(majority(labels, self.classes))
        elif np.all(labels == labels[0]):
            self.nleaves += 1
            return ConstantLeaf(labels[0])
        else:
            nfeatures = data.shape[1]
            # randomly draw m feature indices. 
            # should be more efficient than explicitly constructing a permutation
            # vector and then keeping only the head elements 
            random_feature_indices = random.sample(xrange(nfeatures), m)
            best_split_score = np.inf
            best_feature_idx = None
            best_thresh = None 
            
            for feature_idx in random_feature_indices:
                feature_vec = data[:, feature_idx]
                thresholds = self.get_thresholds(feature_vec)
                
                
                if self.regression:
                    thresh, combined_score = \
                        find_min_variance_split(feature_vec, thresholds, labels)
                else:
                    thresh, combined_score = \
                        find_best_gini_split(self.classes, feature_vec, thresholds, labels)
                
                if combined_score < best_split_score:
                    best_split_score = combined_score
                    best_feature_idx = feature_idx
                    best_thresh = thresh 
                    
            left_mask = data[:, best_feature_idx] < best_thresh 
            right_mask = ~left_mask
            
            left_data = data[left_mask, :] 
            right_data = data[right_mask, :] 
            
            left_labels = labels[left_mask] 
            right_labels = labels[right_mask]
            
            # get rid of references before recursion so data can be deleted
            del labels 
            del data 
            del random_feature_indices 
            del left_mask 
            del right_mask 
            
            left_node = self._split(left_data, left_labels, m, height+1)
            right_node = self._split(right_data, right_labels, m, height+1)
            node = TreeNode(best_feature_idx, best_thresh, left_node, right_node)
            return node 

                
    def fit(self, data, labels, feature_names = None): 
        data = np.atleast_2d(data)
        labels = np.atleast_1d(labels)
        
        if not self.regression: 
            self.classes = np.unique(labels)
            self.nclasses = len(self.classes)
        self.feature_names = feature_names 
        self.nleaves = 0 
        nrows = data.shape[0]
        nfeatures = data.shape[1]
        if self.num_features_per_node is None:
            m = int(round(math.log(nfeatures, 2)))
        else:
            m = self.num_features_per_node 
        self.root = self._split(data, labels, m, 1)

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        # create an output array and let the tree nodes recursively fill it
        outputs = np.zeros(n_samples, dtype=self.leaf_dtype)
        mask = np.ones(n_samples, dtype='bool')
        self.root.fill_predict(X, outputs, mask)
        return outputs 
