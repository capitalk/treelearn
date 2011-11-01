import numpy as np 
import scipy 
import scipy.weave 
import scipy.stats 
import random 
import math 

from sklearn.base import BaseEstimator 
from constant_leaf import ConstantLeaf
from tree_node import TreeNode 
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
    """

    def __init__(self, 
            num_features_per_node=None, 
            min_leaf_size=15, 
            max_height = 100, 
            max_thresholds=None, 
            classes = None, 
            feature_names = None):
        self.root = None 
        self.num_features_per_node = num_features_per_node 
        self.min_leaf_size = min_leaf_size
        self.max_height = max_height 
        if classes is None: 
            self.classes = None
            self.nclasses = 0 
        else: 
            self.classes = np.asarray(classes)
            self.nclasses = len(classes) 
        self.feature_names = feature_names 
        self.max_thresholds = max_thresholds 
        if max_thresholds is None:
            self.get_thresholds = self.all_thresholds
        else:
            self.get_thresholds = self.threshold_subset 


    
    def all_thresholds(self, x): 
        """get midpoints between all unique values"""
        if len(x) > 1: 
            return midpoints(np.unique(x))
        else: 
            return x 
    
    def threshold_subset(self, x):
        """return a set of thresholds smaller in size than the actual number
           of unique values"""
        unique_vals = np.unique(x)
        num_unique_vals = len(unique_vals)
        k = self.max_thresholds
        if num_unique_vals <= k: return self.all_thresholds(unique_vals)
        else:
            mid = unique_vals[num_unique_vals/2] 
            half_k =(k+1)/2
            lower = np.linspace(unique_vals[0], mid, half_k, endpoint=False)
            upper = np.linspace(mid, unique_vals[-1], half_k)
            return np.concatenate( (lower[1:], upper))
            
            
    def split(self, data, labels, m, height):
        nfeatures = data.shape[1]
        # randomly draw m feature indices. 
        # should be more efficient than explicitly constructing a permutation
        # vector and then keeping only the head elements 
        random_feature_indices = random.sample(xrange(nfeatures), m)
        best_split_score = np.inf
        best_feature_idx = None
        best_thresh = None 
        best_left_indicator = None 
        best_right_indicator = None 
        classes = self.classes
        
        for feature_idx in random_feature_indices:
            feature_vec = data[:, feature_idx]
            thresholds = self.get_thresholds(feature_vec)
            for thresh in thresholds:
                left_indicator = feature_vec < thresh
                right_indicator = ~left_indicator
                
                left_labels = labels[left_indicator] 
                right_labels = labels[right_indicator] 
            
                combined_score = eval_gini_split(classes, left_labels, right_labels)
                if combined_score < best_split_score:
                    best_split_score = combined_score
                    best_feature_idx = feature_idx
                    best_thresh = thresh 
                    best_left_indicator = left_indicator
                    best_right_indicator = right_indicator 
    
        left_data = data[best_left_indicator, :] 
        left_labels = labels[best_left_indicator] 
        left_node = self.mk_node(left_data, left_labels, m, height+1)
        right_data = data[best_right_indicator, :] 
        right_labels = labels[best_right_indicator]
        right_node = self.mk_node (right_data, right_labels, m, height+1)
        node = TreeNode(best_feature_idx, best_thresh, left_node, right_node)
        return node 

    def mk_node(self, data, labels, m, height):
        # if labels are all same 
        if len(labels) <= self.min_leaf_size or height > self.max_height:
            self.nleaves += 1
            return ConstantLeaf(majority(labels, self.classes))
            
        elif np.all(labels == labels[0]):
            self.nleaves += 1
            return ConstantLeaf(labels[0])
        else:
            return self.split(data, labels, m, height)
                
    def fit(self, data, labels, feature_names = None): 
        data = np.atleast_2d(data)
        labels = np.atleast_1d(labels)
        
        if self.classes is None: 
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
        self.root = self.mk_node(data, labels, m, 1)

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        # create an output array and let the tree nodes recursively fill it
        outputs = np.zeros(n_samples)
        mask = np.ones(n_samples, dtype='bool')
        self.root.fill_predict(X, outputs, mask)
        return outputs 
