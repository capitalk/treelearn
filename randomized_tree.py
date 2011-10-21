import numpy as np 
import scipy 
import scipy.weave 
import scipy.stats 
import random 
import math 

from tree_helpers import * 

class ConstantLeaf:
    def __init__(self, v):
        self.v = v
    
    def to_str(self, indent="", feature_names=None):
        return indent + "Constant(" + str(self.v) + ")"
    
    def __str__(self): 
        return self.to_str() 
        
    def predict(self, data):
        return self.v 

class TreeNode:
    def __init__(self, feature_idx, split_val, left, right):
        self.feature_idx = feature_idx
        self.split_val = split_val 
        self.left = left
        self.right = right
        
    def predict(self, data):
        x = data[self.feature_idx] 
        if x < self.split_val:
            return self.left.predict(data)
        else:
            return self.right.predict(data) 
    
    def to_str(self, indent="", feature_names=None):
        if feature_names:
            featureStr = feature_names[feature_idx]
        else:
            featureStr = "x[" + str(self.feature_idx) + "]"
        longer_indent = indent + "  " 
        left = self.left.to_str(indent = longer_indent)
        right = self.right.to_str(indent = longer_indent)
        cond = "if %s < %f:" % (featureStr, self.split_val)
        return indent + cond + "\n" +  left + "\n" + indent + "else:\n" + right
        
    def __str__(self, prefix=""):
        return self.to_str()
        
    
class RandomizedTree:
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
    
    classes : int list, optional (default = None). 
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
        self.classes = classes 
        self.nclasses = len(classes) if classes is not None else 0
        self.feature_names = feature_names 
        if max_thresholds is None:
            self.get_thresholds = self.all_thresholds
        else:
            self.max_thresholds = max_thresholds 
            self.get_thresholds = self.threshold_subset 

    def __str__(self):
        return self.root.to_str(feature_names = self.feature_names)
    
    def __repr__(self):
        return str(self)
    
    def threshold_subset(self, x):
        unique_vals = np.unique(x)
        num_unique_vals = len(unique_vals)
        k = self.max_thresholds
        if num_unique_vals <= k: return midpoints(unique_vals)
        else:
            mid = unique_vals[num_unique_vals/2] 
            half_k =(k+1)/2
            lower = np.linspace(unique_vals[0], mid, half_k, endpoint=False)
            upper = np.linspace(mid, unique_vals[-1], half_k)
            return np.concatenate( (lower[1:], upper))
            
    
    def all_thresholds(self, x): 
        """get midpoints between all unique values"""
        return midpoints(np.unique(x))
            
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
            return ConstantLeaf(majority(self.classes, labels))
            
        elif np.all(labels == labels[0]):
            self.nleaves += 1
            return ConstantLeaf(labels[0])
        else:
            return self.split(data, labels, m, height)
                
    def fit(self, data, labels, feature_names = None): 
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

    def predict(self, vec):
        return self.root.predict(vec) 
