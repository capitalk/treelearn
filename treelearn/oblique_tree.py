# TreeLearn
#
# Copyright (C) Capital K Partners
# Author: Alex Rubinsteyn
# Contact: alex [at] capitalkpartners [dot] com 
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.


from copy import deepcopy 
import math 
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm  import LinearSVC

from tree_helpers import majority 
from oblique_tree_node import _ObliqueTreeNode

        
class ObliqueTree(BaseEstimator):
    """A decision tree whose splits are hyperplanes. 
    Used as base learner for oblique random forests. 
    For more information, see 'On oblique random forests'. 
    http://people.csail.mit.edu/menze/papers/menze_11_oblique.pdf
    
    Parameters
    ----------------
    leaf_model : classifier or regressor. 
    
    split_classifier : classifier, optional (default = LinearSVC())
        Learning machine used to assign data points to either side of a tree
        split. 
    
    num_features_per_node : int, optional(default = sqrt of total feature count)
    
    max_depth : int, optional (default=3). 
        The number of SVMs will be at most 2^max_depth
        
    min_leaf_size : int, optional (default=100).
        Don't split data if it gets smaller than this number. 
    
    randomize_split_params : dict, optional (default={})
        Maps names of split classifier parameters to functions which generate
        random values.
        
    randomize_leaf_params : dict, optional (default={})
        Maps names of leaf model params to functions which randomly generate
        their values. 
    """
    
    def __init__(self, 
            leaf_model = LinearSVC(), 
            split_classifier = LinearSVC(), 
            num_features_per_node = None, 
            max_depth=3, 
            min_leaf_size=50, 
            randomize_split_params={}, 
            randomize_leaf_params={}, 
            verbose = False):
                
        self.leaf_model = leaf_model 
        self.split_classifier = split_classifier 
        self.max_depth = max_depth 
        self.min_leaf_size = min_leaf_size 
        self.num_features_per_node = num_features_per_node 
        
        self.randomize_split_params = randomize_split_params
        self.randomize_leaf_params = randomize_leaf_params 
        self.verbose = verbose 
        
        self.root = None
        self.classes = None
        
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
        
        self.root = _ObliqueTreeNode(
            split_classifier = self.split_classifier, 
            leaf_model = self.leaf_model, 
            num_features_per_node = num_features_per_node, 
            classes = self.classes, 
            depth = 1, 
            max_depth = self.max_depth,
            min_leaf_size = self.min_leaf_size, 
            randomize_split_params = self.randomize_split_params,
            randomize_leaf_params = self.randomize_leaf_params, 
            verbose = self.verbose 
        )
        self.root.fit(X, Y, **fit_keywords)
        
    def predict(self, X):
        return self.root.predict(X)
        
