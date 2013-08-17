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


from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

class ViterbiTreeNode(BaseEstimator):
    def __init__(self, depth, max_depth, num_retries, leaf_model):
        self.depth = depth
        self.max_depth = max_depth 
        self.is_leaf = (depth == max_depth)
        self.num_retries = num_retries 
        self.C = C
        if depth == max_depth:
            self.leaf_model = leaf_model 
        else:
            self.left = ViterbiTreeNode(depth+1, max_depth, num_retries, leaf_model)
            self.right = ViterbiTreeNode(depth+1, max_depth, num_retries, leaf_model)
            
    def gen_random_cs(self):
        return 10 ** (np.random.randn(self.num_retries) - 1)
    
    def init_fit(self, X,Y):
        """Initialize partitions and leaf models to minimize training error"""
        best_model = None
        best_error = np.inf  
        
        for c in self.gen_random_cs():
            if self.is_leaf: 
                model = self.leaf_model(C=c)
            else:
                model = LinearSVC(C=c)
                
            model.fit(X,Y)
            error = model.score(X,Y)
            if err < best_error:
                best_model = model
                best_error = error
        self.model = best_model 
        if not self.is_leaf:
            pred = model.predict(X)
            mask = (pred != 1)
            self.left.init_fit(X[mask, :], Y[mask])
            self.right.init_fit(X[~mask, :], Y[~mask])
        
    def refit_partition(X,partition,Y):
        """Assumes that 'init_fit' has already been run."""
        if self.is_leaf:
            self.model.fit(X,Y)
        else:
            nrows = X.shape[0]
            # get probabilities of y=1
            left_prob = self.left.predict_proba(X)[:, 1]
            right_prob = self.right.predict_proba(X)[:, 1]
            assignments = np.zeros(nrows)
            right_mask = (left_prob < right_prob) & Y == 1
            
            # TODO
            # assignments[]
    def refit_leaves(X,Y):
        # TODO
        pass
        
    def predict(X):
        # TODO
        pass
        
class ViterbiTree(BaseEstimator):
    def __init__(self, max_depth=3, num_retries = 3, leaf_model=LogisticRegression):
        self.root = ViterbiTreeNode(1, max_depth, num_retries, leaf_model)
    
    def fit(self, X, Y):
        self.root.init_fit(X,Y)

    def predict(self, X)
        return self.root.predict(X)
