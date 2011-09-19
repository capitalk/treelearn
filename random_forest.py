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

import numpy as np 
import random 
import math 

from tree_helpers import * 
import randomized_tree as tree 
    
class RandomForest:
    def __init__(self, classes = None, bag_percent=0.7, ntree = 50, **keywords):
        self.classes = None
        self.trees = [] 
        self.bag_percent = bag_percent
        self.ntrees = ntrees 
        self.tree_params = keywords 
    
    def __str__(self): 
        treeStrings = ["tree " + str(i) + ": " + str(t) for i,t in enumerate(self.trees)]
        return "[RandomForest]\n" + "\n".join(treeStrings)
        
    def fit(self, X,Y):
        if self.classes is None: 
            classes = np.unique(Y) 
            self.classes = classes 
        else: classes = self.classes 
        
        self.trees = [] 
            
        n = X.shape[0]
        bagsize = int(self.bag_percent * n)
        permute = np.random.permutation
        
        for i in xrange(self.ntrees):
            p = permute(n)
            indices = p[:bagsize] 
            data_bag = X[indices, :]
            label_bag = Y[indices] 
            
            t = tree.RandomizedTree(classes=classes, **self.tree_params)
            t.fit(data_bag,label_bag)
            self.trees.append(t)
            
    def predict(self, X):
        votes = [t.predict(X) for t in self.trees]
        return majority(self.classes, votes)
        
