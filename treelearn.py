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

import bagging
from randomized_tree import RandomizedTree 
from svm_tree import SVM_Tree

def train_random_forest(X, Y, num_trees = 50, sample_percent=0.65,  **tree_args):
    """A random forest is a bagging ensemble of randomized trees, so it can
    be implemented by combining the BaggedClassifier and RandomizedTree objects.
    This function is just a helper to your life easier.
    
    Parameters
    ----------
    X : numpy array containing input data.
        Should have samples for rows and features for columns. 
    
    Y : numpy array containing class labels for each sample
    
    num_trees : how big is the forest?
    
    sample_percent : what subset of the data is each tree trained on?
    
    **tree_args :  parameters for individual decision tree. 
    """
    tree = RandomizedTree(**tree_args)

    forest = bagging.BaggedClassifier(
        base_classifier = tree, 
        num_classifiers=num_trees,
        sample_percent = sample_percent)
    forest.fit(X,Y)
    return forest
    


def train_svm_forest(X, Y, num_trees = 50, sample_percent=0.65, **tree_args):
    """A random forest whose base classifier is a SVM-Tree (rather
    than splitting individual features we project each point onto a hyperplane)
    
    Parameters
    ----------
    X : numpy array containing input data.
        Should have samples for rows and features for columns. 
    
    Y : numpy array containing class labels for each sample
    
    num_trees : how big is the forest?
    
    sample_percent : what subset of the data is each tree trained on?
    
    **tree_args :  parameters for individual svm decision tree
    """
    tree = SVM_Tree(**tree_args)

    forest = bagging.BaggedClassifier(
        base_classifier = tree, 
        num_classifiers=num_trees,
        sample_percent = sample_percent)
    forest.fit(X,Y)
    return forest
