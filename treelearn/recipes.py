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
from randomized_tree import RandomizedTree 
from oblique_tree import ObliqueTree
from classifier_ensemble import ClassifierEnsemble 
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier

def train_random_forest(X, Y, num_trees = 10, bagging_percent=0.65,  **tree_args):
    """A random forest is a bagging ensemble of randomized trees, so it can
    be implemented by combining the BaggedClassifier and RandomizedTree objects.
    This function is just a helper to your life easier.
    
    Parameters
    ----------
    X : numpy array containing input data.
        Should have samples for rows and features for columns. 
    
    Y : numpy array containing class labels for each sample
    
    num_trees : how big is the forest?
    
    bagging_percent : what subset of the data is each tree trained on?
    
    **tree_args :  parameters for individual decision tree. 
    """
    tree = RandomizedTree(**tree_args)

    forest = ClassifierEnsemble(
        base_model = tree, 
        num_models=num_trees,
        bagging_percent = bagging_percent)
    forest.fit(X,Y)
    return forest
    
def mk_svm_tree(randomize_C = True, model_args = {}, tree_args = {}):
    randomize_split_params = {}
    randomize_leaf_params = {}
    if randomize_C:
        def mk_c():
            return 10 ** (np.random.randn())
        randomize_split_params['C'] = mk_c
        randomize_leaf_params['C'] = mk_c

    split_classifier = LinearSVC(**model_args)
    leaf_classifier = LinearSVC(**model_args)
    
    tree = ObliqueTree(
        split_classifier=split_classifier, 
        leaf_model=leaf_classifier, 
        randomize_split_params = randomize_split_params,
        randomize_leaf_params = randomize_leaf_params, 
        **tree_args
    )
    return tree 

def train_svm_tree(X, Y, randomize_C = True, model_args = {}, tree_args={}):
    tree = mk_svm_tree(randomize_C, model_args, tree_args)
    tree.fit(X, Y)
    return tree 

def train_svm_forest(X, Y, num_trees = 10, bagging_percent=0.65, randomize_C = True, model_args ={}, tree_args={}):
    """A random forest whose base classifier is a SVM-Tree (rather
    than splitting individual features we project each point onto a hyperplane)
    
    Parameters
    ----------
    X : numpy array containing input data.
        Should have samples for rows and features for columns. 
    
    Y : numpy array containing class labels for each sample
    
    num_trees : how big is the forest?
    
    bagging_percent : what subset of the data is each tree trained on?
    
    randomize_C : bool 
    
    model_args : parameters for each SVM classifier 
    
    tree_args :  parameters for each tree of classifiers 
    """
    tree = mk_svm_tree(randomize_C, model_args, tree_args)
    forest = ClassifierEnsemble(
        base_model = tree, 
        num_models = num_trees,
        bagging_percent = bagging_percent)
    forest.fit(X,Y)
    return forest

def mk_sgd_tree(n_examples, randomize_alpha=True, model_args={}, tree_args={}):
    randomize_split_params = {}
    randomize_leaf_params = {}
    if randomize_alpha:
        def mk_alpha():
            return 10**(-np.random.random()*7)
        randomize_split_params['alpha'] = mk_alpha
        randomize_leaf_params['alpha'] = mk_alpha
    
    n_iter = np.ceil(10**6 / n_examples)
    split_classifier = SGDClassifier(n_iter = n_iter, **model_args)
    leaf_classifier = SGDClassifier(n_iter = n_iter)
    
    tree = ObliqueTree(
        split_classifier=split_classifier, 
        leaf_model=leaf_classifier, 
        randomize_split_params = randomize_split_params,
        randomize_leaf_params = randomize_leaf_params, 
        **tree_args
    )
    return tree 

def train_sgd_tree(X, Y, randomize_alpha=True, model_args = {}, tree_args = {}):
    tree = mk_sgd_tree(X.shape[0], randomize_alpha, model_args, tree_args)
    tree.fit(X, Y)
    return tree 
    
def train_sgd_forest(X, Y, num_trees = 10, bagging_percent=0.65, randomize_alpha=True, model_args = {}, tree_args= {}):
    """A random forest whose base classifier is a tree of SGD classifiers
    
    Parameters
    ----------
    X : numpy array containing input data.
        Should have samples for rows and features for columns. 
    
    Y : numpy array containing class labels for each sample
    
    num_trees : how big is the forest?
    
    bagging_percent : what subset of the data is each tree trained on?
    
    randomize_alpha : bool
    
    model_args : parameters for each SGD classifier 
    
    tree_args :  parameters for each tree
    """
    bagsize = bagging_percent * X.shape[0]
    tree = mk_sgd_tree(bagsize, randomize_alpha, model_args, tree_args)
    forest = ClassifierEnsemble(
        base_model = tree, 
        num_models = num_trees,
        bagging_percent = bagging_percent)
    forest.fit(X,Y)
    return forest
