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
from regression_ensemble import RegressionEnsemble
from cluster_regression import ClusterRegression 
from randomized_tree import RandomizedTree 
from oblique_tree import ObliqueTree
from classifier_ensemble import ClassifierEnsemble 
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression, Ridge 

def train_random_forest(X, Y, num_trees = 10, max_thresholds = 10, bagging_percent=0.65,  **tree_args):
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
    if isinstance(Y[0], float):
        regression = True
        ensemble_class = RegressionEnsemble 
    else: 
        regression = False 
        ensemble_class = ClassifierEnsemble 
    tree = RandomizedTree(regression = regression, max_thresholds = max_thresholds, **tree_args)

    
    forest = ensemble_class (
        base_model = tree, 
        num_models=num_trees,
        bagging_percent = bagging_percent
    )
    forest.fit(X,Y)
    return forest
    
def gen_random_C():
    return 10 ** (np.random.randn())
        
def mk_svm_tree(max_depth = 3, randomize_C = False, model_args = {}, tree_args = {}):
    randomize_split_params = {}
    randomize_leaf_params = {}
    if randomize_C:
        randomize_split_params['C'] = gen_random_C
        randomize_leaf_params['C'] = gen_random_C

    split_classifier = LinearSVC(**model_args)
    leaf_classifier = LinearSVC(**model_args)
    
    tree = ObliqueTree(
        max_depth = max_depth, 
        split_classifier=split_classifier, 
        leaf_model=leaf_classifier, 
        randomize_split_params = randomize_split_params,
        randomize_leaf_params = randomize_leaf_params, 
        **tree_args
    )
    return tree 

def train_svm_tree(X, Y, max_depth = 3, randomize_C = False, model_args = {}, tree_args={}):
    tree = mk_svm_tree(max_depth, randomize_C, model_args, tree_args)
    tree.fit(X, Y)
    return tree 

def train_svm_forest(X, Y, num_trees = 10, max_depth = 3, bagging_percent=0.65, randomize_C = False, model_args ={}, tree_args={}):
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
    tree = mk_svm_tree(max_depth, randomize_C, model_args, tree_args)
    forest = ClassifierEnsemble(
        base_model = tree, 
        num_models = num_trees,
        bagging_percent = bagging_percent)
    forest.fit(X,Y)
    return forest

def gen_random_alpha():
    return 10**(-np.random.random()*7)

def mk_sgd_tree(n_examples=200000, max_depth=3, randomize_alpha=False, model_args={}, tree_args={}):
    randomize_split_params = {}
    randomize_leaf_params = {}
    if randomize_alpha:
        randomize_split_params['alpha'] = gen_random_alpha
        randomize_leaf_params['alpha'] = gen_random_alpha
    
    n_iter = np.ceil(10**6 / n_examples)
    split_classifier = SGDClassifier(n_iter = n_iter, shuffle=True, **model_args)
    leaf_classifier = SGDClassifier(n_iter = n_iter, shuffle=True, **model_args)
    
    tree = ObliqueTree(
        max_depth = max_depth, 
        split_classifier=split_classifier, 
        leaf_model=leaf_classifier, 
        randomize_split_params = randomize_split_params,
        randomize_leaf_params = randomize_leaf_params, 
        **tree_args
    )
    return tree 

def train_sgd_tree(X, Y, max_depth=3, randomize_alpha=False, model_args = {}, tree_args={}):
    tree = mk_sgd_tree(X.shape[0], max_depth, randomize_alpha, model_args, tree_args)
    tree.fit(X, Y)
    return tree 
    
def train_sgd_forest(X, Y, num_trees = 10, max_depth = 3, bagging_percent=0.65, randomize_alpha=False, model_args = {}, tree_args= {}):
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
    tree = mk_sgd_tree(bagsize, max_depth, randomize_alpha, model_args, tree_args)
    forest = ClassifierEnsemble(
        base_model = tree, 
        num_models = num_trees,
        bagging_percent = bagging_percent)
    forest.fit(X,Y)
    return forest

def train_clustered_ols(X, Y, k = 10): 
    """Cluster data and then train a linear regressor per cluster"""
    cr = ClusterRegression(k)
    cr.fit(X, Y)
    return cr 

def mk_regression_ensemble(lowest_k = 3, highest_k = 50, num_models = 10, stacking= False, additive=False, bagging_percent = 0.65, feature_subset_percent=0.5): 
    if stacking:
        stacking_model = LinearRegression(fit_intercept=False)
    else:
        stacking_model = None 
    def random_k():
        return np.random.randint(lowest_k, highest_k+1)
    return RegressionEnsemble(
        base_model = ClusterRegression(highest_k), 
        num_models = num_models, 
        bagging_percent = bagging_percent, 
        feature_subset_percent = feature_subset_percent, 
        stacking_model = stacking_model, 
        randomize_params = {'k': random_k}, 
        additive = additive 
    )
    
def train_regression_ensemble(X, Y, lowest_k = 2, highest_k = 50, num_models=10, stacking=False, additive=False, bagging_percent = 0.65, feature_subset_percent=0.5):
    ensemble = mk_regression_ensemble (
        lowest_k = lowest_k, 
        highest_k = highest_k, 
        num_models = num_models, 
        stacking = stacking, 
        additive = additive, 
        bagging_percent = bagging_percent, 
        feature_subset_percent = feature_subset_percent 
    )
    ensemble.fit(X, Y)
    return ensemble 

def mk_additive_regression_forest(num_trees=50, bagging_percent = 0.65, feature_subset_percent = 0.5, max_height=3, min_leaf_size=10, max_thresholds=50):
    tree = RandomizedTree(max_height= max_height, min_leaf_size=min_leaf_size, max_thresholds=max_thresholds, regression=True)
    forest = RegressionEnsemble(
        base_model = tree, 
        num_models=num_trees,
        bagging_percent = bagging_percent, 
        feature_subset_percent = feature_subset_percent, 
        additive=True)
    return forest 
    
def train_additive_regression_forest(X, Y, num_trees=50, bagging_percent = 0.65, feature_subset_percent = 0.5, max_height=3, min_leaf_size=10, max_thresholds=50):
    forest = mk_additive_regression_forest(num_trees, bagging_percent, feature_subset_percent, max_height, min_leaf_size, max_thresholds)
    forest.fit(X,Y)
    return forest 
    
