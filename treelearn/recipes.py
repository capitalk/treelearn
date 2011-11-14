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

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model import LinearRegression, Ridge 

from regression_ensemble import RegressionEnsemble
from classifier_ensemble import ClassifierEnsemble 
from clustered_regression import ClusteredRegression 
from clustered_classifier import ClusteredClassifier
from randomized_tree import RandomizedTree 
from oblique_tree import ObliqueTree


def train_random_forest(
        X, 
        Y, 
        num_trees = 20, 
        max_thresholds = 10, 
        max_height = None, 
        min_leaf_size = None, 
        bagging_percent=0.65):
    """A random forest is a bagging ensemble of randomized trees, so it can
    be implemented by combining the BaggedClassifier and RandomizedTree objects.
    This function is just a helper to your life easier.
    
    Parameters
    ----------
    X : numpy array containing input data.
        Should have samples for rows and features for columns. 
    
    Y : numpy array containing class labels for each sample
    
    num_trees : how big is the forest?
    
    max_thresholds : rather than evaluating all possible thresholds at each 
        split, randomly sample this number of thresholds
    
    max_height : don't let tree grow past given height, inferred if omitted. 
    
    min_leaf_size : don't split nodes smaller than this, inferred if omitted. 
    
    bagging_percent : what subset of the data is each tree trained on?
    
    **tree_args :  parameters for individual decision tree. 
    """
    if isinstance(Y[0], float):
        regression = True
    else: 
        regression = False 
        
    if max_height is None: 
        max_height = int(np.log2(X.shape[0])) + 1
    if min_leaf_size is None: 
        min_leaf_size = int(np.log2(X.shape[0])) 
        
    tree = RandomizedTree(
        regression = regression, 
        max_thresholds = max_thresholds, 
        max_height = max_height, 
        min_leaf_size = min_leaf_size, 
    )

    if regression:
        forest = RegressionEnsemble(
            base_model = tree, 
            num_models= num_trees,
            bagging_percent = bagging_percent
        )
    else: 
        forest = ClassifierEnsemble(
            base_model = tree, 
            num_models = num_trees, 
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
        **tree_args)
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
    
def train_sgd_forest(X, Y, 
        num_trees = 20, 
        max_depth = 3, 
        bagging_percent=0.65, 
        randomize_alpha=False, 
        model_args = {}, 
        tree_args= {}):
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

def train_clustered_ols(X, Y, k = 20): 
    """Cluster data and then train a linear regressor per cluster"""
    cr = ClusteredRegression(k)
    cr.fit(X, Y)
    return cr 

def train_clustered_svm(X, Y, k = 20, C = 1, verbose = True):
    base_model = LinearSVC(C = C)
    cc = ClusteredClassifier(k = k, base_model = base_model, verbose = verbose)
    cc.fit(X, Y)
    return cc 

def mk_clustered_svm_ensemble(
        num_models = 20, 
        C = 1, 
        k = 20, 
        stacking= False, 
        bagging_percent = 0.65, 
        feature_subset_percent=0.5, 
        verbose = True): 
    
    base_model = LinearSVC(C = C)
    clustered_model = ClusteredClassifier(k, base_model = base_model, verbose=verbose)
    
    if stacking:
        stacking_model = LogisticRegression(fit_intercept=False)
    else:
        stacking_model = None 
        
    return ClassifierEnsemble(
        base_model = clustered_model, 
        num_models = num_models, 
        bagging_percent = bagging_percent, 
        feature_subset_percent = feature_subset_percent, 
        stacking_model = stacking_model)

def train_clustered_svm_ensemble(
        X, 
        Y, 
        num_models = 10, 
        C = 1, 
        k =  20, 
        stacking= False, 
        bagging_percent = 0.65, 
        feature_subset_percent=0.5, 
        verbose = True): 
    ensemble = mk_clustered_svm_ensemble(
                num_models, 
                C, 
                k, 
                stacking, 
                bagging_percent, 
                feature_subset_percent, 
                verbose)
    ensemble.fit(X, Y)
    return ensemble 

def mk_clustered_regression_ensemble(
            num_models = 20, 
            k = 20,
            stacking= False, 
            additive=False, 
            bagging_percent = 0.65, 
            feature_subset_percent=0.5):


    clustered_model = ClusteredRegression(k=k, base_model = LinearRegression())
    
    if stacking:
        stacking_model = LinearRegression(fit_intercept=False)
    else:
        stacking_model = None 
        
    return RegressionEnsemble(
        base_model = clustered_model, 
        num_models = num_models, 
        bagging_percent = bagging_percent, 
        feature_subset_percent = feature_subset_percent, 
        stacking_model = stacking_model, 
        additive = additive 
    )
    
def train_clustered_regression_ensemble(
        X, 
        Y, 
        num_models=10, 
        k = 20,
        stacking=False, 
        additive=False, 
        bagging_percent = 0.65, 
        feature_subset_percent=0.5):
    ensemble = mk_clustered_regression_ensemble (
        num_models = num_models, 
        k = k, 
        stacking = stacking, 
        additive = additive, 
        bagging_percent = bagging_percent, 
        feature_subset_percent = feature_subset_percent 
    )
    ensemble.fit(X, Y)
    return ensemble 

def mk_additive_regression_forest(
        num_trees=50, 
        bagging_percent = 0.65, 
        feature_subset_percent = 0.5, 
        max_height=3, 
        min_leaf_size=10, 
        max_thresholds=100):
    tree = RandomizedTree(
        max_height= max_height, 
        min_leaf_size=min_leaf_size, 
        max_thresholds=max_thresholds, 
        regression=True)
    forest = RegressionEnsemble(
        base_model = tree, 
        num_models=num_trees,
        bagging_percent = bagging_percent, 
        feature_subset_percent = feature_subset_percent, 
        additive=True)
    return forest 
    
def train_additive_regression_forest(X, Y, 
        num_trees=50, 
        bagging_percent = 0.65, 
        feature_subset_percent = 0.5, 
        max_height=3, 
        min_leaf_size=10, 
        max_thresholds=50):
    forest = mk_additive_regression_forest(
        num_trees, 
        bagging_percent, 
        feature_subset_percent, 
        max_height, 
        min_leaf_size, 
        max_thresholds)
    forest.fit(X,Y)
    return forest 
    
