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


import copy 
import numpy as np 
import random 
import math 

from sklearn.base import BaseEstimator 
from sklearn.svm import LinearSVC

class BaggedClassifier(BaseEstimator):
    """
        Train an ensemble of classifiers using a 
        subset of the data for each base classifier.  
        
    Parameters
    ----------
    base_classifier : Any classifier which obeys the fit/predict protocol. 
        Defaults to a Linear SVM with C = 1. 
        
    num_classifiers : int, optional (default = 50)
        How many base classifiers to train. 
    
    sample_percent : float, optional (default=0.5). 
        How much of the data set goes into each bootstrap sample. 
    
    replacement : bool, optional (default = True). 
        Sample with our without replacement. 
    """

    def __init__(self, 
            base_classifier=LinearSVC(), 
            replacement=True, 
            num_classifiers = 50, 
            sample_percent=0.5):
            #num_sample_features=None, 
            #weighted=False):
        self.classifiers = [] 
        self.classes = []
        self.replacement = replacement 
        self.base_classifier = base_classifier
        self.num_classifiers = num_classifiers
        self.sample_percent = sample_percent 
        #self.num_sample_features = num_sample_features 
        #self.weighted = weighted
    
    def fit(self, X, Y, **fit_keywords):
        self.classes = np.unique(Y) 
        self.classifiers = [] 
        
        n = X.shape[0]
        bagsize = int(math.ceil(self.sample_percent * n))
        for i in xrange(self.num_classifiers):
            if self.replacement: 
                indices = np.random.random_integers(0,n-1,bagsize)
            else:
                p = np.random.permutation(n)
                indices = p[:bagsize] 
            data_subset = X[indices, :]
            label_subset = Y[indices] 
            clf = copy.copy(self.base_classifier)
            clf.fit(data_subset, label_subset, **fit_keywords)
            self.classifiers.append(clf)

    def predict(self, X, return_probs=False):
        """Every classifier in the ensemble votes for a class. Return the class
           which got a majority vote and optionally also return the full 
           n_samples * n_classes probability matrix."""
           
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape    
        class_list = list(self.classes)
        n_classes = len(self.classes)
        votes = np.zeros( [n_samples, n_classes] )
        
        for clf in self.classifiers:
            ys = clf.predict(X)
            weight = 1  # possibly change this later to have varying weights
            for c in self.classes:
                class_index = class_list.index(c)
                votes[ys == c, class_index] += weight
                
        majority_indices = np.argmax(votes, axis=1)        
        majority_labels = np.array([class_list[i] for i in majority_indices])
        if return_probs:
            sums = np.sum(votes, axis=1)
            probs = votes / np.array([sums], dtype='float').T
            return majority_labels, probs
        else:
            return majority_labels 
            
