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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score


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
    
    sample_replacement : bool, optional (default = True). 
        Sample with our without replacement. 
    
    weighting : None or float, optional (default=None). 
        Weight individual classifiers in the ensemble by 
            None : all classifiers given equal weight
            <beta> : compute F_beta score for each classifier. 
                     Only works for binary classification.
    
    stacking : bool, optional (default=False).
        Feed output of weighted individual predictions into 
        a logistic regressor.
    
    verbose : bool, optional (default = False).
        Print diagnostic output. 
    """

    def __init__(self, 
            base_classifier=LinearSVC(), 
            num_classifiers = 50, 
            sample_percent=0.5, 
            sample_replacement=True, 
            weighting=None, 
            stacking=False,
            verbose=False):
                
        self.base_classifier = base_classifier
        self.num_classifiers = num_classifiers
        self.sample_percent = sample_percent 
        self.sample_replacement = sample_replacement 
        self.weighting = weighting
        self.stacking = stacking 
        self.verbose = verbose 
        
        self.need_to_fit = True
        self.weights = None 
        self.classifiers = None
        self.classes = None
        self.class_list = None 
        self.stacked_classifier = None 
    
    def fit(self, X, Y, **fit_keywords):
        self.need_to_fit = False 
        self.classes = np.unique(Y) 
        self.class_list = list(self.classes)
        self.classifiers = [] 
        
        n = X.shape[0]
        bagsize = int(math.ceil(self.sample_percent * n))
        # initialize weights to be uniform, change if some other weighting
        # style required 
        self.weights = np.ones(self.num_classifiers, dtype='float')
        
        # if user wants F-score weighting, use their given value of beta
        beta = self.weighting
            
        for i in xrange(self.num_classifiers):
            if self.verbose:
                print "Training iteration", i 
                
            if self.sample_replacement: 
                indices = np.random.random_integers(0,n-1,bagsize)
            else:
                p = np.random.permutation(n)
                indices = p[:bagsize] 
                
            data_subset = X[indices, :]
            label_subset = Y[indices] 
            clf = copy.copy(self.base_classifier)
            clf.fit(data_subset, label_subset, **fit_keywords)
            self.classifiers.append(clf)
            
            # to assign an F-score weight to each classifier, 
            # sample another subset of the data and use the model 
            # we just train to generate predictions 
            if beta or self.verbose:
                error_sample_indices = np.random.random_integers(0,n-1,bagsize)
                error_subset = X[error_sample_indices, :] 
                error_labels = Y[error_sample_indices]
                y_pred = clf.predict(error_subset)
                if beta: 
                    f_score = fbeta_score(error_labels, y_pred, beta)
                    self.weights[i] = f_score 
                if self.verbose:
                    print "Actual non-zero:", np.sum(error_labels != 0)
                    pred_nnz = np.sum(y_pred != 0)
                    print "Predicted non-zero:", pred_nnz
                    y_pos = error_labels == 1
                    pred_pos = y_pred == 1
                    tp = np.sum(y_pos & pred_pos)
                    print "True positives:", tp 
                    print "False positives:", pred_nnz - tp 
                    
                    
        # stacking works by treating the outputs of each base classifier as the 
        # inputs to an additional meta-classifier
        if self.stacking:
            self.stacked_classifier = LogisticRegression()
            transformed_data = self._predict_normalized_votes(X)
            self.stacked_classifier.fit(transformed_data, Y)
            

    def _predict_votes(self, X): 
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape    
        n_classes = len(self.classes)
        votes = np.zeros( [n_samples, n_classes] )
        
        for weight, clf in zip(self.weights, self.classifiers):
            ys = clf.predict(X)
            for c in self.classes:
                class_index = self.class_list.index(c)
                votes[ys == c, class_index] += weight
        return votes
    
    def _predict_normalized_votes(self, X): 
        votes = self._predict_votes(X)
        sums = np.sum(votes, axis=1)
        return votes / np.array([sums], dtype='float').T
    
    def _predict_stacked_probs(self, X):
        votes = self._predict_votes(X)
        return self.stacked_classifier.predict_proba(votes)

    def predict_proba(self, X):
        if self.need_to_fit:
            raise RuntimeError("Trying to call 'predict_proba' before 'fit'")
        ps = self._predict_normalized_votes(X)
        if self.stacking:
            return self.stacked_classifier.predict_proba(ps)
        else:
            return ps

    def predict(self, X, return_probs=False):
        """Every classifier in the ensemble votes for a class. 
           If we're doing stacking, then pass the votes as features into 
           the stacking classifier, otherwise return the majority vote."""
        if self.need_to_fit:
            raise RuntimeError("Trying to call 'predict' before 'fit'")
        
        if self.stacking:
            majority_indices = np.argmax(self._predict_stacked_probs(X), axis=1)
        else: 
            majority_indices = np.argmax(self._predict_votes(X), axis=1)        
        return np.array([self.class_list[i] for i in majority_indices])
            
