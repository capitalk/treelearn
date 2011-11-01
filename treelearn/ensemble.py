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

class BaseEnsemble(BaseEstimator):
    def __init__(self):
        self.need_to_fit = True
        self.bagging_percent = None 
        self.bagging_replacement = None 
        self.base_model = None
        self.num_models = None 
        self.verbose = None 
        
    def fit(self, X, Y, **fit_keywords):
        assert self.base_model is not None
        assert self.bagging_percent is not None 
        assert self.bagging_replacement is not None 
        assert self.num_models is not None 
        assert self.verbose is not None
        
        self.need_to_fit = False 
        self.models = [] 
        
        n = X.shape[0]
        bagsize = int(math.ceil(self.bagging_percent * n))
        # initialize weights to be uniform, change if some other weighting
        # style required 
        self.weights = np.ones(self.num_models, dtype='float')
        
        # each derived class needs to implement this 
        self._init_fit(X,Y)
        
        for i in xrange(self.num_models):
            if self.verbose:
                print "Training iteration", i 
                
            if self.bagging_replacement: 
                indices = np.random.random_integers(0,n-1,bagsize)
            else:
                p = np.random.permutation(n)
                indices = p[:bagsize] 
                
            data_subset = X[indices, :]
            label_subset = Y[indices] 
            model = copy.copy(self.base_model)
            model.fit(data_subset, label_subset, **fit_keywords)
            self.models.append(model)
            
            self._created_model(X, Y, indices, model)
        
        self.weights /= np.sum(self.weights)
        
        # stacking works by treating the outputs of each base classifier as the 
        # inputs to an additional meta-classifier
        if self.stacking_model:
            transformed_data = self.weighted_transform(X)
            self.stacking_model.fit(transformed_data, Y)
        
    
    """Can't be instantiated, since member fields like 'models' and 'weights'
    have to be created by descendant classes"""
    def transform(self, X):
        """Convert each feature vector into a row of predictions."""
        assert self.weights is not None
        assert self.models is not None 
        
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape    
        n_models = len(self.models)
        pred = np.zeros([n_samples, n_models])
        for i, model in enumerate(self.models):
            pred[:, i] = model.predict(X)
        return pred
    
    def weighted_transform(self, X):
        
        """Output of each model, multiplied by that model's weight. A weighted
        mean can be recovered by summing across the columns of the result."""
        pred = self.transform(X)
        for i, weight in enumerate(self.weights):
            pred[:, i] *= weight 
        return pred
        
            
class ClassifierEnsemble(BaseEnsemble):
    """
        Train an ensemble of classifiers using a 
        subset of the data for each base classifier.  
        
    Parameters
    ----------
    base_model : Any classifier which obeys the fit/predict protocol. 
        Defaults to a Linear SVM with C = 1. 
        
    num_models : int, optional (default = 50)
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
    
    stacking : classifier, optional (default=None).
        Feed output of weighted individual predictions into another classifier. 
        Suggested model: LogisticRegression. 
        
    
    verbose : bool, optional (default = False).
        Print diagnostic output. 
    """

    def __init__(self, 
            base_model=LinearSVC(), 
            num_models = 50, 
            bagging_percent=0.5, 
            bagging_replacement=True, 
            weighting=None, 
            stacking_model=None,
            verbose=False):
        BaseEnsemble.__init__(self)
        self.base_model = base_model
        self.num_models = num_models
        self.bagging_percent = bagging_percent 
        self.bagging_replacement = bagging_replacement 
        self.weighting = weighting
        self.stacking_model = stacking_model 
        self.verbose = verbose 
        
        self.need_to_fit = True
        self.weights = None 
        self.models = None
        self.classes = None
        self.class_list = None 
        
    
    def _init_fit(self, X, Y): 
        self.classes = np.unique(Y) 
        self.class_list = list(self.classes)
        
        
    def _created_model(self, X, Y, indices, model):
        # to assign an F-score weight to each classifier, 
        # sample another subset of the data and use the model 
        # we just train to generate predictions 
        beta = self.weighting 
        if beta or self.verbose:
            error_sample_indices = np.random.random_integers(0,n-1,bagsize)
            error_subset = X[error_sample_indices, :] 
            error_labels = Y[error_sample_indices]
            y_pred = model.predict(error_subset)
            if self.weighting: 
                f_score = fbeta_score(error_labels, y_pred, )
                self.weights[i] = f_score 
            if self.verbose:
                print "Actual non-zero:", np.sum(error_labels != 0)
                num_pred_nz = np.sum(y_pred != 0)
                print "Predicted non-zero:", num_pred_nz
                pred_correct = (y_pred == error_labels)
                pred_nz = (y_pred != 0)
                num_true_nz = np.sum(pred_correct & pred_nz)
                print "True non-zero:", num_true_nz
                print "False non-zero:", num_pred_nz - num_true_nz
                print "---" 
    # normalize weights to add up to 1 
        

    def _predict_votes(self, X): 
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape    
        n_classes = len(self.classes)
        votes = np.zeros( [n_samples, n_classes] )
        
        for weight, model in zip(self.weights, self.models):
            ys = model.predict(X)
            for c in self.classes:
                class_index = self.class_list.index(c)
                votes[ys == c, class_index] += weight
        return votes
    
    def _predict_normalized_votes(self, X): 
        votes = self._predict_votes(X)
        sums = np.sum(votes, axis=1)
        return votes / np.array([sums], dtype='float').T
    
    def _predict_stacked_probs(self, X):
        transformed = self.weighted_transform(X)
        return self.stacking_model.predict_proba(transformed)

    def predict_proba(self, X):
        if self.need_to_fit:
            raise RuntimeError("Trying to call 'predict_proba' before 'fit'")
        if self.stacking_model:
            return self._predict_stacked_probs(X) 
        else:
            return self._predict_normalized_votes(X)

    def predict(self, X, return_probs=False):
        """Every classifier in the ensemble votes for a class. 
           If we're doing stacking, then pass the votes as features into 
           the stacking classifier, otherwise return the majority vote."""
        if self.need_to_fit:
            raise RuntimeError("Trying to call 'predict' before 'fit'")
        
        if self.stacking_model:
            majority_indices = np.argmax(self._predict_stacked_probs(X), axis=1)
        else: 
            majority_indices = np.argmax(self._predict_votes(X), axis=1)        
        return np.array([self.class_list[i] for i in majority_indices])
            
