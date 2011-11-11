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

class BaseEnsemble(BaseEstimator):
    def __init__(self, 
            base_model, 
            num_models, 
            bagging_percent,
            bagging_replacement,
            stacking_model, 
            randomize_params, 
            verbose):
        self.base_model = base_model
        self.num_models = num_models
        self.bagging_percent = bagging_percent 
        self.bagging_replacement = bagging_replacement 
        self.stacking_model = stacking_model 
        self.randomize_params = randomize_params 
        self.verbose = verbose
        self.need_to_fit = True
        self.models = None
        
        
    def fit(self, X, Y, **fit_keywords):
        assert self.base_model is not None
        assert self.bagging_percent is not None 
        assert self.bagging_replacement is not None 
        assert self.num_models is not None 
        assert self.verbose is not None
        
        self.need_to_fit = False 
        self.models = [] 
        
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y) 
        
        n = X.shape[0]
        bagsize = int(math.ceil(self.bagging_percent * n))

        
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
            model = copy.deepcopy(self.base_model)
            # randomize parameters using given functions
            for param_name, fn in self.randomize_params.items():
                setattr(model, param_name, fn())
            model.fit(data_subset, label_subset, **fit_keywords)
            self.models.append(model)
            
            self._created_model(X, Y, indices, i, model)
    
        
        # stacking works by treating the outputs of each base classifier as the 
        # inputs to an additional meta-classifier
        if self.stacking_model:
            transformed_data = self.transform(X)
            self.stacking_model.fit(transformed_data, Y)
        
    
    """Can't be instantiated, since member fields like 'models' and 'weights'
    have to be created by descendant classes"""
    def transform(self, X):
        """Convert each feature vector into a row of predictions."""
        assert self.models is not None 
        
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape    
        n_models = len(self.models)
        pred = np.zeros([n_samples, n_models])
        for i, model in enumerate(self.models):
            pred[:, i] = model.predict(X)
        return pred
    
