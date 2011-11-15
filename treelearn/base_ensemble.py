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


from copy import deepcopy 
import numpy as np 
import random 
import math 

from sklearn.base import BaseEstimator 

from tree_helpers import clear_sklearn_fields
from typecheck import check_estimator, check_dict, check_int, check_bool

class BaseEnsemble(BaseEstimator):
    def __init__(self, 
            base_model, 
            num_models, 
            bagging_percent,
            bagging_replacement,
            feature_subset_percent, 
            stacking_model, 
            randomize_params, 
            additive, 
            verbose):
        check_estimator(base_model)
        check_int(num_models)
        
        self.base_model = base_model
        self.num_models = num_models
        self.bagging_percent = bagging_percent 
        self.bagging_replacement = bagging_replacement 
        self.feature_subset_percent = feature_subset_percent 
        self.stacking_model = stacking_model 
        self.randomize_params = randomize_params 
        self.additive = additive 
        self.verbose = verbose
        self.need_to_fit = True
        self.models = None
        self.weights = None 
        
        
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
        
        n_rows, total_features = X.shape
        bagsize = int(math.ceil(self.bagging_percent * n_rows))
        
        
        if self.additive: 
            self.weights = np.ones(self.num_models, dtype='float') 
        else:
            self.weights = np.ones(self.num_models, dtype='float') / self.num_models            
        
        
        # each derived class needs to implement this 
        self._init_fit(X,Y)
        if self.feature_subset_percent < 1:
            n_features = int(math.ceil(self.feature_subset_percent * total_features))
            self.feature_subsets = [] 
        else:
            n_features = total_features 
            self.feature_subsets = None 
            
        for i in xrange(self.num_models):
            if self.verbose:
                print "Training iteration", i 
            
            if self.bagging_replacement: 
                indices = np.random.random_integers(0,n_rows-1,bagsize)
            else:
                p = np.random.permutation(n_rows)
                indices = p[:bagsize] 
                
            data_subset = X[indices, :]
            if n_features < total_features: 
                feature_indices = np.random.permutation(total_features)[:n_features]
                self.feature_subsets.append(feature_indices)
                data_subset = data_subset[:, feature_indices]
                
            label_subset = Y[indices] 
            model = deepcopy(self.base_model)
            # randomize parameters using given functions
            for param_name, fn in self.randomize_params.items():
                setattr(model, param_name, fn())
            model.fit(data_subset, label_subset, **fit_keywords)
            
            self.models.append(model)
            self._created_model(X, Y, indices, i, model)
            
            if self.additive: 
                if n_features < total_features:
                    Y -= model.predict(X[:, feature_indices])
                else: 
                    Y -= model.predict(X)
                    
            clear_sklearn_fields(model) 
        # stacking works by treating the outputs of each base classifier as the 
        # inputs to an additional meta-classifier
        if self.stacking_model:
            transformed_data = self.transform(X)
            self.stacking_model.fit(transformed_data, Y)
        
    
    def transform(self, X):
        """Convert each feature vector into a row of predictions."""
        assert self.models is not None 
        
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape    
        n_models = len(self.models)
        pred = np.zeros([n_samples, n_models])
        if self.feature_subsets:
            for i, model in enumerate(self.models):
                feature_indices = self.feature_subsets[i]
                X_subset = X[:, feature_indices] 
                pred[:, i] = model.predict(X_subset)
        else:
            for i, model in enumerate(self.models):
                pred[:, i] = model.predict(X)
        return pred
    
