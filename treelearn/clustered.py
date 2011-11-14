import numpy as np 
from copy import deepcopy 


from sklearn.base import BaseEstimator
from sklearn.cluster import MiniBatchKMeans 
from sklearn.linear_model import LinearRegression 

from typecheck import check_estimator, check_dict, check_int, check_bool
from tree_helpers import clear_sklearn_fields

class ClusteredEstimator(BaseEstimator): 
    """Base class for ClusteredRegression and ClusteredClassifier"""
    def __init__(self, k, base_model, verbose=False): 
        check_int(k)
        check_estimator(base_model)
        check_bool(verbose) 
        
        self.k = k
        self.base_model = base_model 
        self.verbose = verbose 
        self.clusters = MiniBatchKMeans(k)
        self.models = None 
        
    def fit(self, X, Y, **fit_keywords):
        self.models = {}
        if self.verbose:
            print "Clustering X"
        # also get back the labels so we can use them to create regressors 
        self.clusters.fit(X)
        labels = self.clusters.labels_ 
        # clear this field so that it doesn't get serialized later
        self.clusters.labels_ = None 
        for label in np.unique(labels):
            if self.verbose: 
                print "Fitting model for cluster", label 
            model = deepcopy(self.base_model)
            mask = (labels == label)
            X_slice = X[mask, :] 
            Y_slice = Y[mask] 
            model.fit(X_slice, Y_slice, **fit_keywords)
            
            # clear sklearn's left over junk to make pickled strings smaller  
            clear_sklearn_fields(model)
            self.models[label] = model
             
    def predict(self, X): 
        if self.verbose:
            print "Prediction inputs of shape", X.shape 
        nrows = X.shape[0] 
        Y = np.zeros(nrows)
        if self.verbose:
            print "Assigning cluster labels to input data" 
        labels = self.clusters.predict(X)
        for label in self.models.keys(): 
            mask = (labels == label)
            if self.verbose: 
                print "Predicting cluster", label, "nvectors = ", np.sum(mask)
            
            X_slice = X[mask, :] 
            model = self.models[label] 
            Y[mask] = model.predict(X_slice) 
        return Y 

