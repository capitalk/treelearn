import numpy as np 
from sklearn.base import BaseEstimator
from sklearn.cluster import MiniBatchKMeans 
from sklearn.linear_model import LinearRegression 
from copy import deepcopy 

class ClusterRegression(BaseEstimator): 
    def __init__(self, k=10, base_model = LinearRegression(), verbose=False): 
        self.k = k
        self.base_model = base_model 
        self.verbose = verbose 
        self.clusters = MiniBatchKMeans(k)
        self.models = None 
        
    def fit(self, X, Y):
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
                print "Fitting regression model for cluster", label 
            model = deepcopy(self.base_model)
            mask = (labels == label)
            X_slice = X[mask, :] 
            Y_slice = Y[mask] 
            model.fit(X_slice, Y_slice) 
            self.models[label] = model
             
            
        
    def predict(self, X):
        nrows = X.shape[0] 
        Y = np.zeros(nrows)
        labels = self.clusters.predict(X)
        for label in np.unique(labels):
            mask = (labels == label)
            X_slice = X[mask, :] 
            Y_slice = self.models[label].predict(X_slice) 
            Y[mask] = Y_slice 
        return Y
        
        

