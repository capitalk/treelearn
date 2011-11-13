import numpy as np 
from sklearn.base import BaseEstimator
from sklearn.cluster import MiniBatchKMeans 
from sklearn.linear_model import LinearRegression 
from copy import deepcopy 

class ClusterRegression(BaseEstimator): 
    def __init__(
            self, 
            k=10, 
            base_model = LinearRegression(), 
            cluster_prediction_weights = 'hard', # or 'soft' 
            verbose=False): 
        self.k = k
        self.base_model = base_model 
        self.verbose = verbose 
        self.clusters = MiniBatchKMeans(k)
        self.cluster_prediction_weights = cluster_prediction_weights 
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
        if self.cluster_prediction_weights == 'hard':
            labels = self.clusters.predict(X)
            
            for label in self.models.keys(): 
                mask = (labels == label)
                X_slice = X[mask, :] 
                Y_slice = self.models[label].predict(X_slice) 
                Y[mask] = Y_slice 
        else:
            distances = self.clusters.transform(X)
            inv_dist_squared = 1.0 / (distances ** 2)
            Z = np.sum(inv_dist_squared, axis=1)
            # normalize weights so they add to 1 
            weights = inv_dist_squared / np.array([Z], dtype='float').T
            if self.verbose:
                "First row of weights:", weights[0, :] 
            for label in self.models.keys(): 
                Y_curr = self.models[label].predict(X) 
                Y += Y_curr * weights[:, label]
        return Y
        
        

