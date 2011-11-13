import numpy as np 
from clustered import ClusteredEstimator
from sklearn.linear_model import LinearRegression 
from copy import deepcopy 

class ClusteredRegression(ClusteredEstimator): 
    def __init__(
            self, 
            k=10, 
            base_model = LinearRegression(), 
            cluster_prediction_weights = 'hard', # or 'soft' 
            verbose=False): 
        ClusteredEstimator.__init__(self, k,  base_model, verbose)
        self.cluster_prediction_weights = cluster_prediction_weights 

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
        
        

