
from sklearn.base import BaseEstimator
from sklearn.cluster import MiniBatchKMeans 

class ClusterRegression(BaseEstimator): 
    def __init__(self, k): 
        self.k = k
        self.kmeans = MiniBatchKMeans(k)

    def fit(self, X, Y): 
        # also get back the labels so we can use them to create regressors 
        self.kmeans.fit(X)
        
    def predict(self, X)): 
        pass
        

