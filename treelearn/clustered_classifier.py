import numpy as np 
from clustered import ClusteredEstimator
from sklearn.svm import LinearSVC
from copy import deepcopy 

class ClusteredClassifier(ClusteredEstimator): 
    def __init__(self, k=10, base_model = LinearSVC(), verbose=False): 
        ClusteredEstimator.__init__(self, k,  base_model, verbose)

