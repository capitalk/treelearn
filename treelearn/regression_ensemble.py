import numpy as np 
from sklearn.linear_model import LinearRegression 
from base_ensemble import BaseEnsemble

class RegressionEnsemble(BaseEnsemble):
    def __init__(self, 
            base_model=LinearRegression(), 
            num_models = 50, 
            bagging_percent=0.5, 
            bagging_replacement=True,
            feature_subset_percent = 1.0,  
            stacking_model=None,
            randomize_params = {}, 
            additive = False, 
            verbose=False):
                
        BaseEnsemble.__init__(self, 
            base_model, 
            num_models, 
            bagging_percent,
            bagging_replacement, 
            feature_subset_percent, 
            stacking_model, 
            randomize_params, 
            additive, 
            verbose)
            
       
        
    def predict(self, X):
        pred = self.transform(X)
        if self.stacking_model: 
            return self.stacking_model.predict(pred)
        else: 
            return np.dot(pred, self.weights)

    def _init_fit(self, X, Y): 
        pass 
        
    def _created_model(self, X, Y, indices, i, model):
        pass 
