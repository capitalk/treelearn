
from sklearn.linear_model import LinearRegression 
from base_ensemble import BaseEnsemble

class RegressionEnsemble(BaseEnsemble):
    def __init__(self, 
            base_model=LinearRegression(), 
            num_models = 50, 
            bagging_percent=0.5, 
            bagging_replacement=True, 
            weighting=None, 
            stacking_model=None,
            verbose=False):
                
        BaseEnsemble.__init__(self, 
            base_model, 
            num_models, 
            bagging_percent,
            bagging_replacement, 
            self.weighting, 
            self.stacking_model, 
            self.verbose)
        
    def predict(self, X):
        weighted_outputs = self.weighted_transform(X)
        return np.sum(weighted_outputs, axis=1)


    def _init_fit(self, X, Y): 
        pass 
        
    def _created_model(self, X, Y, indices, i, model):
        pass 
