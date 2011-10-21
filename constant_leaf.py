import numpy as np 

class ConstantLeaf:
    """Decision tree node which always predicts the same value."""
    def __init__(self, v):
        self.v = v
    
    def to_str(self, indent="", feature_names=None):
        return indent + "Constant(" + str(self.v) + ")"
    
    def __str__(self): 
        return self.to_str() 
        
    def predict(self, X):
        X = np.atleast_2d(X)
        outputs = np.zeros(X.shape[0])
        outputs[:] = self.v
        return outputs 
        
    def fill_predict(self, X, outputs, mask):
        outputs[mask] = self.v 
        
    
    
