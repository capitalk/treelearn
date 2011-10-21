
class ConstantLeaf:
    """Decision tree node which always predicts the same value."""
    def __init__(self, v):
        self.v = v
    
    def to_str(self, indent="", feature_names=None):
        return indent + "Constant(" + str(self.v) + ")"
    
    def __str__(self): 
        return self.to_str() 
        
    def predict(self, data):
        return self.v 
