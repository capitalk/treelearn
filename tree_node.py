
from sklearn.base import BaseEstimator 

class TreeNode(BaseEstimator):
    """Basic decision tree interior node.""" 
    
    def __init__(self, feature_idx, split_val, left, right):
        self.feature_idx = feature_idx
        self.split_val = split_val 
        self.left = left
        self.right = right
        
   
    def predict(self, X):
        """Inefficient since calling this method recursively copy outputs"""
        outputs = np.zeros(X.shape[0])
        col = X[:, self.feature_idx] 
        split = col < self.split_val 
        left_mask = mask & split
        outputs[left_mask] = self.left.predict(X[left_mask, :])
        right_mask = mask & ~split 
        outputs[right_mask] = self.right.predict(X[right_mask, :])
        return outputs 
        
     
    def fill_predict(self, X, outputs, mask):
        """instead of returning output values, let the leaves fill an 
        output matrix
        """
        col = X[:, self.feature_idx] 
        split = col < self.split_val 
        left_mask = mask & split
        right_mask = mask & ~split 
        self.left.fill_predict(X, outputs, left_mask)
        self.right.fill_predict(X, outputs, right_mask)
    
        
    def to_str(self, indent="", feature_names=None):
        if feature_names:
            featureStr = feature_names[feature_idx]
        else:
            featureStr = "x[" + str(self.feature_idx) + "]"
        longer_indent = indent + "  " 
        left = self.left.to_str(indent = longer_indent)
        right = self.right.to_str(indent = longer_indent)
        cond = "if %s < %f:" % (featureStr, self.split_val)
        return indent + cond + "\n" +  left + "\n" + indent + "else:\n" + right
        
        
