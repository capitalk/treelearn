class TreeNode:
    """Basic decision tree interior node.""" 
    
    def __init__(self, feature_idx, split_val, left, right):
        self.feature_idx = feature_idx
        self.split_val = split_val 
        self.left = left
        self.right = right
        
    def predict(self, data):
        x = data[self.feature_idx] 
        if x < self.split_val:
            return self.left.predict(data)
        else:
            return self.right.predict(data) 
    
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
        
    def __str__(self, prefix=""):
        return self.to_str()
        
