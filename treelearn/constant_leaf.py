# TreeLearn
#
# Copyright (C) Capital K Partners
# Author: Alex Rubinsteyn
# Contact: alex [at] capitalkpartners [dot] com 
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.


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
        if isinstance(self.v, int):
            dtype = 'int32'
        else:
            dtype = 'float'
        outputs = np.zeros(X.shape[0], dtype=dtype)
        outputs[:] = self.v
        return outputs 
        
    def fill_predict(self, X, outputs, mask):
        outputs[mask] = self.v 
        
    
    
