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

from tree_helpers import * 

def test_majority():
    labels = np.array([1,1,1,2,3,3,3,2,3,3,3,1,3,3,3,3])
    result = majority(labels)
    print "Expected 3:, Received:", result 
    assert result == 3
    classes = [1,2]
    result = majority(labels, classes)
    print "Expected 1:, Received:", result 
    assert result == 1
    

def test_gini():
    classes = np.array([0,1])
    all_zero = np.array([0,0,0,0])
    result1 = gini(classes, all_zero)
    print "Expected 0.0, Received:", result1 
    assert result1 == 0.0 
    mixed = np.array([0,1,0,1])
    result2 = gini(classes, mixed)
    print "Expected 0.5, Received:", result2 
    assert result2 == 0.5 
