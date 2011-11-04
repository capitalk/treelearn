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
    

classes = np.array([0,1])
all_zero = np.array([0,0,0,0])
mixed = np.array([0,1,0,1])
def test_gini():
    result1 = gini(classes, all_zero)
    print "Expected 0.0, Received:", result1 
    assert result1 == 0.0 
    result2 = gini(classes, mixed)
    print "Expected 0.5, Received:", result2 
    assert result2 == 0.5

feature_vec = np.array([0.1, 0.5, 0.9, 1.1])
def test_eval_split():
    slow = slow_eval_split(classes, feature_vec, 0.5, mixed)
    print "Slow GINI", slow 
    fast = eval_gini_split(classes, feature_vec, 0.5, mixed)
    print "Fast GINI", fast 
    assert slow == fast 


labels = np.array([0, 0, 1, 1])    
thresholds = np.unique(feature_vec)
def test_eval_all_splits():
    thresh_slow, score_slow = slow_find_best_gini_split(classes, feature_vec, thresholds, labels)
    print "Slow Thresh", thresh_slow, "Score", score_slow
    assert thresh_slow == 0.5 
    thresh_fast, score_fast = find_best_gini_split(classes, feature_vec, thresholds, labels)
    print "Fast Thresh", thresh_fast, "Score", score_fast
    assert thresh_fast == 0.5
    
