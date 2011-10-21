import numpy as np 

from tree_helpers import * 

def test_majority():
    classes = np.array([1,2,3])
    labels = np.array([1,1,1,2,3,3,3,2,3,3,3,1,3,3,3,3])
    result = majority(classes, labels)
    print "Expected 3:, Received:", result 
    assert result == 3

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
