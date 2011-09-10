
import numpy as np 
import tree

def test_gini():
    classes = np.array([0,1])
    all_zero = np.array([0,0,0,0])
    result1 = tree.gini(classes, all_zero)
    print "Expected 0.0, Received:", result1 
    assert result1 == 0.0 
    mixed = np.array([0,1,0,1])
    result2 = tree.gini(classes, mixed)
    print "Expected 0.5, Received:", result2 
    assert result2 == 0.5 
    

def test_simple_tree():
    data = np.array([[0,0], [0.1, 0.1], [1.0, 1.0], [.99,.99]])
    labels = np.array([0,0,1,1])
    t = tree.RandomizedTree(min_leaf_size=1)
    t.fit(data,labels)
    pred0 = t.predict(np.array([0.05, 0.05]))
    print "Expected: 0, Received:", pred0
    assert pred0 == 0
    
    pred1 = t.predict(np.array([0.995, 0.995]))
    print "Expected: 1, Received:", pred1
    assert pred1 == 1
