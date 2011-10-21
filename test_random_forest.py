
import numpy as np 
import random_forest as rf 
import randomized_tree as rt 

def test_simple_tree():
    data = np.array([[0,0], [0.1, 0.1], [0.15, 0.15], [0.98, 0.98], [1.0, 1.0], [.99,.99]])
    labels = np.array([0,0,0, 1,1,1])
    t = rt.RandomizedTree(min_leaf_size=1)
    forest = rf.RandomForest(base_classifier=t)
    forest.fit(data,labels)
    print forest 
    pred0 = forest.predict(np.array([0.05, 0.05]))
    print "Expected: 0, Received:", pred0
    assert pred0 == 0
    
    pred1 = forest.predict(np.array([0.995, 0.995]))
    print "Expected: 1, Received:", pred1
    assert pred1 == 1 
