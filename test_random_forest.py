
import numpy as np 
import rf

def test_majority():
    x = np.array([0,0,0,1,1,0,2,2,2,2,2,0,1,0])
    assert rf.majority([0,1,2], x) == 0

def test_simple_tree():
    data = np.array([[0,0], [0.1, 0.1], [0.15, 0.15], [0.98, 0.98], [1.0, 1.0], [.99,.99]])
    labels = np.array([0,0,0, 1,1,1])
    forest = rf.RandomForest()
    forest.fit(data,labels)
    pred0 = forest.predict(np.array([0.05, 0.05]))
    print "Expected: 0, Received:", pred0
    assert pred0 == 0
    
    pred1 = forest.predict(np.array([0.995, 0.995]))
    print "Expected: 1, Received:", pred1
    assert pred1 == 1 
