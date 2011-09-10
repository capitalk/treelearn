
import numpy as np 
import randomized_tree as tree


    

def test_simple_tree():
    data = np.array([[0,0], [0.1, 0.1], [1.0, 1.0], [.99,.99]])
    labels = np.array([0,0,1,1])
    t = tree.RandomizedTree(min_leaf_size=1)
    t.fit(data,labels)
    print t 
    pred0 = t.predict(np.array([0.05, 0.05]))
    print "Expected: 0, Received:", pred0
    assert pred0 == 0
    
    pred1 = t.predict(np.array([0.995, 0.995]))
    print "Expected: 1, Received:", pred1
    assert pred1 == 1

def test_big_tree(n=1000, d = 50, thresholds=10):
    t = tree.RandomizedTree(thresholds=thresholds)
    x = np.random.randn(n,d)
    y = np.random.randint(0,2,n)
    t.fit(x,y)
    return t 
