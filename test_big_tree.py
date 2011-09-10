
import numpy as np
import tree

def test(n=10000, d = 200, thresholds=10):
    t = tree.RandomizedTree(thresholds=thresholds)
    x = np.random.randn(n,d)
    y = np.random.randint(0,2,n)
    t.fit(x,y)
    return t 
