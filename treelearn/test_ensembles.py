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
import treelearn 

data = np.array([[0,0], [0.1, 0.1], [0.15, 0.15], [0.98, 0.98], [1.0, 1.0], [.99,.99]])
labels = np.array([0,0,0, 1,1,1])

expect0 = np.array([0.05, 0.05])
expect1 = np.array([0.995, 0.995])

def try_predictor(model):
    print "Trying predictor:", model 

    pred0 = model.predict(expect0)
    print "Expected: 0, Received:", pred0
    assert pred0 == 0
    
    pred1 = model.predict(expect1)
    print "Expected: 1, Received:", pred1
    assert pred1 == 1


def test_simple_forest():
    try_predictor(treelearn.train_random_forest(data, labels, min_leaf_size=1))
    
def test_svm_forest():
    try_predictor(treelearn.train_svm_forest(data, labels, C='random', min_leaf_size=1))
