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
from recipes import train_random_forest, train_svm_forest
from classifier_ensemble import ClassifierEnsemble
from randomized_tree import RandomizedTree 
from sklearn.linear_model import LogisticRegression

n = 200
left_data = np.random.randn(n, 10)
left_labels = np.zeros(n, dtype='int')

right_data = 10*(np.random.randn(n,10)-2)
right_labels = np.ones(n, dtype='int')

data = np.concatenate([left_data, right_data])
labels = np.concatenate([left_labels, right_labels])


def try_predictor(model):
    print "Trying predictor:", model 

    pred0 = model.predict(left_data)
    fp = np.sum(pred0 != 0)
    print "False positives:", fp
    assert fp < (n / 10)
    
    pred1 = model.predict(right_data)
    fn = np.sum(pred1 != 1)
    print "False negatives:", fn
    assert fn < (n/ 10)


def test_simple_forest():
    try_predictor(train_random_forest(data, labels))
    
def test_svm_forest():
    try_predictor(train_svm_forest(data, labels,  tree_args={'verbose':True}))

def test_stacked_random_forest():
    t = RandomizedTree(min_leaf_size=1)
    lr = LogisticRegression()
    ensemble = ClassifierEnsemble(base_model=t, stacking_model=lr)
    ensemble.fit(data, labels)
    try_predictor(ensemble)
