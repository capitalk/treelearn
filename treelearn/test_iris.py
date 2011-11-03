

import numpy as np 
import sklearn.datasets
import recipes

iris = sklearn.datasets.load_iris()
x = iris.data
y = iris.target


classifiers = [
    
    recipes.train_svm_tree, 
    recipes.train_sgd_tree, 
    recipes.train_svm_forest, 
    recipes.train_sgd_forest, 
    recipes.train_random_forest
]

def test_all_classifiers():
    for model_constructor in classifiers:
        
        print model_constructor
        model = model_constructor(x,y)
        print model 
        pred = model.predict(x)
        num_incorrect = np.sum(pred != y)
        print "Expected num_incorrect <= 7, got:", num_incorrect 
        assert num_incorrect <= 7
        
