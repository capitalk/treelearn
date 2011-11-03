

import numpy as np 
import sklearn.datasets
import recipes

iris = sklearn.datasets.load_iris()
x = iris.data
y = iris.target


classifiers = [
    
    recipes.train_svm_tree(x,y), 
    recipes.train_sgd_tree(x,y), 
    recipes.train_svm_forest(x, y), 
    recipes.train_sgd_forest(x,y), 
    recipes.train_random_forest(x, y)
]

def test_all_classifiers():
    for model in classifiers:
        print model 
        pred = model.predict(x)
        num_incorrect = np.sum(pred != y)
        print "Expected num_incorrect <= 7, got:", num_incorrect 
        assert num_incorrect <= 7
        
