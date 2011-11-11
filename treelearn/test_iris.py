

import recipes
import numpy as np 
import sklearn.datasets
from test_helpers import split_dataset

iris = sklearn.datasets.load_iris()
x_train, y_train, x_test, y_test = split_dataset(iris.data, iris.target)


classifiers = [
    recipes.train_svm_tree, 
    recipes.train_sgd_tree, 
    #recipes.train_svm_forest, 
    #recipes.train_sgd_forest, 
    recipes.train_random_forest
]

def test_all_classifiers():
    for model_constructor in classifiers:
        
        print model_constructor
        model = model_constructor(x_train, y_train)
        print model 
        pred = model.predict(x_test)
        num_incorrect = np.sum(pred != y_test)
        print "Expected num_incorrect <= 20, got:", num_incorrect 
        assert num_incorrect <= 15
        
