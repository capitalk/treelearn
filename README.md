TreeLearn started as a Python implementation of Breiman's Random Forest 
but is being slowly generalized into a tree ensemble library. Currently 
the only ensemble method is a BaggedClassifier and the two base classifiers
are RandomizedTree and SVM_Tree (uses a hyperplane for each split). 


## Creating a Random Forest

A random forest is simply a bagging ensemble of randomized tree. To construct
these with default parameters:

    forest = treelearn.ClassifierEnsemble(base_model = treelearn.RandomizedTree())


## Training

Place your training data in a n-by-d numpy array, where n is the number of 
training  examples and d is the dimensionality of your data. 
Place labels in an n-length numpy array. Then call: 

    forest.fit(Xtrain,Y)

If you're lazy, there's a helper for simultaneously creating and training a random forest:

    forest = treelearn.train_random_forest(X, Y)


## Classification

    forest.predict(Xtest)
 

## ClassifierEnsemble options

 * base_model = any classifier which obeys the fit/predict protocol

 * num_models = size of the forest 
 
 * bagging_percent = what percentage of your data each classifier is trained on
 
 * bagging_replacement = sample with or without replacement 

 * stacking_model = treat outputs of base classifiers as inputs to given model

 
## RandomizedTree options 
    
 * num_features_per_node = number of features each node of a tree should
        consider (default = log2 of total features)
    
 * min_leaf_size = stop splitting if we get down to this number of data points 

 * max_height = stop splitting if we exceed this number of tree levels

 * max_thresholds = how many feature value thesholds to consider (use None for all values)

## SVM_Tree options 
 * num_features_per_node = size of random feature subset at each node, 
        default = sqrt(total number of features)
 
 * C = Tradeoff between error and L2 regularizer of linear SVM
        
 * max_depth = When you get to this depth, train an SVM on all features 
        and stop splitting the data. 
        
 * min_leaf_size = stop splitting when any subset of the data gets smaller 
        than this. 
