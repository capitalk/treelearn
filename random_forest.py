
import bagging
import randomized_tree 


def train_random_forest(X, Y, num_trees = 50, sample_percent=0.65, **tree_args)
    """A random forest is a bagging ensemble of randomized trees, so it can
    be implemented by combining the BaggedClassifier and RandomizedTree objects.
    This function is just a helper to your life easier."""
    
    tree = randomized_tree.RandomizedTree(**tree_args)
    rf = bagging.BaggedClassifier(
        base_classifier = tree, 
        num_classifiers=num_trees,
        sample_percent = sample_percent)
    rf.fit(X,y)
    return rf 

