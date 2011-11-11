import numpy as np 

def split_dataset(x, y, prct_train=0.5):
    nrows, ncols = x.shape
    indices = np.arange(nrows)
    np.random.shuffle(indices)
    ntrain = int(nrows * prct_train)
    train_indices = indices[:ntrain]
    test_indices = indices[ntrain:] 
    x_train = x[train_indices, :] 
    x_test = x[test_indices, :] 
    y_train = y[train_indices]
    y_test = y[test_indices] 
    return x_train, y_train, x_test, y_test 
