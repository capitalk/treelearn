
import recipes
import test_helpers  
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3 
from sklearn.metrics  import mean_square_error
from sklearn.linear_model import LinearRegression



regressors = [
    recipes.train_clustered_ols, 
    recipes.train_clustered_regression_ensemble
]
 
def test_all_regressors():
    x, y  = make_friedman2(10000)
    x_train, y_train, x_test, y_test = test_helpers.split_dataset(x,y)
    #print y_test[:100]
    ols = LinearRegression()
    ols.fit(x_train, y_train)
    ols_pred = ols.predict(x_test)
    #print ols_pred[:100]
    ols_mse = mean_square_error(y_test, ols_pred)
    
    for fn in regressors:
        
        print fn
        model = fn(x_train,y_train)
        print model 
        pred = model.predict(x_test)
        #print pred[:100]
        mse = mean_square_error(y_test, pred)
        
        print "OLS MSE:", ols_mse, " Current MSE:", mse
        assert ols_mse > 1.1*mse


