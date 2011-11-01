import numpy as np 

from sklearn.svm import LinearSVC
from sklearn.metrics import fbeta_score

from base_ensemble import BaseEnsemble 

class ClassifierEnsemble(BaseEnsemble):
    """
        Train an ensemble of classifiers using a 
        subset of the data for each base classifier.  
        
    Parameters
    ----------
    base_model : Any classifier which obeys the fit/predict protocol. 
        Defaults to a Linear SVM with C = 1. 
        
    num_models : int, optional (default = 50)
        How many base classifiers to train. 
    
    sample_percent : float, optional (default=0.5). 
        How much of the data set goes into each bootstrap sample. 
    
    sample_replacement : bool, optional (default = True). 
        Sample with our without replacement. 
    
    weighting : None or float, optional (default=None). 
        Weight individual classifiers in the ensemble by 
            None : all classifiers given equal weight
            <beta> : compute F_beta score for each classifier. 
                     Only works for binary classification.
    
    stacking : classifier, optional (default=None).
        Feed output of weighted individual predictions into another classifier. 
        Suggested model: LogisticRegression. 
        
    
    verbose : bool, optional (default = False).
        Print diagnostic output. 
    """

    def __init__(self, 
            base_model=LinearSVC(), 
            num_models = 50, 
            bagging_percent=0.5, 
            bagging_replacement=True, 
            weighting=None, 
            stacking_model=None,
            verbose=False):
        BaseEnsemble.__init__(self)
        self.base_model = base_model
        self.num_models = num_models
        self.bagging_percent = bagging_percent 
        self.bagging_replacement = bagging_replacement 
        self.weighting = weighting
        self.stacking_model = stacking_model 
        self.verbose = verbose 
        
        self.need_to_fit = True
        self.weights = None 
        self.models = None
        self.classes = None
        self.class_list = None 
        
    
    def _init_fit(self, X, Y): 
        self.classes = np.unique(Y) 
        self.class_list = list(self.classes)
        
        
    def _created_model(self, X, Y, indices, model):
        # to assign an F-score weight to each classifier, 
        # sample another subset of the data and use the model 
        # we just train to generate predictions 
        beta = self.weighting 
        n = X.shape[0]
        if beta or self.verbose:
            error_sample_indices = np.random.random_integers(0,n-1,bagsize)
            error_subset = X[error_sample_indices, :] 
            error_labels = Y[error_sample_indices]
            y_pred = model.predict(error_subset)
            if self.weighting: 
                f_score = fbeta_score(error_labels, y_pred, )
                self.weights[i] = f_score 
            if self.verbose:
                print "Actual non-zero:", np.sum(error_labels != 0)
                num_pred_nz = np.sum(y_pred != 0)
                print "Predicted non-zero:", num_pred_nz
                pred_correct = (y_pred == error_labels)
                pred_nz = (y_pred != 0)
                num_true_nz = np.sum(pred_correct & pred_nz)
                print "True non-zero:", num_true_nz
                print "False non-zero:", num_pred_nz - num_true_nz
                print "---" 
    # normalize weights to add up to 1 
        

    def _predict_votes(self, X): 
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape    
        n_classes = len(self.classes)
        votes = np.zeros( [n_samples, n_classes] )
        
        for weight, model in zip(self.weights, self.models):
            ys = model.predict(X)
            for c in self.classes:
                class_index = self.class_list.index(c)
                votes[ys == c, class_index] += weight
        return votes
    
    def _predict_normalized_votes(self, X): 
        votes = self._predict_votes(X)
        sums = np.sum(votes, axis=1)
        return votes / np.array([sums], dtype='float').T
    
    def _predict_stacked_probs(self, X):
        transformed = self.weighted_transform(X)
        return self.stacking_model.predict_proba(transformed)

    def predict_proba(self, X):
        if self.need_to_fit:
            raise RuntimeError("Trying to call 'predict_proba' before 'fit'")
        if self.stacking_model:
            return self._predict_stacked_probs(X) 
        else:
            return self._predict_normalized_votes(X)

    def predict(self, X, return_probs=False):
        """Every classifier in the ensemble votes for a class. 
           If we're doing stacking, then pass the votes as features into 
           the stacking classifier, otherwise return the majority vote."""
        if self.need_to_fit:
            raise RuntimeError("Trying to call 'predict' before 'fit'")
        
        if self.stacking_model:
            majority_indices = np.argmax(self._predict_stacked_probs(X), axis=1)
        else: 
            majority_indices = np.argmax(self._predict_votes(X), axis=1)        
        return np.array([self.class_list[i] for i in majority_indices])
            