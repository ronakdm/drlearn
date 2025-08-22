"""Functions for implementing scikit-learn style estimators."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.special import expit, softmax

from src.drlearn.optim import minimize_prospect, minimize_lsvrg
from src.drlearn.spectral_risk import make_superquantile_spectrum, make_spectral_risk_measure_oracle

def fit(X, y, optim, loss, dual_max_oracle, weight_decay, fit_intercept, penalty, shift_cost):

    # Check that X and y have correct shape
    X, y = check_X_y(X, y)
    n = len(X)

    # Default maximization oracle is superquantile
    if dual_max_oracle is None:
        spectrum = make_superquantile_spectrum(n, 0.5)
        dual_max_oracle = make_spectral_risk_measure_oracle(spectrum, penalty, shift_cost)

    # Check intercept
    if fit_intercept:
        X = np.concatenate([np.ones(shape=(n, 1)), X], axis=1)

    if optim == "prospect":
        opt_func = minimize_prospect
    elif optim == "lsvrg":
        opt_func = minimize_lsvrg
    else:
        raise ValueError(f"unrecognized optimizer '{optim}'! options: 'prospect', 'lsvrg'")

    result = opt_func(
        loss,
        dual_max_oracle,
        X,
        y,
        weight_decay=weight_decay,
        eval_interval=5 * n,
        fit_intercept=fit_intercept,
        penalty=penalty,
        shift_cost=shift_cost,
    )

    return result['primal_solution']

def get_scores(estimator, X):

    # Check if fit has been called
    check_is_fitted(estimator)
    
    # Input validation
    X = check_array(X)
    n = len(X)
    if estimator.fit_intercept:
        X = np.concatenate([np.ones(shape=(n, 1)), X], axis=1)

    if estimator.loss in ["squared_error", "binary_cross_entropy"]:
        return X @ estimator.coef
    elif estimator.loss == ["multinomial_cross_entropy"]:
        n_class = len(estimator.classes_)
        return X @ estimator.coef.reshape((-1, n_class))

class Ridge(BaseEstimator, RegressorMixin):
    def __init__(
            self, 
            spectrum=None, 
            penalty="chi2",
            shift_cost=0.05,
            weight_decay=1.0, 
            fit_intercept=True, 
            optim="prospect"
        ):
        self.dual_max_oracle = None if spectrum is None else make_spectral_risk_measure_oracle(spectrum, penalty, shift_cost)
        self.optim = optim
        self.weight_decay = weight_decay
        self.loss = "squared_error"
        self.is_fitted = False
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.shift_cost = shift_cost

    def fit(self, X, y):
        self.coef = fit(
            X, y, 
            self.optim, 
            self.loss, 
            self.dual_max_oracle, 
            self.weight_decay, 
            self.fit_intercept,
            self.penalty,
            self.shift_cost
        )
        self.is_fitted = True
        return self
        
    def __sklearn_is_fitted__(self):
        return self.is_fitted

    def predict(self, X):
        return get_scores(self, X)


class BinaryLogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(
            self, 
            spectrum=None, 
            penalty="chi2",
            shift_cost=0.05,
            weight_decay=1.0, 
            fit_intercept=True, 
            optim="prospect"
        ):
        self.dual_max_oracle = None if spectrum is None else make_spectral_risk_measure_oracle(spectrum, penalty, shift_cost)
        self.optim = optim
        self.weight_decay = weight_decay
        self.loss = "binary_cross_entropy"
        self.is_fitted = False
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.shift_cost = shift_cost

    def fit(self, X, y):
        self.coef = fit(
            X, y, 
            self.optim, 
            self.loss, 
            self.dual_max_oracle, 
            self.weight_decay, 
            self.fit_intercept,
            self.penalty,
            self.shift_cost
        )
        self.is_fitted = True
        return self
        
    def __sklearn_is_fitted__(self):
        return self.is_fitted

    def predict(self, X):
        scores = get_scores(self, X)
        return (scores >= 0.).astype(int)

    def predict_proba(self, X):
        scores = get_scores(self, X)
        return expit(scores)

class MultinomialLogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(
            self, 
            spectrum=None, 
            penalty="chi2",
            shift_cost=0.05,
            weight_decay=1.0, 
            fit_intercept=True, 
            optim="prospect"
        ):
        self.dual_max_oracle = None if spectrum is None else make_spectral_risk_measure_oracle(spectrum, penalty, shift_cost)
        self.optim = optim
        self.weight_decay = weight_decay
        self.loss = "multinomial_cross_entropy"
        self.is_fitted = False
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.shift_cost = shift_cost

    def fit(self, X, y):
        self.coef = fit(
            X, y, 
            self.optim, 
            self.loss, 
            self.dual_max_oracle, 
            self.weight_decay, 
            self.fit_intercept,
            self.penalty,
            self.shift_cost
        )
        self.is_fitted = True
        self.classes_ = unique_labels(y)
        return self
        
    def __sklearn_is_fitted__(self):
        return self.is_fitted

    def predict(self, X):
        scores = get_scores(self, X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = get_scores(self, X)
        return softmax(scores, axis=1)