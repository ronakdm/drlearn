"""Functions for implementing scikit-learn style estimators."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.special import expit, softmax

from .optim import minimize_prospect, minimize_lsvrg
from .spectral_risk import make_superquantile_spectrum, make_spectral_risk_measure_oracle

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

    # if estimator.loss in ["squared_error", "binary_cross_entropy"]:
    #     return X @ estimator.coef_ + estimator.intercept_
    # elif estimator.loss == "multinomial_cross_entropy":
    #     n_class = len(estimator.classes_)
    #     return X @ estimator.coef_.reshape((-1, n_class)) + estimator.intercept_
    # else:
    #     raise ValueError(f"unrecognized loss '{estimator.loss}'! options: 'squared_error', 'binary_cross_entropy', 'multinomial_cross_entropy'")
    return X @ estimator.coef_ + estimator.intercept_

class Ridge(BaseEstimator, RegressorMixin):
    """
    Linear distributionally robust least squares with L2 regularization.

    Parameters
    ----------
    spectrum : np.ndarray or NoneType, default=None
        Spectrum weights for the spectral risk measure. If None, defaults to superquantile spectrum with head_prob=0.5.
    penalty : str, default="chi2"
        Penalty type for the dual maximization oracle. Options are "chi2" or "kl".
    shift_cost : float, default=0.05
        Shift cost for the dual maximization oracle. Must be non-negative.
    weight_decay : float, default=1.0
        Regularization strength. Must be non-negative.
    fit_intercept : bool, default=True
        Whether to fit the intercept term.
    optim : string, default="prospect"
        Optimization algorithm to use. Options are "prospect" or "lsvrg".

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Intercept term.
    """
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
        """
        Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : Ridge
            Fitted estimator.
        """
        coef = fit(
            X, y, 
            self.optim, 
            self.loss, 
            self.dual_max_oracle, 
            self.weight_decay, 
            self.fit_intercept,
            self.penalty,
            self.shift_cost
        )
        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coef
        self.is_fitted = True
        return self
        
    def __sklearn_is_fitted__(self):
        return self.is_fitted

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        return get_scores(self, X)


class BinaryLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Linear distributionally robust binary logistic regression with L2 regularization.

    Parameters
    ----------
    spectrum : np.ndarray or NoneType, default=None
        Spectrum weights for the spectral risk measure. If None, defaults to superquantile spectrum with head_prob=0.5.
    penalty : str, default="chi2"
        Penalty type for the dual maximization oracle. Options are "chi2" or "kl".
    shift_cost : float, default=0.05
        Shift cost for the dual maximization oracle. Must be non-negative.
    weight_decay : float, default=1.0
        Regularization strength. Must be non-negative.
    fit_intercept : bool, default=True
        Whether to fit the intercept term.
    optim : string, default="prospect"
        Optimization algorithm to use. Options are "prospect" or "lsvrg".

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Intercept term.
    """
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
        """
        Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : BinaryLogisticRegression
            Fitted estimator.
        """
        assert np.isin(y, np.array([0, 1])).all(), "y should be binary labels {0, 1}"
        coef = fit(
            X, y, 
            self.optim, 
            self.loss, 
            self.dual_max_oracle, 
            self.weight_decay, 
            self.fit_intercept,
            self.penalty,
            self.shift_cost
        )
        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coef
        self.is_fitted = True
        return self
        
    def __sklearn_is_fitted__(self):
        return self.is_fitted

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values (either 0 or 1).
        """
        scores = get_scores(self, X)
        return (scores >= 0.).astype(int)

    def predict_proba(self, X):
        """
        Class probabilities for logistic models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, )
            Probabilities for class 1.
        """
        scores = get_scores(self, X)
        return expit(scores)

class MultinomialLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Linear distributionally robust multinomial logistic regression with L2 regularization.

    Parameters
    ----------
    spectrum : np.ndarray or NoneType, default=None
        Spectrum weights for the spectral risk measure. If None, defaults to superquantile spectrum with head_prob=0.5.
    penalty : str, default="chi2"
        Penalty type for the dual maximization oracle. Options are "chi2" or "kl".
    shift_cost : float, default=0.05
        Shift cost for the dual maximization oracle. Must be non-negative.
    weight_decay : float, default=1.0
        Regularization strength. Must be non-negative.
    fit_intercept : bool, default=True
        Whether to fit the intercept term.
    optim : string, default="prospect"
        Optimization algorithm to use. Options are "prospect" or "lsvrg".

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Intercept term.
    """
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
        """
        Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : MultinomialLogisticRegression
            Fitted estimator.
        """
        self.classes_, y = np.unique(y, return_inverse=True)
        coef = fit(
            X, y, 
            self.optim, 
            self.loss, 
            self.dual_max_oracle, 
            self.weight_decay, 
            self.fit_intercept,
            self.penalty,
            self.shift_cost
        ).reshape((-1, len(self.classes_)))
        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coef
        self.is_fitted = True
        return self
        
    def __sklearn_is_fitted__(self):
        return self.is_fitted

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values (written as the class indices provided during training).
        """
        scores = get_scores(self, X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """
        Class probabilities for logistic models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities. For binary problems this has shape (n_samples, 2).
        """
        scores = get_scores(self, X)
        return softmax(scores, axis=1)