"""
This implements the margin-based ordinal regression methods described
in http://arxiv.org/abs/1408.2327
"""
import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import warnings

def safe_sigmoid(t):
    """
    stable computation of sigmoid function
        1 / (1 + exp(-x))
    """
    idx = t > 0
    out = np.zeros_like(t)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out


def safe_log_loss(Z):
    """
    stable computation of the logistic loss
        ln(1 + exp(-x))
    """
    idx = Z > 0
    out = np.zeros_like(Z)
    out[idx] = np.log(1 + np.exp(-Z[idx]))
    out[~idx] = (-Z[~idx] + np.log(1 + np.exp(Z[~idx])))
    return out


def obj_margin(x0, X, y, alpha, n_class, loss_fd_weights, sample_weight):
    """
    Objective function for the general margin-based formulation

    Parameters
    ----------
    x0: array-like (n_features + n_classes -1,)
        Current guess of optimization weights
    X: array-like (n_samples, n_features)
        Data matrix
    y: array-like (n_samples,)
        target values arranged as ints {0...n_class -1}
    alpha: float
        regularization strength. 0 = no regularization
    n_class: int
        # of unique values in y 
        provided to accelerate iteration
    loss_fd_weights: array-like (n_class, n_class - 1)
        Loss weights for difference between prediction and class int label
        Used to adjust loss for different variants
    L: array-like (n_class - 1, n_class - 1)
        lower triangular matrix
        Used to map cutpoints to theta
        provided to accelerate iteration
    sample_weight: array_like (n_samples,)
        sample weights for training
    """
    w = x0[:X.shape[1]]
    cutweights = x0[X.shape[1]:]
    # cutpoints = L @ cutweights # L @ c forces ordered cutpoints
    cutpoints = np.cumsum(cutweights, dtype='float64')
    loss_fd = loss_fd_weights[y]

    Xw = X @ w
    cut_distance = cutpoints[:, None] - Xw  # (n_class - 1, n_samples)
    err_sign = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)
    err = loss_fd.T * safe_log_loss(err_sign * cut_distance)
    if sample_weight is not None:
        err *= sample_weight
    obj = np.sum(err)
    # regularization
    obj += alpha * 0.5 * (w @ w)
    return obj


def grad_margin(x0, X, y, alpha, n_class, loss_fd_weights, sample_weight):
    """
    Gradient for the general margin-based formulation
    """
    w = x0[:X.shape[1]]
    cutweights = x0[X.shape[1]:]
    cutpoints = np.cumsum(cutweights, dtype='float64')
    loss_fd = loss_fd_weights[y]

    Xw = X @ w
    cut_distance = cutpoints[:, None] - Xw  # (n_class - 1, n_samples)
    err_sign = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)

    Sigma = err_sign * loss_fd.T * safe_sigmoid(-err_sign * cut_distance)
    if sample_weight is not None:
        Sigma *= sample_weight
    grad_w = X.T @ Sigma.sum(axis=0) + alpha * w

    grad_theta = -Sigma.sum(axis=1)
    # Equivalent to grad_c = L.T @ grad_theta
    # (with L as lower triangular matrix) eg. reverse cumulative sum
    grad_c = np.flip(np.cumsum(np.flip(grad_theta), dtype='float64'))
    return np.concatenate((grad_w, grad_c), axis=0)


def threshold_fit(X, y, alpha, n_class, variant='ae',
                  max_iter=1000, verbose=False, tol=1e-12,
                  sample_weight=None):
    """
    Solve the general threshold-based ordinal regression model
    using the logistic loss as surrogate of the 0-1 loss

    Parameters
    ----------
    variant: str (one of {'at', 'se'})
        Loss function variant. One of:
        AT: All-Threshold variant
            Penalizes misclassifications increasingly by distance from target
            Equation 12 in reference [1]
        SE: Squared Error variant
            Penalizes mispredictions by squared error of distance from target
            (treating ordinal labels as integers)
            Section 5 in reference [2]
    References
    ----------
    [1] J. D. M. Rennie and N. Srebro, "Loss Functions for Preference Levels :
    Regression with Discrete Ordered Labels," in Proceedings of the IJCAI
    Multidisciplinary Workshop on Advances in Preference Handling, 2005.
    [2] F. Pedregosa, F. Bach and A. Gramfort, "On the Consistency of Ordinal 
    Regression Methods"	Journal of Machine Learning Research 18, 2014,
    https://arxiv.org/abs/1408.2327
    """
    _, n_features = X.shape
    # set loss forward difference
    if variant == 'at':
        loss_fd = np.ones((n_class, n_class - 1))
    elif variant == 'se':
        a = np.arange(n_class-1)
        b = np.arange(n_class)
        loss_fd = np.abs((a - b[:, None])**2 - (a - b[:, None]+1)**2)
    else:
        raise NotImplementedError("Variant must be in {'at', 'se'}")
    x0 = np.zeros(n_features + n_class - 1)
    x0[X.shape[1]:] = np.arange(n_class - 1)
    # Lower Bound the cutoff points at 0
    # TODO: why is first cutoff unbounded?
    bounds = [(None, None)] * (n_features + 1) + [(0, None)] * (n_class - 2)
    optres = optimize.minimize(
        obj_margin, x0,
        args=(X, y, alpha, n_class, loss_fd, sample_weight),
        method='L-BFGS-B',
        jac=grad_margin, 
        bounds=bounds,
        options={'maxiter':max_iter, 'disp':verbose, 'maxfun':150000},
        tol=tol)
    if not optres.success:
        warnings.warn(
            "\nOrdered Probit Optimization did not succeed" +
            "\nTermination Status: " + str(optres.status) + 
            "\nTermination Message: " + str(optres.message) +
            "\nfunction Value:" + str(optres.fun) +
            "\nIterations:" + str(optres.nit),
            RuntimeWarning)
    weights, cutweights = optres.x[:X.shape[1]], optres.x[X.shape[1]:]
    # cutpoints = L @ cutweights
    cutpoints = np.cumsum(cutweights, dtype='float64')
    return weights, cutpoints


def threshold_predict(X, w, theta):
    """
    Class numbers are assumed to be between 0 and k-1
    """
    tmp = theta[:, None] - np.asarray(X @ w)
    pred = np.sum(tmp < 0, axis=0).astype(np.int)
    return pred


def threshold_proba(X, w, theta):
    """
    Class numbers are assumed to be between 0 and k-1. Assumes
    the `sigmoid` link function is used.
    """
    eta = theta[:, None] - np.asarray(X @ w, dtype=np.float64)
    prob = np.pad(
        safe_sigmoid(eta).T,
        pad_width=((0, 0), (1, 1)),
        mode='constant',
        constant_values=(0, 1))
    return np.diff(prob)


class OrderedLogitRanker(BaseEstimator, ClassifierMixin):
    """
    Classifier that implements the ordinal logistic model 
    (All-Threshold variant)

    Parameters
    ----------
    alpha: float
        Regularization parameter. Zero is no regularization, higher values
        increate the squared l2 regularization.

    variant: str (one of {'at', 'se'})
        Loss function variant. One of:
        AT: All-Threshold variant
            Equation 12 in reference [1]
        SE: Squared Error variant
            Penalizes mispredictions by squared error of distance between labels
            (treating ordinal labels as integers)
            Section 5 in reference [2]
    References
    ----------
    [1] J. D. M. Rennie and N. Srebro, "Loss Functions for Preference Levels :
    Regression with Discrete Ordered Labels," in Proceedings of the IJCAI
    Multidisciplinary Workshop on Advances in Preference Handling, 2005.
    [2] F. Pedregosa, F. Bach and A. Gramfort, "On the Consistency of Ordinal 
    Regression Methods"	Journal of Machine Learning Research 18, 2014,
    https://arxiv.org/abs/1408.2327
    """
    def __init__(self, alpha=1., verbose=0, variant='at'):
        self.alpha = alpha
        self.verbose = verbose
        self.variant = variant
        accepted_variants = ['at', 'se']
        if self.variant not in accepted_variants:
            raise ValueError("Variant must be in {0}".format(accepted_variants))

    def fit(self, X, y, sample_weight=None):
        """

        """
        X, y = check_X_y(X, y, accept_sparse='csr',
                         dtype=np.float64, order="C",)
        self.classes_ = unique_labels(y)
        # hack around sklearn.check_estimator test
        if len(self.classes_) < 2:
            raise ValueError("Can't fit to one class")
        # Map classes to 0...J ints for training
        self.class_nums_ = np.arange(len(self.classes_))
        # maintain mapping for inference        
        class_int_dict = dict(zip(self.classes_, self.class_nums_))
        y_tmp = np.vectorize(class_int_dict.get)(y)
        self.coef_, self.cuts_ = threshold_fit(
            X, y_tmp, self.alpha, len(self.classes_), 
            variant=self.variant, verbose=self.verbose, 
            max_iter=100000,
            sample_weight=sample_weight)
        return self

    def predict(self, X):
        """

        """
        check_is_fitted(self, ['coef_'])
        X = check_array(X, accept_sparse=True)
        int_class = threshold_predict(X, self.coef_, self.cuts_)
        return self.classes_[int_class]

    def predict_proba(self, X):
        """

        """
        check_is_fitted(self, ['coef_'])
        X = check_array(X, accept_sparse=True)
        return threshold_proba(X, self.coef_, self.cuts_)
