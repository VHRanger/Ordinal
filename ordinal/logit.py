"""
some ordinal regression algorithms

This implements the margin-based ordinal regression methods described
in http://arxiv.org/abs/1408.2327
"""
import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def sigmoid(t):
    """
    sigmoid function, 1 / (1 + exp(-t))
    stable computation
    """
    idx = t > 0
    out = np.zeros_like(t)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out


def log_loss(Z):
    """
    stable computation of the logistic loss
    """
    idx = Z > 0
    out = np.zeros_like(Z)
    out[idx] = np.log(1 + np.exp(-Z[idx]))
    out[~idx] = (-Z[~idx] + np.log(1 + np.exp(Z[~idx])))
    return out


def obj_margin(x0, X, y, alpha, n_class, weights, L, sample_weight):
    """
    Objective function for the general margin-based formulation
    """
    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]
    theta = L.dot(c)
    loss_fd = weights[y]

    Xw = X.dot(w)
    Alpha = theta[:, None] - Xw  # (n_class - 1, n_samples)
    S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)

    err = loss_fd.T * log_loss(S * Alpha)
    if sample_weight is not None:
        err *= sample_weight
    obj = np.sum(err)
    obj += alpha * 0.5 * (np.dot(w, w))
    return obj


def grad_margin(x0, X, y, alpha, n_class, weights, L, sample_weight):
    """
    Gradient for the general margin-based formulation
    """
    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]
    theta = L.dot(c)
    loss_fd = weights[y]

    Xw = X.dot(w)
    Alpha = theta[:, None] - Xw  # (n_class - 1, n_samples)
    S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)
    # Alpha[idx] *= -1
    # W[idx.T] *= -1

    Sigma = S * loss_fd.T * sigmoid(-S * Alpha)
    if sample_weight is not None:
        Sigma *= sample_weight

    grad_w = X.T.dot(Sigma.sum(0)) + alpha * w

    grad_theta = -Sigma.sum(1)
    grad_c = L.T.dot(grad_theta)
    return np.concatenate((grad_w, grad_c), axis=0)


def threshold_fit(X, y, alpha, n_class, mode='ae',
                  max_iter=1000, verbose=False, tol=1e-12,
                  sample_weight=None):
    """
    Solve the general threshold-based ordinal regression model
    using the logistic loss as surrogate of the 0-1 loss

    Parameters
    ----------
    mode : string, one of {'at', 'se', 'it'}
    """
    n_samples, n_features = X.shape

    # convert from c to theta
    L = np.zeros((n_class - 1, n_class - 1))
    L[np.tril_indices(n_class-1)] = 1.

    if mode == 'at':
        # loss forward difference
        loss_fd = np.ones((n_class, n_class - 1))
    elif mode == 'it':
        loss_fd = np.diag(np.ones(n_class - 1)) + \
            np.diag(np.ones(n_class - 2), k=-1)
        loss_fd = np.vstack((loss_fd, np.zeros(n_class - 1)))
        loss_fd[-1, -1] = 1  # border case
    elif mode == 'se':
        a = np.arange(n_class-1)
        b = np.arange(n_class)
        loss_fd = np.abs((a - b[:, None])**2 - (a - b[:, None]+1)**2)
    else:
        raise NotImplementedError

    x0 = np.zeros(n_features + n_class - 1)
    x0[X.shape[1]:] = np.arange(n_class - 1)
    options = {'maxiter' : max_iter, 'disp': verbose, 
               'maxfun': 150000}
    if n_class > 2:
        bounds = [(None, None)] * (n_features + 1) + \
                 [(0, None)] * (n_class - 2)
    else:
        bounds = None

    sol = optimize.minimize(obj_margin, x0, method='L-BFGS-B',
        jac=grad_margin, bounds=bounds, options=options,
        args=(X, y, alpha, n_class, loss_fd, L, sample_weight),
        tol=tol)
    if verbose and not sol.success:
        print(sol.message)

    w, c = sol.x[:X.shape[1]], sol.x[X.shape[1]:]
    theta = L.dot(c)
    return w, theta


def threshold_predict(X, w, theta):
    """
    Class numbers are assumed to be between 0 and k-1
    """
    tmp = theta[:, None] - np.asarray(X.dot(w))
    pred = np.sum(tmp < 0, axis=0).astype(np.int)
    return pred


def threshold_proba(X, w, theta):
    """
    Class numbers are assumed to be between 0 and k-1. Assumes
    the `sigmoid` link function is used.
    """
    eta = theta[:, None] - np.asarray(X.dot(w), dtype=np.float64)
    prob = np.pad(
        sigmoid(eta).T,
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

    variant: str (one of {'at', 'se', 'it'})
        Loss function variant. One of:
        AT: All-Threshold variant
        IT: Immediate-Threshold variant
            Contrary to the OrdinalLogistic model, this variant
            minimizes a convex surrogate of the 0-1 loss. Closer to 
            classification than AT
        SE: Squared Error variant

    References
    ----------
    J. D. M. Rennie and N. Srebro, "Loss Functions for Preference Levels :
    Regression with Discrete Ordered Labels," in Proceedings of the IJCAI
    Multidisciplinary Workshop on Advances in Preference Handling, 2005.
    """
    def __init__(self, alpha=1., verbose=0, variant='at'):
        self.alpha = alpha
        self.verbose = verbose
        self.variant = variant
        accepted_variants = ['at', 'se', 'it']
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
        self.coef_, self.theta_ = threshold_fit(
            X, y_tmp, self.alpha, len(self.classes_), 
            mode=self.variant, verbose=self.verbose, 
            max_iter=100000,
            sample_weight=sample_weight)
        return self

    def predict(self, X):
        """

        """
        check_is_fitted(self, ['coef_'])
        X = check_array(X, accept_sparse=True)
        int_class = threshold_predict(X, self.coef_, self.theta_)
        return self.classes_[int_class]

    def predict_proba(self, X):
        """

        """
        check_is_fitted(self, ['coef_'])
        X = check_array(X, accept_sparse=True)
        return threshold_proba(X, self.coef_, self.theta_)
