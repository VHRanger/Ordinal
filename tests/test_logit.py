import functools
import numpy as np
import scipy.sparse as sp
from scipy import stats, optimize, sparse
from sklearn import datasets, metrics, linear_model
from sklearn.utils.estimator_checks import check_no_attributes_set_in_init
from sklearn.utils.estimator_checks import check_parameters_default_constructible
from sklearn.utils.testing import assert_greater
import unittest

import ordinal
from ordinal import OrderedLogitRanker
from test_probit import check_predictions
from test_probit import check_ranker_train
from test_probit import _yield_rank_checks

class TestOrderedLogit(unittest.TestCase):
    """

    """
    def test_check_estimator(self):
        """Runs sklearn necessary estimator checks
           Since OrderedProbitRanker is not a proper classifier
           It fails some tests (specifically training perf)
        """
        estimator = OrderedLogitRanker()
        name = OrderedLogitRanker.__name__
        check_parameters_default_constructible(name, OrderedLogitRanker)
        check_no_attributes_set_in_init(name, estimator)
        for check in _yield_rank_checks(name, estimator):
            check(name, estimator)


    def test_predict_2_classes(self):
        """Simple sanity check on a 2 class dataset"""
        # Similar test data as logistic
        # Since this should perform similarly in 2 class case
        _X = [[-1, 0], [0, 1], [1, 1]]
        _X_sp = sp.csr_matrix(_X)
        _Y = [0, 1, 1]
        check_predictions(OrderedLogitRanker(), _X, _Y)
        check_predictions(OrderedLogitRanker(), _X_sp, _Y)


    def test_model_overfit_mode(self):
        np.random.seed(0)
        n_class = 5
        n_samples = 100
        n_dim = 10
        X = np.random.randn(n_samples, n_dim)
        w = np.random.randn(n_dim)
        y = X.dot(w)
        bins = stats.mstats.mquantiles(y, np.linspace(0, 1, n_class + 1))
        y = np.digitize(y, bins[:-1])
        y -= y.min()
        clf2 = OrderedLogitRanker(alpha=0., variant='at')
        clf2.fit(X, y)
        # the score is - absolute error, 0 is perfect
        # assert clf1.score(X, y) < clf2.score(X, y)
        clf3 = OrderedLogitRanker(alpha=0., variant='se')
        clf3.fit(X, y)
        pred3 = clf3.predict(X)
        pred2 = clf2.predict(X)
        ### check that it predicts better than the surrogate for other loss
        assert np.abs(pred2 - y).mean() <= np.abs(pred3 - y).mean()
        ### test on sparse matrices
        X_sparse = sparse.csr_matrix(X)
        clf4 = OrderedLogitRanker(alpha=0., variant='at')
        clf4.fit(X_sparse, y)
        pred4 = clf4.predict(X_sparse)
        assert metrics.mean_absolute_error(y, pred4) < 1.

    def test_grad(self):
        np.random.seed(0)
        n_class = 5
        n_samples = 100
        n_dim = 10
        X = np.random.randn(n_samples, n_dim)
        w = np.random.randn(n_dim)
        y = X.dot(w)
        bins = stats.mstats.mquantiles(y, np.linspace(0, 1, n_class + 1))
        y = np.digitize(y, bins[:-1])
        y -= y.min()
        x0 = np.random.randn(n_dim + n_class - 1)
        x0[n_dim + 1:] = np.abs(x0[n_dim + 1:])
        loss_fd = np.diag(np.ones(n_class - 1)) + \
                np.diag(np.ones(n_class - 2), k=-1)
        loss_fd = np.vstack((loss_fd, np.zeros(n_class - 1)))
        loss_fd[-1, -1] = 1  # border case
        L = np.eye(n_class - 1) - np.diag(np.ones(n_class - 2), k=-1)
        def fun(x, sample_weights=None):
            return ordinal.logit.obj_margin(
                x, X, y, 100.0, n_class, loss_fd, L, sample_weights)
        def grad(x, sample_weights=None):
            return ordinal.logit.grad_margin(
                x, X, y, 100.0, n_class, loss_fd, L, sample_weights)
        assert_greater(
            1e-4,
            optimize.check_grad(fun, grad, x0),
            msg='unweighted')
        sample_weights = np.random.rand(n_samples)
        assert_greater(
            1e-4,
            optimize.check_grad(
                functools.partial(fun, sample_weights=sample_weights),
                functools.partial(grad, sample_weights=sample_weights),
                x0),
            msg='weighted')
