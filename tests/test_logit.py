import functools
import numpy as np
import scipy.sparse as sp
from scipy import stats, optimize, sparse
from sklearn import datasets, metrics, linear_model
from sklearn.base import clone
from sklearn.datasets import load_iris, make_blobs, load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
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
    def __init__(self, *args, **kwargs):
        super(TestOrderedLogit, self).__init__(*args, **kwargs)
        diabetes = load_diabetes()
        self.Xd = diabetes['data']
        yd = diabetes['target']
        kbd = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='kmeans')
        self.yd = kbd.fit_transform(yd.reshape(-1, 1)).flatten().astype(np.int32)

    def test_diabetes_at(self):
        """test on boston dataset"""
        opr = OrderedLogitRanker(variant='at')
        opr.fit(self.Xd, self.yd)
        pred_val = opr.predict(self.Xd)
        assert metrics.mean_absolute_error(pred_val, self.yd) < 1.0
        assert metrics.mean_squared_error(pred_val, self.yd) < 1.5
        assert (opr.cuts_ == np.sort(opr.cuts_)).all()
        print("\n\n\nAT\nscore: ", metrics.accuracy_score(pred_val, self.yd))
        print("mse: ", metrics.mean_squared_error(pred_val, self.yd))
        print("mae: ", metrics.mean_absolute_error(pred_val, self.yd))
        print("cuts: ", opr.cuts_)

    def test_diabetes_se(self):
        """test on boston dataset"""
        opr = OrderedLogitRanker(variant='se')
        opr.fit(self.Xd, self.yd)
        pred_val = opr.predict(self.Xd)
        assert metrics.mean_absolute_error(pred_val, self.yd) < 1.0
        assert metrics.mean_squared_error(pred_val, self.yd) < 1.5
        assert (opr.cuts_ == np.sort(opr.cuts_)).all()
        print("\n\n\nSE\nscore: ", metrics.accuracy_score(pred_val, self.yd))
        print("mse: ", metrics.mean_squared_error(pred_val, self.yd))
        print("mae: ", metrics.mean_absolute_error(pred_val, self.yd))
        print("cuts: ", opr.cuts_)


    def test_check_estimatorAT(self):
        """Runs sklearn necessary estimator checks
           Since OrderedProbitRanker is not a proper classifier
           It fails some tests (specifically training perf)

        """
        # All Threshold Variant
        estimatorat = OrderedLogitRanker(variant='at')
        nameat = OrderedLogitRanker.__name__ + "_Variant: AT"
        check_parameters_default_constructible(nameat, OrderedLogitRanker)
        check_no_attributes_set_in_init(nameat, estimatorat)
        for check in _yield_rank_checks(nameat, estimatorat):
            check(nameat, estimatorat)

    def test_check_estimatorIT(self):
        """Runs sklearn necessary estimator checks
           Since OrderedProbitRanker is not a proper classifier
           It fails some tests (specifically training perf)
        """
        # Immediate Threshold Variant
        estimatorit = OrderedLogitRanker(variant='it')
        nameit = OrderedLogitRanker.__name__ + "_Variant: IT"
        check_parameters_default_constructible(nameit, OrderedLogitRanker)
        check_no_attributes_set_in_init(nameit, estimatorit)
        for check in _yield_rank_checks(nameit, estimatorit):
            check(nameit, estimatorit)

    def test_check_estimatorSE(self):
        """Runs sklearn necessary estimator checks
           Since OrderedProbitRanker is not a proper classifier
           It fails some tests (specifically training perf)
        """
        # Squared Error Variant
        estimatorse = OrderedLogitRanker(variant='se')
        namese = OrderedLogitRanker.__name__ + "_Variant: SE"
        check_parameters_default_constructible(namese, OrderedLogitRanker)
        check_no_attributes_set_in_init(namese, estimatorse)
        for check in _yield_rank_checks(namese, estimatorse):
            check(namese, estimatorse)

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
        # L = np.eye(n_class - 1) - np.diag(np.ones(n_class - 2), k=-1)
        L = np.zeros((n_class - 1, n_class - 1))
        L[np.tril_indices(n_class-1)] = 1.
        def fun(x, sample_weights=None):
            return ordinal.logit.obj_margin(
                x, X, y, 100.0, n_class, loss_fd, sample_weights)
        def grad(x, sample_weights=None):
            return ordinal.logit.grad_margin(
                x, X, y, 100.0, n_class, loss_fd, sample_weights)
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
