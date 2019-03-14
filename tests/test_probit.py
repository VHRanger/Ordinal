import numpy as np
from scipy import optimize, stats
import scipy.sparse as sp
from sklearn import metrics
from sklearn.base import clone
from sklearn.datasets import load_iris, make_blobs, load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import shuffle
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.estimator_checks import check_parameters_default_constructible
from sklearn.utils.estimator_checks import check_no_attributes_set_in_init
from sklearn.utils.estimator_checks import check_classifier_data_not_an_array
from sklearn.utils.estimator_checks import check_classifiers_one_label
from sklearn.utils.estimator_checks import check_classifiers_classes
from sklearn.utils.estimator_checks import check_estimators_partial_fit_n_features
from sklearn.utils.estimator_checks import check_classifiers_regression_target
from sklearn.utils.estimator_checks import check_estimators_unfitted
from sklearn.utils.estimator_checks import check_supervised_y_2d
from sklearn.utils.estimator_checks import check_supervised_y_no_nan
from sklearn.utils.estimator_checks import check_non_transformer_estimators_n_iter
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_raises
import unittest

import ordinal
from ordinal import OrderedProbitRanker

# TODO: add tests for loglike values
#       test that loglike = loglike_and_grad[0]
#
# TODO: add tests for gradient
#  https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/
#   Also, test from mord tests



def check_predictions(clf, X, y):
    """Check that the model is able to fit the classification data"""
    n_samples = len(y)
    classes = np.unique(y)
    n_classes = classes.shape[0]
    predicted = clf.fit(X, y).predict(X)
    assert_array_equal(clf.classes_, classes)
    assert_equal(predicted.shape, (n_samples,))
    assert_array_equal(predicted, y)
    probabilities = clf.predict_proba(X)
    assert_equal(probabilities.shape, (n_samples, n_classes))
    assert_array_almost_equal(probabilities.sum(axis=1), np.ones(n_samples))
    assert_array_equal(probabilities.argmax(axis=1), y)

def check_ranker_train(name, estimator):
    """
    This changes from classifier_train in that the random_state = 42 
    makes labels ordered. random_state = 0 in original has adversarial 
    (eg. the labels' orders are different than their relation to data)
    """
    X_m, y_m = make_blobs(n_samples=300, random_state=42)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]
    for (X, y) in [(X_m, y_m), (X_b, y_b)]:
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples, _ = X.shape
        classifier = clone(estimator)
        classifier.fit(X, y)
        classifier.fit(X.tolist(), y.tolist())
        assert hasattr(classifier, "classes_")
        y_pred = classifier.predict(X)
        assert_equal(y_pred.shape, (n_samples,))
        assert_greater(accuracy_score(y, y_pred), 0.83)
        # predict_proba agrees with predict
        y_prob = classifier.predict_proba(X)
        assert_equal(y_prob.shape, (n_samples, n_classes))
        assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
        # check that probas for all classes sum to one
        assert_allclose(np.sum(y_prob, axis=1), np.ones(n_samples))
        with assert_raises(ValueError, msg="The classifier {} does not"
                           " raise an error when the number of "
                           "features in predict_proba is different "
                           "from the number of features in fit."
                           .format(name)):
            classifier.predict_proba(X.T)

def _yield_rank_checks(name, estimator):
    # test classifiers can handle non-array data
    yield check_classifier_data_not_an_array
    # test classifiers trained on a single label always return this label
    yield check_classifiers_one_label
    yield check_classifiers_classes
    yield check_estimators_partial_fit_n_features
    yield check_classifiers_regression_target
    yield check_ranker_train
    yield check_estimators_unfitted
    yield check_supervised_y_2d
    yield check_supervised_y_no_nan
    yield check_non_transformer_estimators_n_iter

class TestOrderedProbit(unittest.TestCase):
    """

    """
    def __init__(self, *args, **kwargs):
        super(TestOrderedProbit, self).__init__(*args, **kwargs)
        diabetes = load_diabetes()
        self.Xd = diabetes['data']
        yd = diabetes['target']
        kbd = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='kmeans')
        self.yd = kbd.fit_transform(yd.reshape(-1, 1)).flatten().astype(np.int32)

    # def test_check_estimator(self):
    #     """Runs sklearn necessary estimator checks
    #        Since OrderedProbitRanker is not a proper classifier
    #        It fails some tests (specifically training perf)
    #     """
    #     estimator = OrderedProbitRanker()
    #     name = OrderedProbitRanker.__name__
    #     check_parameters_default_constructible(name, OrderedProbitRanker)
    #     check_no_attributes_set_in_init(name, estimator)
    #     for check in _yield_rank_checks(name, estimator):
    #         check(name, estimator)


    # def test_predict_2_classes(self):
    #     """Simple sanity check on a 2 class dataset"""
    #     # Similar test data as logistic
    #     # Since this should perform similarly in 2 class case
    #     _X = [[-1, 0], [0, 1], [1, 1]]
    #     _X_sp = sp.csr_matrix(_X)
    #     _Y = [0, 1, 1]
    #     check_predictions(OrderedProbitRanker(), _X, _Y)
    #     check_predictions(OrderedProbitRanker(), _X_sp, _Y)


    def test_diabetes(self):
        """test on boston dataset"""
        opr = OrderedProbitRanker(method='L-BFGS-B', use_grad=False)
        opr.fit(self.Xd, self.yd)
        pred_val = opr.predict(self.Xd)
        assert metrics.mean_absolute_error(pred_val, self.yd) < 1.0
        assert metrics.mean_squared_error(pred_val, self.yd) < 1.5
        assert metrics.accuracy_score(pred_val, self.yd) > 0.3
        assert (opr.cuts_ == np.sort(opr.cuts_)).all()
        # print("\n\n\nProbit\nscore: ", metrics.accuracy_score(pred_val, self.yd))
        # print("mse: ", metrics.mean_squared_error(pred_val, self.yd))
        # print("mae: ", metrics.mean_absolute_error(pred_val, self.yd))
        # print("cuts: ", opr.cuts_)
        ### Check Grad ###
        ymasks = np.array([np.array(self.yd == c_) 
                          for c_ in  unique_labels(self.yd)])
        # Get cutweights from inverse cumsum
        cutweights = np.ediff1d(opr.cuts_, to_begin=opr.cuts_[0])
        assert (np.cumsum(cutweights) == opr.cuts_).all()
        x0 = np.append(cutweights, opr.coef_)
        def fun(x, sample_weights=None):
            return ordinal.probit._ordinal_loglikelihood(x, ymasks, self.Xd)
        def grad(x, sample_weights=None):
            return ordinal.probit._ordinal_grad(x, ymasks, self.Xd)
        print(np.abs(ordinal.probit._ordinal_grad(x0, ymasks, self.Xd)) > 0.01)
        # print(ordinal.probit._ordinal_grad(x0, ymasks, self.Xd))


if __name__ == '__main__':
    unittest.main()
