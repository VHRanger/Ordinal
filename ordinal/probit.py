import warnings

import numpy as np
import scipy as sc
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def safe_divide(x1, x2, large_value=1e15):
    """division with denominator possibly being 0"""
    res = np.divide(
        x1, x2,
        out=x1 * large_value, # sign matches
        where=(x2 > 0))
    res[x1 == 0] = 0
    return res


def assert_is_ascending_ordered(classlist):
    """
    Asserts that input is strictly ascending, ordered, transitive.
    Uses "<" operator for ordering
    """
    for i in range(len(classlist)-1):
        if not classlist[i] < classlist[i+1]:
            raise ValueError(
                """Input is not correctly ordered on Elements {0} and {1}
                """.format(str(classlist[i]), str(classlist[i+1])))


def _ordinal_loglikelihood(betas, ymasks, X):
    """
    Log Likelihood function of ordered choice model.

    References
    ----------
    Ordered choice models (Greene, William) chpt. 5
    http://pages.stern.nyu.edu/~wgreene/DiscreteChoice/Readings/OrderedChoiceSurvey.pdf
    
    see section 3.5 (p.89, eq. 3.6) for log likelihood function
    """
    n_cuts = ymasks.shape[0] - 1
    xb = X @ betas[n_cuts:]
    # bottom and top cutpoints are -inf and inf
    # cumsum ensures cutpoints remain ordered
    cuts = np.hstack((-np.inf, 
                        np.cumsum(betas[:n_cuts]), 
                        np.inf))
    # Get the distribution's area between each cutpoint
    # expr x[:,None] - xb outputs shape (n_class - 1, n_samples)
    cdf_areas = norm.cdf(cuts[:, None] - xb)
    dist_areas = np.diff(cdf_areas, axis=0)
    res = np.sum(ymasks * dist_areas, axis=0)
    res = np.sum(np.log(res))
    return -res


def _ordinal_grad(betas, ymasks, X):
    """
    ref:
    http://pages.stern.nyu.edu/~wgreene/DiscreteChoice/Readings/OrderedChoiceSurvey.pdf
    section 5.9.5 (p. 134)

    Gradient calculations are on equation 5.15
    """
    n_cuts = ymasks.shape[0] - 1
    xb = X @ betas[n_cuts:]
    # bottom and top cutpoints are -inf and inf
    # cumsum ensures cutpoints remain ordered
    cuts = np.hstack((-np.inf, 
                        np.cumsum(betas[:n_cuts]), 
                        np.inf))
    cdf_areas = norm.cdf(cuts[:, None] - xb)
    dist_areas = np.diff(cdf_areas, axis=0)
    grad = np.empty_like(betas)
    pdf_areas = norm.pdf(cuts[:, None] - xb)
    grad_areas = np.diff(pdf_areas, axis=0)
    # pdf / cdf areas on each observations
    # output shape: [n_classes, n_samples]
    grad_areas = safe_divide(grad_areas, dist_areas)
    grad_areas = np.multiply(grad_areas, ymasks)
    grad_areas = grad_areas @ -X # [n_classes, n_features]
    # sum over resulting classes for each feature's gradient
    grad[n_cuts:] = grad_areas.sum(axis=0)
    #
    # cutoff point gradient
    #
    cut_areas = norm.pdf(cuts[:, None] - xb)
    
    print(cut_areas, "\n\n\n")
    
    tmp = (ymasks / dist_areas)

    for i in range(n_cuts-1):
        cut_areas[i] = (
            cut_areas[i]
            * (safe_divide(ymasks[i], dist_areas[i]) 
                - safe_divide(ymasks[i+1], dist_areas[i+1]))
        )

    # Last one has sign flipped because = (0 - pdf)
    cut_areas[n_cuts-1] = (
        cut_areas[n_cuts-1] 
        * (safe_divide(ymasks[n_cuts-1], dist_areas[n_cuts-1])
            + safe_divide(ymasks[n_cuts], dist_areas[n_cuts]))
    )
    grad[:n_cuts] = cut_areas[1:-1].sum(axis=1)
    return -grad


class OrderedProbitRanker(BaseEstimator, ClassifierMixin):
    """ Ordered probit ranking classifier

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
        TO UPDATE

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        The unique classes seen at :meth:`fit`. 
        The classes must have a transitive ordering with the `<` operator
        The `classes_` attribute will be in ascending order by `<` operator
    """
    def __init__(self, method='L-BFGS-B', use_grad=False, verbose=0):
        self.method=method
        self.use_grad=use_grad
        self.verbose=verbose


    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # TODO: drop perfect predicting features (see 5.9.2 in reference)
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, accept_sparse='csr',
                         dtype=np.float64, order="C",)
        _, n_features = X.shape
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        n_class = len(self.classes_)
        assert_is_ascending_ordered(self.classes_)
        # masks_ is an array of boolean mask arrays for each category
        masks_ = np.array([np.array(y == c_) for c_ in self.classes_])
        # hack around sklearn.check_estimator test
        if n_class == 1:
            raise ValueError("Can't fit to one class")
        n_cuts = n_class - 1
        # coefs are arranged as [cutpoints, feature coefs, intercept]
        self.coef_ = np.zeros(n_features + n_cuts)
        # TODO: add smarter cutpoint init values
        #       based on quantiles of y and inverse cdf
        # cutpoints must be ascending ordered
        self.coef_[:n_cuts] = np.linspace(0, 2, n_cuts)
        self.coef_[0] = -3
        # Lower Bound the cutoff points at 0
        bounds = ([(None, None)] # First cutoff can be negative
                + [(0, None)] * (n_cuts - 1) # other cutoffs only add in cumsum
                + [(None, None)] * n_features) # features are unbounded
        if self.use_grad:
            optres = sc.optimize.minimize(
                fun=_ordinal_loglikelihood,
                x0=self.coef_,
                args=(masks_, X),
                bounds=bounds,
                method=self.method,
                jac=_ordinal_grad,
                options={"disp":self.verbose, 'maxiter':50000, "maxfun":150000})
        else:
            optres = sc.optimize.minimize(
                fun=_ordinal_loglikelihood,
                x0=self.coef_,
                args=(masks_, X),
                bounds=bounds,
                method=self.method,
                jac=False,
                options={"disp":self.verbose, 'maxiter':50000, "maxfun":150000})
        if not optres.success:
            warnings.warn(
                "\nOrdered Probit Optimization did not succeed" +
                "\nTermination Status: " + str(optres.status) + 
                "\nTermination Message: " + str(optres.message) +
                "\nfunction Value:" + str(optres.fun) +
                "\nIterations:" + str(optres.nit),
                RuntimeWarning)
        self.coef_ = optres.x[n_cuts:]
        self.cuts_ = np.cumsum(optres.x[:n_cuts])
        return self


    def predict(self, X):
        """ Returns predicted category

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The argmax of the predicted probability of each class
        """
        check_is_fitted(self, ['coef_'])
        y_pred = self.predict_proba(X)
        res_idx = np.argmax(y_pred, axis=1)
        return self.classes_[res_idx]


    def predict_proba(self, X):
        """ Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Logistic uses a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.

        Reference
        ---------
        http://pages.stern.nyu.edu/~wgreene/DiscreteChoice/Readings/OrderedChoiceSurvey.pdf
        section 5.7 (Prediction - Computing Probabilities)
        The predicted probability of a class j for an input vector x is

            P_j(x) = F(mu_j - Beta'x) - F(mu_j-1 - Beta'x)

        Where F is the cdf of the distribution, mu is the cutoff point 
        for a category and beta are the extimated coefficients. Note that 
        the bottom cutoff has mu=0 and the top cutoff has mu=1.
        This is the same computation as in the log likelihood function.
        """
        check_is_fitted(self, ['coef_'])
        X = check_array(X, accept_sparse=True)
        if sc.sparse.issparse(X):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)
        n_cuts = len(self.classes_) - 1
        xb = X @ self.coef_
        # pre allocate result array
        pred = np.zeros((n_samples, len(self.classes_)))
        # TODO: optimize (by vectorizing)
        for i in range(n_samples):
            # cdf areas between cutpoints
            cdf_areas = [norm.cdf(c - xb[i]) for c in self.cuts_]
            # last cdf area is from last cutpoint on
            cdf_areas.append(cdf_areas[-1])
            cdf_areas = np.array(cdf_areas)
            # first is area up to cutpoint 0
            pred[i][0] = cdf_areas[0]
            # last area is 1 - last cutpoint
            pred[i][-1] = 1 - cdf_areas[-1]
            # middle cuts are cdf area between each
            for j in range(1, n_cuts):
                pred[i][j] = cdf_areas[j] - cdf_areas[j-1]
        return pred
