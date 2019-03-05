import numpy as np
import scipy as sc
import warnings
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


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

    @staticmethod
    def _orderedProbitLogLike(betas, ymasks, X):
        """
        Log Likelihood function of ordered probit model.

        This is the function that actually gets optimized over.

        Example of the function  with 5 classes
        ------------------------------------------------------
        # betas is [cutoffs] + [parameters]
        c0, c1, c2, c3 = betas[:3]  # isolate cutoff point parameters
        xb = np.dot(X, betas[4:])   # multiply params with X
        llf = np.sum(np.log(
                ((y==0) * (norm.cdf(c0 - xb))) +
                ((y==1) * (norm.cdf(c1 - xb) - norm.cdf(c0 - xb))) +
                ((y==2) * (norm.cdf(c2 - xb) - norm.cdf(c1 - xb))) +
                ((y==3) * (norm.cdf(c3 - xb) - norm.cdf(c2- xb))) +
                ((y==4) * (1 - norm.cdf(c3- xb)))
             ))
        return -llf   # return negative to minimize over
        ------------------------------------------------------
        ref: ordered choice models (Greene, William) chpt. 5
        http://pages.stern.nyu.edu/~wgreene/DiscreteChoice/Readings/OrderedChoiceSurvey.pdf
        
        see section 3.5 (p.89, eq. 3.6) for log likelihood function
        """
        # assumes ymasks come from fit() method
        # eg. 2d array with rows being masks ordered by classes
        # 1 fewer cutpoints than categories
        n_samples = ymasks.shape[1]
        n_cuts = ymasks.shape[0] - 1
        # TODO: xb can "explode" out of normal cdf bounds
        #       eg. values above 8 and below -8 have cdf of 0 and 1 
        #       regardless of cutoff point
        xb = X @ betas[n_cuts:]
        # ensure cutpoints remain ordered
        # TODO: This can be done by reparametrizing the cutpoints...
        cuts = np.sort(betas[:n_cuts])
        # cdf up to cutpoints
        cdf_areas = [norm.cdf(ct - xb) for ct in cuts]
        # last cdf area is from last cutpoint on
        cdf_areas.append(cdf_areas[-1])
        cdf_areas = np.array(cdf_areas)
        # pdf areas between cutpoints = cdf[i] - cdf[i-1]
        pdf_areas = np.empty_like(ymasks, dtype='float')
        # first is cdf[cut_0] - 0
        pdf_areas[0] = cdf_areas[0]
        # last is 1 - cdf[last_cut]
        pdf_areas[-1] = 1 - cdf_areas[-1]
        # middle cuts are cdf area between each
        for i in range(1, n_cuts):
            pdf_areas[i] = cdf_areas[i] - cdf_areas[i-1]
        res = np.zeros(n_samples)
        for i in range(len(ymasks)):
            res += (ymasks[i] * pdf_areas[i])
        res = np.sum(np.log(res))
        return -res

    @staticmethod
    def _ordered_probit_loss_and_grad(betas, ymasks, X):
        """
        ref:
        http://pages.stern.nyu.edu/~wgreene/DiscreteChoice/Readings/OrderedChoiceSurvey.pdf
        section 5.9.5 (p. 134)

        Gradient calculations are on equation 5.15
        """
        # assumes ymasks come from fit() method
        # eg. 2d array with rows being masks ordered by classes
        # 1 fewer cutpoints than categories
        n_samples = ymasks.shape[1]
        n_cuts = ymasks.shape[0] - 1
        # TODO: xb can "explode" out of normal cdf bounds
        #       eg. values above 8 and below -8 have cdf of 0 and 1 
        #       regardless of cutoff point
        xb = X @ betas[n_cuts:]
        # ensure cutpoints remain ordered
        # TODO: This can be done by reparametrizing the cutpoints...
        cuts = np.sort(betas[:n_cuts])
        # cdf up to cutpoints
        cdf_areas = [norm.cdf(ct - xb) for ct in cuts]
        # last cdf area is from last cutpoint on
        cdf_areas.append(cdf_areas[-1])
        cdf_areas = np.array(cdf_areas)
        # pdf areas between cutpoints = cdf[i] - cdf[i-1]
        pdf_areas = np.empty_like(ymasks, dtype='float')
        # first is cdf[cut_0] - 0
        pdf_areas[0] = cdf_areas[0]
        # last is 1 - cdf[last_cut]
        pdf_areas[-1] = 1 - cdf_areas[-1]
        # middle cuts are cdf area between each
        for i in range(1, n_cuts):
            pdf_areas[i] = cdf_areas[i] - cdf_areas[i-1]
        res = np.zeros(n_samples)
        for i in range(len(ymasks)):
            res += (ymasks[i] * pdf_areas[i])
        res = np.sum(np.log(res))
        #
        # Now calculate gradient
        #
        grad = np.empty_like(betas)
        grad_areas = [norm.pdf(ct - xb) for ct in cuts]
        # pdf of inf is 0 but abs(0 - x) = x, so copy values
        grad_areas.append(grad_areas[-1])
        grad_areas = np.array(grad_areas)
        cut_areas = np.array(grad_areas)
        for i in range(1, n_cuts):
            grad_areas[i] = grad_areas[i] - grad_areas[i-1]
        # pdf / cdf areas on each observations
        # output shape: [n_classes, n_samples]
        grad_areas = safe_divide(grad_areas, pdf_areas)
        grad_areas = np.multiply(grad_areas, ymasks)
        grad_areas = grad_areas @ -X # [n_classes, n_features]
        # sum over resulting classes for each feature's gradient
        grad[n_cuts:] = grad_areas.sum(axis=0)
        #
        # cutoff point gradient
        #
        for i in range(n_cuts-1):
            cut_areas[i] = (
                cut_areas[i] 
                * (safe_divide(ymasks[i], pdf_areas[i]) 
                    - safe_divide(ymasks[i+1], pdf_areas[i+1]))
            )
        # Last one has sign flipped because = (0 - pdf)
        cut_areas[n_cuts-1] = (
            cut_areas[n_cuts-1] 
            * (safe_divide(ymasks[n_cuts-1], pdf_areas[n_cuts-1])
               + safe_divide(ymasks[n_cuts], pdf_areas[n_cuts]))
        )
        grad[:n_cuts] = cut_areas[:-1].sum(axis=1)
        return -res, -grad


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
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        assert_is_ascending_ordered(self.classes_)
        # masks_ is an array of boolean mask arrays for each category
        masks_ = np.array([np.array(y == c_) for c_ in self.classes_])
        # hack around sklearn.check_estimator test
        if masks_.shape[0] == 1:
            raise ValueError("Can't fit to one class")
        n_cuts = masks_.shape[0] - 1
        # coefs are arranged as [cutpoints, feature coefs, intercept]
        self.coef_ = np.zeros(X.shape[1] + n_cuts)
        # TODO: add smarter cutpoint init values
        #       based on quantiles of y and inverse cdf
        # cutpoints must be ascending ordered
        self.coef_[:n_cuts] = np.linspace(-3.5, 3.5, n_cuts)
        if self.use_grad:
            optres = sc.optimize.minimize(
                fun=self._ordered_probit_loss_and_grad,
                x0=self.coef_,
                args=(masks_, X),
                method=self.method,
                jac=True,
                options={"disp":self.verbose, 'maxiter':50000, "maxfun":150000})
        else:
            optres = sc.optimize.minimize(
                fun=self._orderedProbitLogLike,
                x0=self.coef_,
                args=(masks_, X),
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
        self.cuts_ = optres.x[:n_cuts]
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
