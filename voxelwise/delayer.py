import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Delayer(BaseEstimator, TransformerMixin):
    """Scikit-learn Transformer to add delays to features.

    This assumes that the samples are ordered in time.
    Adding a delay of 0 corresponds to leave the features unchanged.
    Adding a delay of 1 corresponds to using features from the previous sample.

    Adding multiple delays can be used to take into account the slow
    hemodynamic response, with for example `delays=[1, 2, 3, 4]`.

    Parameters
    ----------
    delays : array-like or None
        Indices of the delays applied to each feature. If multiple values are
        given, each feature is duplicated for each delay.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during the fit.

    Example
    -------
    >>> from sklearn.pipeline import make_pipeline
    >>> from voxelwise.delayer import Delayer
    >>> from himalaya.kernel_ridge import KernelRidgeCV
    >>> pipeline = make_pipeline(Delayer(delays=[1, 2, 3, 4), KernelRidgeCV())
    >>> pipeline.fit(..., ...)
    """
    def __init__(self, delays=[1, 2, 3, 4]):
        self.delays = delays

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                'Different number of features in X than during fit.')

        if self.delays is None:
            return X

        X_delayed = np.zeros((n_samples, n_features * len(self.delays)),
                             dtype=X.dtype)

        for idx, delay in enumerate(self.delays):
            beg, end = idx * n_features, (idx + 1) * n_features
            if delay == 0:
                X_delayed[:, beg:end] = X
            elif delay > 0:
                X_delayed[delay:, beg:end] = X[:-delay]
            elif delay < 0:
                X_delayed[:-abs(delay), beg:end] = X[abs(delay):]

        return X_delayed
