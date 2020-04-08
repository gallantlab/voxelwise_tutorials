import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Delayer(BaseEstimator, TransformerMixin):
    """Transformer to add delays to features.
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
