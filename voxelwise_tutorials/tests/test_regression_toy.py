import numpy as np
import matplotlib.pyplot as plt

from voxelwise_tutorials.regression_toy import create_regression_toy
from voxelwise_tutorials.regression_toy import plot_1d
from voxelwise_tutorials.regression_toy import plot_2d
from voxelwise_tutorials.regression_toy import plot_kfold2
from voxelwise_tutorials.regression_toy import plot_cv_path


def test_smoke_regression_toy():
    """Follow tutorials/shortclips/02_plot_ridge_regression.py."""
    X, y = create_regression_toy(n_samples=50, n_features=1)
    plot_1d(X, y, w=[0])
    plt.close('all')
    w_ols = np.linalg.solve(X.T @ X, X.T @ y)
    plot_1d(X, y, w=w_ols)
    plt.close('all')

    X, y = create_regression_toy(n_features=2)
    plot_2d(X, y, w=[0, 0], show_noiseless=False)
    plt.close('all')
    plot_2d(X, y, w=[0.4, 0], show_noiseless=False)
    plt.close('all')
    plot_2d(X, y, w=[0, 0.3], show_noiseless=False)
    plt.close('all')

    w_ols = np.linalg.solve(X.T @ X, X.T @ y)
    plot_2d(X, y, w=w_ols)
    plt.close('all')

    X, y = create_regression_toy(n_features=2, correlation=0.9)
    w_ols = np.linalg.solve(X.T @ X, X.T @ y)
    plot_2d(X, y, w=w_ols)
    plt.close('all')

    X, y = create_regression_toy(n_features=2, correlation=0.9)
    alpha = 23
    w_ridge = np.linalg.solve(X.T @ X + np.eye(X.shape[1]) * alpha, X.T @ y)
    plot_2d(X, y, w_ridge, alpha=alpha)
    plt.close('all')

    X, y = create_regression_toy(n_features=1)
    plot_kfold2(X, y, fit=False)
    plt.close('all')
    alpha = 0.1
    plot_kfold2(X, y, alpha, fit=True)
    plt.close('all')
    plot_kfold2(X, y, alpha, fit=True, flip=True)
    plt.close('all')

    noise = 0.1
    X, y = create_regression_toy(n_features=2, noise=noise)
    plot_cv_path(X, y)
    plt.close('all')
