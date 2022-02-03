"""
================================================
Understand ridge regression and cross-validation
================================================

In the following examples, we will model the fMRI responses using a regularized
linear regression known as ridge regression. Before building any model, the
present example explains why we use ridge regression, and how to use
cross-validation to select the appropriate regularization hyper-parameter.

Linear regression is a method to model an output variable ``y`` (or target)
using a linear combination of some input variables ``X`` (or features). For
each sample ``X_i`` in ``X``, the model predicts an output ``y_i* = X_i @ w``,
where ``w`` is a vector of coefficients, and ``@`` is the dot product. The
model is considered accurate if the prediction ``y_i*`` is close to the true
value ``y_i``. Therefore, given a dataset ``(X, y)``, a good linear regression
model is given by the coefficients ``b`` that minimizes the sum of squared
errors: ``||X_i @ w - y_i||^2 = sum_i (X_i @ w - y_i)^2``. This particular
model is called "ordinary least squares" (OLS).
"""
# sphinx_gallery_thumbnail_number = 3
###############################################################################
# Ordinary least squares (OLS)
# ----------------------------
#
# To illustrate OLS, let's use a toy dataset with a single features x1. On the
# plot below, each dot is a sample (X_i, y_i), and the linear regression model
# is the line ``y = x1 @ w1``. On each sample, the error between the prediction
# and the true value is shown by a gray line. By summing the squared errors
# over all samples, we get a particular value of the squared loss function.
from voxelwise_tutorials.regression_toy import create_regression_toy
from voxelwise_tutorials.regression_toy import plot_1d

X, y = create_regression_toy(n_features=1)
plot_1d(X, y, coefs=[0])

###############################################################################
# By varying the linear coefficient ``w1``, we can change the prediction
# accuracy of the model, and thus the squared loss.
plot_1d(X, y, coefs=[0.2])
plot_1d(X, y, coefs=[0.7])

###############################################################################
# The linear coefficient leading to the minimum squared loss can be found
# analytically by the following formula: ``w = (X.T @ X)^-1 @ X.T @ y``.
# This is the OLS solution.
import numpy as np

coefs_ols = np.linalg.solve(X.T @ X, X.T @ y)
plot_1d(X, y, coefs=coefs_ols)

###############################################################################
# Linear regression can also be used on more than one feature. On the next
# toy dataset, we will use two features x1 and x2. The linear regression model
# is a now plane. Here again, summing the squared errors over all samples gives
# the squared loss.
from voxelwise_tutorials.regression_toy import create_regression_toy
from voxelwise_tutorials.regression_toy import plot_2d

X, y = create_regression_toy(n_features=2)

coefs_ols = np.linalg.solve(X.T @ X, X.T @ y)

plot_2d(X, y, coefs=[0, 0], show_noiseless=False)
plot_2d(X, y, coefs=[0.25, 0], show_noiseless=False)
plot_2d(X, y, coefs=[0.25, 0.15], show_noiseless=False)

###############################################################################
# Here again, the OLS solution can be found analytically with the same formula.
# Note that the OLS solution is not equal to the ground truth coefficients used
# to generate the toy dataset (black cross), because we added some noise to the
# target values ``y``.
coefs_ols = np.linalg.solve(X.T @ X, X.T @ y)
plot_2d(X, y, coefs=coefs_ols)

###############################################################################
# The situation becomes more interesting when the features ``X```are
# correlated. Here, we add a correlation between the first feature ``X[:, 0]``
# and the second feature ``X[:, 1]``. With this correlation, the squared loss
# function is no more isotropic, so the levels of equal loss are now ellipses.
# Starting from the OLS solution, a small change in ``w`` to the top left leads
# to a small change in the loss, whereas a small change to the top right leads
# to a large change in the loss. This anisotropy makes the OLS solution less
# robust to noise in some directions.

X, y = create_regression_toy(n_features=2, correlation=0.9)

coefs_ols = np.linalg.solve(X.T @ X, X.T @ y)
plot_2d(X, y, coefs=coefs_ols)

###############################################################################
# If the correlation is even higher, the loss function is even more
# anisotropic, and the OLS solution becomes even less stable. This can be
# understood mathematically by the fact that the OLS solution requires
# inverting the matrix ``X.T @ X``, which amounts to inverting the eigenvalues
# ``lambda_k`` of the matrix. When the features are highly correlated, some
# eigenvalues are close to zero, which reduces the stability of the inversion
# (a small change in the features can have a large effect on the result).

X, y = create_regression_toy(n_features=2, correlation=0.99)

coefs_ols = np.linalg.solve(X.T @ X, X.T @ y)
plot_2d(X, y, coefs=coefs_ols)


###############################################################################
# Ridge regression
# ----------------
#
# To solve this instability, OLS can be extended to ridge regression. Ridge
# regression considers a different optimization problem, which optimizes the
# sum of the squared loss function ``||X_i @ w - y_i||^2`` and of a
# regularization term ``alpha * ||w||_2``. The ridge solution finds a balance
# between being close to the OLS solution and being close to the origin (``w =
# 0``). In the regularization term, ``alpha`` is a positive hyperparameter that
# controls the regularization strength. With a small ``alpha``, the solution
# will be close to the OLS solution, and with a large ``alpha``, the solution
# will be further from the OLS solution and closer to the origin.
#
# To illustrate this, the following plot shows the ridge solution for a
# particular value of ``alpha``. The black circle shows the different values of
# ``w`` leading to the same regularization value, while the blue ellipses show
# the different different values of ``w`` leading to the same squared loss
# value.

X, y = create_regression_toy(n_features=2, correlation=0.9)

alpha = 23
coefs_ridge = np.linalg.solve(X.T @ X + np.eye(X.shape[1]) * alpha, X.T @ y)
plot_2d(X, y, coefs_ridge, alpha=alpha)

###############################################################################
# Adding a regularization term makes the solution more robust to noise. Indeed,
# the ridge solution can be found analytically with the following formula: ``w
# = (X.T @ X + alpha * I)^-1 @ X.T @ y``. In this formula, we can see that the
# inverted matrix is now ``X.T @ X + alpha * I``, which adds a positive value
# ``alpha`` to all eigenvalues ``lambda_i `` of ``X^TX`` before the matrix
# inversion. Inverting ``(lambda_i + alpha)`` instead of ``lambda_i`` reduces
# the instability caused by small eigenvalues. This makes the ridge solution
# more robust to noise than the OLS solution.
#
# Here, we can see that with even more correlation features, the ridge solution
# is still reasonably close to the noiseless ground truth, while the OLS
# solution would be far off.

X, y = create_regression_toy(n_features=2, correlation=0.999)

alpha = 23
coefs_ridge = np.linalg.solve(X.T @ X + np.eye(X.shape[1]) * alpha, X.T @ y)
plot_2d(X, y, coefs_ridge, alpha=alpha)

###############################################################################
# Hyperparameter selection
# ------------------------
#


###############################################################################
# Kernel ridge regression
# -----------------------
#
