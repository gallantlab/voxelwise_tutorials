"""
=====================================================================
Fit a banded ridge model with both wordnet and motion energy features
=====================================================================

In this example, we model the fMRI responses with a `banded ridge regression`,
with two different feature spaces: motion energy, and wordnet categories.

*Banded ridge regression:* Since the relative scaling of both feature spaces is
unknown, we use two regularization hyperparameters (one per feature space) in a
model called banded ridge regression [1]_. Just like with ridge regression, we
optimize the hyperparameters over cross-validation. An efficient implementation
of this model is available in the `himalaya
<https://github.com/gallantlab/himalaya>`_ package.

*Running time:* This example is more computationally intensive than previous
examples. With a GPU backend, the fitting of this model takes around 6 minutes.
With a CPU backend, it can last 10 times more.
"""
# sphinx_gallery_thumbnail_number = 2
###############################################################################
# Path of the data directory
import os
from voxelwise_tutorials.io import get_data_home
directory = os.path.join(get_data_home(), "vim-5")
print(directory)

###############################################################################

# modify to use another subject
subject = "S01"

###############################################################################
# Load the data
# -------------
#
# As in the previous examples, we first load the fMRI responses, which are our
# regression targets.
import numpy as np

from voxelwise_tutorials.io import load_hdf5_array

file_name = os.path.join(directory, "responses", f"{subject}_responses.hdf")
Y_train = load_hdf5_array(file_name, key="Y_train")
Y_test = load_hdf5_array(file_name, key="Y_test")

print("(n_samples_train, n_voxels) =", Y_train.shape)
print("(n_repeats, n_samples_test, n_voxels) =", Y_test.shape)

###############################################################################
# We average the test repeats, to remove the non-repeatable part of fMRI
# responses.
Y_test = Y_test.mean(0)

print("(n_samples_test, n_voxels) =", Y_test.shape)

###############################################################################
# We fill potential NaN (not-a-number) values with zeros.
Y_train = np.nan_to_num(Y_train)
Y_test = np.nan_to_num(Y_test)

###############################################################################
# And we make sure the targets are centered.
Y_train -= Y_train.mean(0)
Y_test -= Y_test.mean(0)

###############################################################################
# Then we load both feature spaces, that are going to be used for the
# linear regression model.

feature_names = ["wordnet", "motion_energy"]

Xs_train = []
Xs_test = []
n_features_list = []
for feature_space in feature_names:
    file_name = os.path.join(directory, "features", f"{feature_space}.hdf")
    Xi_train = load_hdf5_array(file_name, key="X_train")
    Xi_test = load_hdf5_array(file_name, key="X_test")

    Xs_train.append(Xi_train.astype(dtype="float32"))
    Xs_test.append(Xi_test.astype(dtype="float32"))
    n_features_list.append(Xi_train.shape[1])

# concatenate the feature spaces
X_train = np.concatenate(Xs_train, 1)
X_test = np.concatenate(Xs_test, 1)

print("(n_samples_train, n_features_total) =", X_train.shape)
print("(n_samples_test, n_features_total) =", X_test.shape)
print("[n_features_wordnet, n_features_motion_energy] =", n_features_list)

###############################################################################
# Define the cross-validation scheme
# ----------------------------------
#
# We define again a leave-one-run-out cross-validation split scheme.

from sklearn.model_selection import check_cv
from voxelwise_tutorials.utils import generate_leave_one_run_out

# indice of first sample of each run
run_onsets = load_hdf5_array(file_name, key="run_onsets")
print(run_onsets)

###############################################################################
# We define a cross-validation splitter, compatible with ``scikit-learn`` API.
n_samples_train = X_train.shape[0]
cv = generate_leave_one_run_out(n_samples_train, run_onsets)
cv = check_cv(cv)  # copy the cross-validation splitter into a reusable list

###############################################################################
# Define the model
# ----------------
#
# The model pipeline contains similar steps than the pipeline from previous
# examples. We remove the mean of each feature with a ``StandardScaler``,
# and add delays with a ``Delayer``.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer
from himalaya.backend import set_backend
backend = set_backend("torch_cuda", on_error="warn")

###############################################################################
# To fit the banded ridge model, we use ``himalaya``'s
# ``MultipleKernelRidgeCV`` model, with a separate linear kernel per feature
# space. Similarly to ``KernelRidgeCV``, the model optimizes the
# hyperparameters over cross-validation. However, while ``KernelRidgeCV`` has
# to optimize only one hyperparameter (``alpha``), ``MultipleKernelRidgeCV``
# has to optimize ``m`` hyperparameters, where ``m`` is the number of feature
# spaces (here ``m = 2``). To do so, the model implements two different
# solvers, one using hyperparameter random search, and one using hyperparameter
# gradient descent. For large number of targets, we recommend using the
# random-search solver.

###############################################################################
# The class takes a number of common parameters during initialization, such as
# ``kernels``, or ``solver``. Since the solver parameters are rather different
# depending on the solver, they are passed in a ``solver_params`` dictionary
# parameter.

from himalaya.kernel_ridge import MultipleKernelRidgeCV

# Here we will use the "random_search" solver.
solver = "random_search"

# We can check its specific parameters in the function docstring:
solver_function = MultipleKernelRidgeCV.ALL_SOLVERS[solver]
print("Docstring of the function %s:" % solver_function.__name__)
print(solver_function.__doc__)

###############################################################################
# The hyperparameter random-search solver separates the hyperparameters into a
# shared regularization ``alpha`` and a vector of positive kernel weights which
# sum to one. This separation of hyperparameters allows to explore efficiently
# a large grid of values for ``alpha`` for each sampled kernel weights vector.
#
# We use *20* random-search iterations to have a reasonably fast example. To
# have better results, especially for larger number of feature spaces, one
# might need more iterations. (Note that there is currently no stopping
# criterion in the random-search method.)
n_iter = 20

alphas = np.logspace(1, 20, 20)

###############################################################################
# Batch parameters, used to reduce the necessary GPU memory. A larger value
# will be a bit faster, but the solver might crash if it is out of memory.
# Optimal values depend on the size of your dataset.
n_targets_batch = 200
n_alphas_batch = 5
n_targets_batch_refit = 200

###############################################################################
# We put all these parameters in a dictionary ``solver_params``, and define
# the main estimator ``MultipleKernelRidgeCV``.

solver_params = dict(n_iter=n_iter, alphas=alphas,
                     n_targets_batch=n_targets_batch,
                     n_alphas_batch=n_alphas_batch,
                     n_targets_batch_refit=n_targets_batch_refit)

mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver=solver,
                                  solver_params=solver_params, cv=cv)

###############################################################################
# We need a bit more work than in previous examples before defining the full
# pipeline, since the banded ridge model requires `multiple` precomputed
# kernels, one for each feature space. To compute them, we use the
# ``ColumnKernelizer``, which can create multiple kernels from different
# column of your features array. ``ColumnKernelizer`` works similarly to
# ``scikit-learn``'s ``ColumnTransformer``, but instead of returning a
# concatenation of transformed features, it returns a stack of kernels,
# as required in ``MultipleKernelRidgeCV(kernels="precomputed")``.

###############################################################################
# First, we create a different ``Kernelizer`` for each feature space.
# Here we use a linear kernel for all feature spaces, but ``ColumnKernelizer``
# accepts any ``Kernelizer``, or ``scikit-learn`` ``Pipeline`` ending with a
# ``Kernelizer``.
from himalaya.kernel_ridge import Kernelizer
from sklearn import set_config
set_config(display='diagram')  # requires scikit-learn 0.23

preprocess_pipeline = make_pipeline(
    StandardScaler(with_mean=True, with_std=False),
    Delayer(delays=[1, 2, 3, 4]),
    Kernelizer(kernel="linear"),
)
preprocess_pipeline

###############################################################################
# The column kernelizer applies a different pipeline on each selection of
# features, here defined with ``slices``.
from himalaya.kernel_ridge import ColumnKernelizer

# Find the start and end of each feature space in the concatenated ``X_train``.
start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
slices = [
    slice(start, end)
    for start, end in zip(start_and_end[:-1], start_and_end[1:])
]
slices

###############################################################################
kernelizers_tuples = [(name, preprocess_pipeline, slice_)
                      for name, slice_ in zip(feature_names, slices)]
column_kernelizer = ColumnKernelizer(kernelizers_tuples)
column_kernelizer

# (Note that ``ColumnKernelizer`` has a parameter ``n_jobs`` to parallelize
# each ``Kernelizer``, yet such parallelism does not work with GPU arrays.)

###############################################################################
# Then we can define the model pipeline.

pipeline = make_pipeline(
    column_kernelizer,
    mkr_model,
)
pipeline

###############################################################################
# Fit the model
# -------------
#
# We fit on the train set, and score on the test set.
#
# With a GPU backend, the fitting of this model takes around 6 minutes. With a
# CPU backend, it can last 10 times more.

pipeline.fit(X_train, Y_train)

scores = pipeline.score(X_test, Y_test)
scores = backend.to_numpy(scores)

print("(n_voxels,) =", scores.shape)

###############################################################################
# Compare with a ridge model
# --------------------------
#
# We can compare with a baseline model, which does not use one hyperparameter
# per feature space, but instead shares the same hyperparameter for all feature
# spaces.

from himalaya.kernel_ridge import KernelRidgeCV

pipeline_baseline = make_pipeline(
    StandardScaler(with_mean=True, with_std=False),
    Delayer(delays=[1, 2, 3, 4]),
    KernelRidgeCV(
        alphas=alphas, cv=cv,
        solver_params=dict(n_targets_batch=n_targets_batch,
                           n_alphas_batch=n_alphas_batch,
                           n_targets_batch_refit=n_targets_batch_refit)),
)
pipeline_baseline

###############################################################################
pipeline_baseline.fit(X_train, Y_train)
scores_baseline = pipeline_baseline.score(X_test, Y_test)
scores_baseline = backend.to_numpy(scores_baseline)

###############################################################################
# Here we plot the comparison of model performances with a 2D histogram.
# All 70k voxels are represented in this histogram, where the diagonal
# corresponds to identical performance for both models. A distibution deviating
# from the diagonal means that one model has better predictive performances
# than the other.
import matplotlib.pyplot as plt
from voxelwise_tutorials.viz import plot_hist2d

ax = plot_hist2d(scores_baseline, scores)
ax.set(title='Generalization R2 scores', xlabel='KernelRidgeCV',
       ylabel='MultipleKernelRidgeCV')
plt.show()

###############################################################################
# We see that the banded ridge model (``MultipleKernelRidgeCV``) outperforms
# the ridge model (``KernelRidegeCV``). Indeed, banded ridge regression is able
# to find the optimal scalings of each feature space, independently on each
# voxel. Banded ridge regression is thus able to perform a soft selection
# between the available feature spaces, based on the cross-validation
# performances.

###############################################################################
# Plot the banded ridge split
# ---------------------------
#
# On top of better performances, banded ridge regression also gives a way to
# disentangle the two feature spaces. To do so, we take the kernel weights and
# the ridge (dual) weights corresponding to each feature space, and use them to
# split the prediction on each feature space.
#
# .. math::
#
#       \hat{y} = \sum_i^m \hat{y}_i = \sum_i^m \gamma_i K_i \hat{w}
#
# Then, we use these split predictions to compute split math:`\tilde{R}^2_i`
# scores, corrected so that there sum is equal to the math:`R^2` score of the
# full prediction math:`\hat{y}`.

from himalaya.scoring import r2_score_split

Y_test_pred_split = pipeline.predict(X_test, split=True)
split_scores = r2_score_split(Y_test, Y_test_pred_split)
split_scores.shape

print("(n_kernels, n_samples_test, n_voxels) =", Y_test_pred_split.shape)
print("(n_kernels, n_voxels) =", split_scores.shape)

###############################################################################
# We can then plot the split scores on a flatmap with a 2D colormap.

from voxelwise_tutorials.viz import plot_2d_flatmap_from_mapper

mapper_file = os.path.join(directory, "mappers", f"{subject}_mappers.hdf")
ax = plot_2d_flatmap_from_mapper(split_scores[0], split_scores[1],
                                 mapper_file, vmin=0, vmax=0.25, vmin2=0,
                                 vmax2=0.5, label_1=feature_names[0],
                                 label_2=feature_names[1])
plt.show()

###############################################################################
# The blue regions are predicted using the motion-energy features, the orange
# regions are predicted using the wordnet features, and the white regions are
# well predicted using both feature spaces.
#
# Compared to the last figure of the previous example, we see that most white
# regions have been replaced by either blue or orange regions. Indeed, the
# banded ridge regression disentangled the two feature spaces in voxels where
# both feature spaces had good performances (see previous example). The
# disentanglement is consistent with prior knowledge. For example, the
# motion-energy features are selected in the early visual cortex, while the
# wordnet features are selected in the semantic visual areas. For more
# discussions about these results, we refer the reader to the original
# publication [1]_.

###############################################################################
# References
# ----------
#
# .. [1] Nunez-Elizalde, A. O., Huth, A. G., & Gallant, J. L. (2019).
#     Voxelwise encoding models with non-spherical multivariate normal priors.
#     Neuroimage, 197, 482-492.

del pipeline, pipeline_baseline
