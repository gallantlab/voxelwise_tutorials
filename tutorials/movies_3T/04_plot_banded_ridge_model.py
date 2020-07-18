"""
=====================================================================
Fit a banded ridge model with both wordnet and motion energy features
=====================================================================

In this example, we model the fMRI responses with a banded ridg regression
model, with two different feature spaces.
The two feature spaces used will be motion-energy and wordnet categories.

Since the relative scaling of both feature spaces is unknown, we use two
regularization hyperparameters (one per feature space). Just like with ridge
regression, we need to optimize the hyperparameters over cross-validation.

The banded ridge model is fitted with the package
`himalaya <https://github.com/gallantlab/himalaya>`_.
"""

# path of the data directory
directory = '/data1/tutorials/vim-4/'

# modify to use another subject
subject = "S01"

###############################################################################
# Load the data
# -------------
#
# As in the previous models, we first load the fMRI responses, which are our
# regression targets.
import os.path as op
import numpy as np

from voxelwise.io import load_hdf5_array

file_name = op.join(directory, "responses", f"{subject}_responses.hdf")
Y_train = load_hdf5_array(file_name, key="Y_train")
Y_test = load_hdf5_array(file_name, key="Y_test")
run_onsets = load_hdf5_array(file_name, key="run_onsets")

# Average test repeats, since we cannot model the non-repeatable part of
# fMRI responses.
Y_test = Y_test.mean(0)

# remove nans, mainly present on non-cortical voxels
Y_train = np.nan_to_num(Y_train)
Y_test = np.nan_to_num(Y_test)

###############################################################################
# Then we load both feature spaces, that are going to be used for the
# linear regression model.

print("loading features..")

feature_names = ["motion_energy", "wordnet"]

Xs_train = []
Xs_test = []
n_features_list = []
for feature_space in feature_names:
    file_name = op.join(directory, "features", f"{feature_space}.hdf")
    Xi_train = load_hdf5_array(file_name, key="X_train")
    Xi_test = load_hdf5_array(file_name, key="X_test")

    Xs_train.append(Xi_train.astype(dtype="float32"))
    Xs_test.append(Xi_test.astype(dtype="float32"))
    n_features_list.append(Xi_train.shape[1])

# concatenate the feature spaces
X_train = np.concatenate(Xs_train, 1)
X_test = np.concatenate(Xs_test, 1)

###############################################################################
# Define the cross-validation scheme
# ----------------------------------
#
# We define again a leave-one-run-out cross-validation split scheme.

from sklearn.model_selection import check_cv
from voxelwise.utils import generate_leave_one_run_out

n_samples_train = X_train.shape[0]

# define a cross-validation splitter, compatible with scikit-learn
cv = generate_leave_one_run_out(n_samples_train, run_onsets)
cv = check_cv(cv)  # copy the splitter into a reusable list

###############################################################################
# Define the model
# ----------------

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# display the scikit-learn pipeline with an HTML diagram
from sklearn import set_config
set_config(display='diagram')

###############################################################################
# We first concatenate the features with multiple delays, to account for the
# hemodynamic response. The linear regression model will then weight each
# delayed feature with a different weight, to build a predictive model.
#
# With a sample every 2 seconds, we use 4 delays [1, 2, 3, 4] to cover the
# most part of the hemodynamic response peak.

from voxelwise.delayer import Delayer

###############################################################################
# We set himalaya's backend to "torch_cuda" to fit the model using GPU.
# The available backends are:
#
# - "numpy" (CPU) (default)
# - "torch" (CPU)
# - "torch_cuda" (GPU)
# - "cupy" (GPU)
from himalaya.backend import set_backend
backend = set_backend("torch_cuda")

###############################################################################
# To fit the banded ridge model, we use ``himalaya``'s scikit-learn API, and
# fit a MultipleKernelRidgeCV model.
# The class takes a number of common parameters during initialization, such as
# `kernels` or `solver`. Since the solver parameters might be different
# depending on the solver, they can be passed in the `solver_params` parameter.

from himalaya.kernel_ridge import MultipleKernelRidgeCV

# Here we will use the "random_search" solver.
# We can check its specific parameters in the function docstring:
solver_function = MultipleKernelRidgeCV.ALL_SOLVERS["random_search"]
print("Docstring of the function %s:" % solver_function.__name__)
print(solver_function.__doc__)

###############################################################################
# We use 50 iterations to have a reasonably fast example.
# To have a better convergence, we might need more iterations.
# Note that there is currently no stopping criterion in this method.
n_iter = 50

###############################################################################
# The scale of the regularization hyperparameter alpha is unknown, so we use
# a large range, and we will check after the fit that best hyperparameters are
# not all on one range edge.
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

mkr_model = MultipleKernelRidgeCV(kernels="precomputed",
                                  solver="random_search",
                                  solver_params=solver_params, cv=cv)

###############################################################################
# We need a bit more work than in previous examples before defining the full
# pipeline, since the banded ridge model requires multiple precomputed kernels,
# one for each feature space. To compute them, we use the ColumnKernelizer,
# that can be conveniently integrated in a scikit-learn Pipeline.

from himalaya.kernel_ridge import Kernelizer
from himalaya.kernel_ridge import ColumnKernelizer

# Find the start and end of each feature space in X_train
start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
slices = [
    slice(start, end)
    for start, end in zip(start_and_end[:-1], start_and_end[1:])
]

###############################################################################
# Then we create a different Kernelizer for each feature space.
# Here we use a linear kernel for all feature spaces, but ColumnKernelizer
# accepts any Kernelizer, or scikit-learn Pipeline ending with a Kernelizer.
preprocess_pipeline = make_pipeline(
    StandardScaler(with_mean=True, with_std=False),
    Delayer(delays=[1, 2, 3, 4]),
    Kernelizer(kernel="linear"),
)

# The column kernelizer applies a different pipeline on each selection of
# features, here defined with ``slices``.
column_kernelizer = ColumnKernelizer([
    (name, preprocess_pipeline, slice_)
    for name, slice_ in zip(feature_names, slices)
])

# Note that ColumnKernelizer has a parameter `n_jobs` to parallelize each
# kernelizer, yet such parallelism does not work with GPU arrays.

###############################################################################
# Then we can define the model pipeline.

pipeline = make_pipeline(
    column_kernelizer,
    mkr_model,
)

###############################################################################
# Fit the model
# -------------
#
# We fit on the train set, and score on the test set.

pipeline.fit(X_train, Y_train)

scores = pipeline.score(X_test, Y_test)

###############################################################################
# Since we performed the MultipleKernelRidgeCV on GPU, scores are returned as
# torch.Tensor on GPU. Thus, we need to move them into numpy arrays on CPU, to
# be able to use them e.g. in a matplotlib figure.
scores = backend.to_numpy(scores)

###############################################################################
# Compare with a ridge model
# --------------------------
#
# We can compare with a baseline model, which does not use one hyperparameter
# per feature space, but share the same hyperparameter for all spaces.

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

print("model fitting..")
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
from voxelwise.viz import plot_hist2d

ax = plot_hist2d(scores_baseline, scores)
ax.set(title='Generalization R2 scores', xlabel='KernelRidgeCV',
       ylabel='MultipleKernelRidgeCV')
plt.show()
