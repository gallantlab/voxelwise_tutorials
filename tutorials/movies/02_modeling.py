###############################################################################
# Here we model the fMRI responses using the motion energy features.

path = '/data1/tutorials/vim2/'

###############################################################################
# We first take a peak at the data shape.

import h5py
import os.path as op

# Here the data is not loaded in memory, we only take a peak at the data shape.
with h5py.File(op.join(path, 'VoxelResponses_subject1.mat'), 'r') as f:
    print(f.keys())  # Show all variables
    for key in f.keys():
        print(f[key])

###############################################################################
# Then we load the fMRI responses.

import numpy as np

with h5py.File(op.join(path, 'VoxelResponses_subject1.mat'), 'r') as f:
    Y_train = np.array(f['rt'])
    Y_test_repeats = np.array(f['rva'])

    # transpose to fit in scikit-learn's API
    Y_train = Y_train.T
    Y_test_repeats = np.transpose(Y_test_repeats, [1, 2, 0])

    # Change to True to select only voxels from left hemisphere V1 ("v1lh")
    if False:
        roi = np.array(f['/roi/v1lh']).ravel()
        mask = (roi == 1)
        Y_train = Y_train[:, mask]
        Y_test_repeats = Y_test_repeats[:, :, mask]

    # zscore test runs
    Y_test_repeats /= np.std(Y_test_repeats, axis=1, keepdims=True)
    # average repeats
    Y_test = Y_test_repeats.mean(0)

# remove nans
Y_train = np.nan_to_num(Y_train)
Y_test = np.nan_to_num(Y_test)

###############################################################################
# Then we load the motion energy features.

X_train = np.load(op.join(path, "features", "motion_energy_train.npy"))
X_test = np.load(op.join(path, "features", "motion_energy_test.npy"))

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

###############################################################################
# Then we define a leave-one-run-out cross-validation split.

from sklearn.model_selection import check_cv
from voxelwise.utils import generate_leave_one_run_out

run_onsets = np.arange(0, X_train.shape[0], 600)
cv = generate_leave_one_run_out(X_train.shape[0], run_onsets)
cv = check_cv(cv)  # copy the splitter into a reusable list

###############################################################################
# Then we define the model pipeline.

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise.delayer import Delayer
from himalaya.kernel_ridge import KernelRidgeCV

from himalaya.backend import set_backend
backend = set_backend("torch_cuda")

alphas = np.logspace(0, 20, 40)
pipeline = make_pipeline(
    StandardScaler(with_mean=True, with_std=False),
    Delayer(delays=[1, 2, 3, 4, 5, 6, 7, 8]),
    KernelRidgeCV(
        alphas=alphas, cv=cv,
        solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                           n_targets_batch_refit=100)),
)

###############################################################################
# We fit on the train set, score on the test set.

print("model fitting..")
pipeline.fit(X_train, Y_train)
scores = pipeline.score(X_test, Y_test)
scores = backend.to_numpy(scores)

###############################################################################
# Compare with a (wrong) model that has no feature delays (i.e. no Delayer)

pipeline_nodelay = make_pipeline(
    StandardScaler(with_mean=True, with_std=False),
    KernelRidgeCV(
        alphas=alphas, cv=cv,
        solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                           n_targets_batch_refit=100)),
)

pipeline_nodelay.fit(X_train, Y_train)
scores_nodelay = pipeline_nodelay.score(X_test, Y_test)
scores_nodelay = backend.to_numpy(scores_nodelay)

###############################################################################
# Plot the optimal alphas selected by the solver.
#
# This plot is helpful to refine the alpha grid if the range is too small or
# too large.

import matplotlib.pyplot as plt
from himalaya.viz import plot_alphas_diagnostic

ax = plot_alphas_diagnostic(backend.to_numpy(pipeline[-1].best_alphas_),
                            alphas=alphas)
ax = plot_alphas_diagnostic(
    backend.to_numpy(pipeline_nodelay[-1].best_alphas_), alphas=alphas, ax=ax)
plt.legend(['with delays', 'without delays'])
plt.show()

###############################################################################
# Plot the comparison of performances with a 2D histogram.

from voxelwise.viz import plot_hist2d

ax = plot_hist2d(scores_nodelay, scores)
ax.set(title='Generalization R2 scores', xlabel='model without delays',
       ylabel='model with delays')
plt.show()
