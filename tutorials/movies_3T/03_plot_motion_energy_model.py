"""
=============================================
Fit a ridge model with motion energy features
=============================================

In this second example, we model the fMRI responses with motion-energy features
extracted from the movie stimulus. The model is still a regularized linear
regression model.

Motion-energy features result from filtering a video stimulus with
spatio-temporal Gabor filters. A pyramid of filters is used to compute the
motion-energy features at multiple spatial and temporal scales.

As in the previous example, we first concatenate the features with multiple
delays, to account for the hemodynamic response. The linear regression model
will then weight each delayedbfeature with a different weight, to build a
predictive model.

Again, the linear regression is regularized to improve robustness to correlated
features and to improve generalization. The optimal regularization
hyperparameter is selected over a grid-search with cross-validation.

Finally, the model generalization performance is evaluated on a held-out test
set, comparing the model predictions with the corresponding ground-truth fMRI
responses.

The ridge model uses the package "himalaya", available
at https://github.com/gallantlab/himalaya.
This package can fit the model either on CPU or on GPU.
"""
###############################################################################

# path of the data directory
directory = '/data1/tutorials/vim-4/'

# modify to use another subject
subject = "S01"

###############################################################################
# Load the data
# -------------
#
# We first load the fMRI responses.
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
# Then we load the "motion_energy" features, that are going to be used for the
# linear regression model.

feature_space = "motion_energy"
file_name = op.join(directory, "features", f"{feature_space}.hdf")
X_train = load_hdf5_array(file_name, key="X_train")
X_test = load_hdf5_array(file_name, key="X_test")

# We use single precision float to speed up model fitting on GPU.
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

###############################################################################
# Define the cross-validation scheme
# ----------------------------------
#
# To select the best hyperparameter through cross-validation, we must define a
# train-validation splitting scheme. Since fMRI time-series are autocorrelated
# in time, we should preserve as much as possible the time blocks.
# In other words, since consecutive time samples are correlated, we should not
# put one time sample in the training set and the immediately following time
# sample in the validation set. Thus, we define here a leave-one-run-out
# cross-validation split, which preserves each recording run.

from sklearn.model_selection import check_cv
from voxelwise.utils import generate_leave_one_run_out

n_samples_train = X_train.shape[0]

# indice of first sample of each run, each run having 600 samples
run_onsets = np.arange(0, n_samples_train, 600)

# define a cross-validation splitter, compatible with scikit-learn
cv = generate_leave_one_run_out(n_samples_train, run_onsets)
cv = check_cv(cv)  # copy the splitter into a reusable list

###############################################################################
# Define the model
# ----------------
#
# Now, let's define the model pipeline.

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# display the scikit-learn pipeline with an HTML diagram
from sklearn import set_config
set_config(display='diagram')

###############################################################################
# With one target, we could directly use the pipeline in scikit-learn's
# GridSearchCV, to select the optimal hyperparameters over cross-validation.
# However, GridSearchCV can only optimize one score. Thus, in the multiple
# target case, GridSearchCV can only optimize e.g. the mean score over targets.
# Here, we want to find a different optimal hyperparameter per target/voxel, so
# we use himalaya's KernelRidgeCV instead.
from himalaya.kernel_ridge import KernelRidgeCV

###############################################################################
# We first concatenate the features with multiple delays, to account for the
# hemodynamic response. The linear regression model will then weight each
# delayed feature with a different weight, to build a predictive model.
#
# With a sample every 2 seconds, we use 4 delays [1, 2, 3, 4] to cover the
# most part of the hemodynamic response peak.

from voxelwise.delayer import Delayer

###############################################################################
# We set the backend to "torch_cuda" to fit the model using GPU.
# The available backends are:
# - "numpy" (CPU) (default)
# - "torch" (CPU)
# - "torch_cuda" (GPU)
# - "cupy" (GPU)
from himalaya.backend import set_backend
backend = set_backend("torch_cuda")

###############################################################################
# The scale of the regularization hyperparameter alpha is unknown, so we use
# a large logarithmic range, and we will check after the fit that best
# hyperparameters are not all on one range edge.
alphas = np.logspace(1, 20, 20)

###############################################################################
# The scikit-learn Pipeline can be used as a regular estimator, calling
# pipeline.fit, pipeline.predict, etc.
# Using a pipeline can be useful to clarify the different steps, avoid
# cross-validation mistakes, or automatically cache intermediate results.
pipeline_motion_energy = make_pipeline(
    StandardScaler(with_mean=True, with_std=False),
    Delayer(delays=[1, 2, 3, 4]),
    KernelRidgeCV(
        alphas=alphas, cv=cv,
        solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                           n_targets_batch_refit=100)),
)
pipeline_motion_energy

###############################################################################
# Fit the model
# -------------
#
# We fit on the train set, and score on the test set.

pipeline_motion_energy.fit(X_train, Y_train)

scores_motion_energy = pipeline_motion_energy.score(X_test, Y_test)
# Since we performed the KernelRidgeCV on GPU, scores are returned as
# torch.Tensor on GPU. Thus, we need to move them into numpy arrays on CPU, to
# be able to use them e.g. in a matplotlib figure.
scores_motion_energy = backend.to_numpy(scores_motion_energy)

###############################################################################
# Plot the model performances
# ---------------------------
#
# To visualize the model performances, we can plot them on a flatten
# surface of the brain, using a mapper that is specific to the subject brain.
import matplotlib.pyplot as plt
from voxelwise.viz import plot_flatmap_from_mapper

mapper_file = op.join(directory, "mappers", f"{subject}_mappers.hdf")
ax = plot_flatmap_from_mapper(scores_motion_energy, mapper_file, vmin=0,
                              vmax=0.5)
plt.show()

###############################################################################
# Another possible visualization is to map the voxel data to a Freesurfer
# average surface ("fsaverage").

import cortex
from voxelwise.io import load_hdf5_sparse_array

surface = "fsaverage_pycortex"  # ("fsaverage" outside the Gallant lab)

# First, let's download the fsaverage surface if it does not exist
if not hasattr(cortex.db, surface):
    cortex.utils.download_subject(subject_id=surface)

# Then, we use load the fsaverage mappers, and use it with a dot product
voxel_to_fsaverage = load_hdf5_sparse_array(mapper_file, 'voxel_to_fsaverage')
projected = voxel_to_fsaverage @ scores_motion_energy

# Finally, we use the data projected on a surface, using pycortex
vertex = cortex.Vertex(projected, surface, vmin=0, vmax=0.5, cmap='inferno',
                       with_curvature=True)
fig = cortex.quickshow(vertex)
plt.show()

# Alternatively, we can start a webGL viewer in the browser, to visualize the
# surface in 3D. Note that this cannot be executed on sphinx gallery.
if False:
    cortex.webshow(vertex, open_browser=True)

###############################################################################
# Compare with the wordnet model
# ------------------------------
#
# It is interesting to compare the performances of this motion-energy model,
# to the performances of the semantic "wordnet" model fitted in the previous
# example. To compare them, we first need to fit again the semantic model.

feature_space = "wordnet"
file_name = op.join(directory, "features", f"{feature_space}.hdf")
X_train = load_hdf5_array(file_name, key="X_train")
X_test = load_hdf5_array(file_name, key="X_test")

# We use single precision float to speed up model fitting on GPU.
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# we can create an unfitted copy of the pipeline with the `clone` function
from sklearn.base import clone
pipeline_wordnet = clone(pipeline_motion_energy)

pipeline_wordnet.fit(X_train, Y_train)
scores_wordnet = pipeline_wordnet.score(X_test, Y_test)
scores_wordnet = backend.to_numpy(scores_wordnet)

###############################################################################
# Here we plot the comparison of model performances with a 2D histogram.
# All ~70k voxels are represented in this histogram, where the diagonal
# corresponds to identical performance for both models. A distibution deviating
# from the diagonal means that one model has better predictive performances
# than the other.

from voxelwise.viz import plot_hist2d

ax = plot_hist2d(scores_wordnet, scores_motion_energy)
ax.set(title='Generalization R2 scores', xlabel='semantic wordnet model',
       ylabel='motion energy model')
plt.show()

###############################################################################
# Interestingly, the well predicted voxels are different in the two models.
# To further describe these differences, we can plot the performances on a
# flatmap.

from voxelwise.viz import plot_2d_flatmap_from_mapper

mapper_file = op.join(directory, "mappers", f"{subject}_mappers.hdf")
ax = plot_2d_flatmap_from_mapper(scores_wordnet, scores_motion_energy,
                                 mapper_file, vmin=0, vmax=0.5, vmin2=0,
                                 vmax2=0.5, label_1="wordnet",
                                 label_2="motion energy")
plt.show()
