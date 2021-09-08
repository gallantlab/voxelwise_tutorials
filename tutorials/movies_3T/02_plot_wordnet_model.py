"""
=======================================
Fit a ridge model with wordnet features
=======================================

In this example, we model the fMRI responses with semantic "wordnet" features,
manually annotated on each frame of the movie stimulus. The model is a
regularized linear regression model, known as ridge regression. Since this
model is used to predict brain activity from the stimulus, it is called a
(voxelwise) encoding model.

This example reproduces part of the analysis described in Huth et al (2012)
[1]_. See this publication for more details about the experiment, the wordnet
features, along with more results and more discussions.

*Wordnet features:* The features used in this example are semantic labels
manually annotated on each frame of the movie stimulus. The semantic labels
include nouns (such as "woman", "car", or "building") and verbs (such as
"talking", "touching", or "walking"), for a total of 1705 distinct category
labels. To interpret our model, labels can be organized in a graph of semantic
relashionship based on the `Wordnet <https://wordnet.princeton.edu/>`_ dataset.

*Summary:* We first concatenate the features with multiple temporal delays, to
account for the slow hemodynamic response. We then fit a predictive model of
BOLD activity, using a  linear regression that weighs each delayed feature
differently. The linear regression is regularized to improve robustness to
correlated features and to improve generalization. The optimal regularization
hyperparameter is selected over a grid-search with cross-validation. Finally,
the model generalization performance is evaluated on a held-out test set,
comparing the model predictions with the corresponding ground-truth fMRI
responses.
"""
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
# We first load the fMRI responses. These responses have been preprocessed as
# decribed in [1]_. The data is separated into a training set ``Y_train`` and a
# testing set ``Y_test``. The training set is used for fitting models, and
# selecting the best models and hyperparameters. The testing set is later used
# to estimate the generalization performances of the selected model. The
# testing set contains multiple repetitions of the same experiment, to estimate
# an upper bound of the model performances (cf. previous example).
import numpy as np
from voxelwise_tutorials.io import load_hdf5_array

file_name = os.path.join(directory, "responses", f"{subject}_responses.hdf")
Y_train = load_hdf5_array(file_name, key="Y_train")
Y_test = load_hdf5_array(file_name, key="Y_test")

print("(n_samples_train, n_voxels) =", Y_train.shape)
print("(n_repeats, n_samples_test, n_voxels) =", Y_test.shape)

###############################################################################
# If we repeat an experiment multiple times, part of the fMRI responses might
# change. However the modeling features do not change over the repeats, so the
# voxelwise encoding model predicts the same signal for each repeat. To have an
# upper bound of the model performances, we keep only the repeatable part of
# the signal by averaging the test repeats. It means that the prediction
# :math:`R^2` scores will be relative to the explainable variance (cf. previous
# example).
Y_test = Y_test.mean(0)

print("(n_samples_test, n_voxels) =", Y_test.shape)

###############################################################################
# We fill potential NaN (not-a-number) values with zeros.
Y_train = np.nan_to_num(Y_train)
Y_test = np.nan_to_num(Y_test)

###############################################################################
# Then, we load the semantic "wordnet" features, extracted from the stimulus at
# each time point. The features corresponding to the training set are noted
# ``X_train``, and the features corresponding to the testing set are noted
# ``X_test``.
feature_space = "wordnet"

file_name = os.path.join(directory, "features", f"{feature_space}.hdf")
X_train = load_hdf5_array(file_name, key="X_train")
X_test = load_hdf5_array(file_name, key="X_test")

print("(n_samples_train, n_features) =", X_train.shape)
print("(n_samples_test, n_features) =", X_test.shape)

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
# Now, let's define the model pipeline.
#
# We first center the features, since we will not use an intercept. Indeed, the
# mean value in fMRI recording is non-informative, so each run is detrended and
# demeaned independently, and we do not need to predict an intercept value in
# the linear model.
#
# However, we prefer not to normalize by the standard deviation of each
# feature. Indeed, if the features are extracted in a consistent way from the
# stimulus, their relative scale is meaningful. Normalizing them independently
# from each other would remove this meaning. Moreover, the wordnet features are
# one-hot-encoded, which means that each feature is either present (1) or not
# present (0) in each sample. Normalizing one-hot-encoded features is not
# recommended, since it would scale disproportionately the infrequent features.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=True, with_std=False)

###############################################################################
# Then we concatenate the features with multiple delays to account for the
# hemodynamic response. Indeed, the BOLD signal recorded in fMRI experiments is
# delayed in time with respect to the stimulus. With different delayed versions
# of the features, the linear regression model will weight each delayed feature
# with a different weight, to maximize the predictions. With a sample every 2
# seconds, we typically use 4 delays [1, 2, 3, 4] to cover the most part of the
# hemodynamic response peak. In the next example, we further describe this
# hemodynamic response estimation.
from voxelwise_tutorials.delayer import Delayer
delayer = Delayer(delays=[1, 2, 3, 4])

###############################################################################
# Finally, we use a ridge regression model. Ridge regression is a linear
# regression with a L2 regularization. The L2 regularizatin improves robustness
# to correlated features and improves generalization. However, the L2
# regularization is controled by a hyperparameter ``alpha`` that needs to be
# tuned. This regularization hyperparameter is usually selected over a grid
# search with cross-validation, selecting the hyperparameter that maximizes the
# predictive performances on the validation set. More details about
# cross-validation can be found in the `scikit-learn documentation
# <https://scikit-learn.org/stable/modules/cross_validation.html>`_.
#
# For computational reasons, when the number of features is larger than the
# number of samples, it is more efficient to solve a ridge regression using the
# (equivalent) dual formulation [2]_. This dual formulation is equivalent to
# kernel ridge regression with a linear kernel. Here, we have 3600 training
# samples, and 1705 * 4 = 6820 features (we multiply by 4 since we use 4 time
# delays), therefore it is more efficient to use kernel ridge regression.
#
# With one target, we could directly use the pipeline in ``scikit-learn``'s
# ``GridSearchCV``, to select the optimal regularization hyperparameter
# (``alpha``) over cross-validation. However, ``GridSearchCV`` can only
# optimize one score. Thus, in the multiple-target case, ``GridSearchCV`` can
# only optimize (for example) the mean score over targets. Here, we want to
# find a different optimal hyperparameter per target/voxel, so we use the
# package `himalaya <https://github.com/gallantlab/himalaya>`_ which implements
# a ``scikit-learn`` compatible estimator ``KernelRidgeCV``, with
# hyperparameter selection independently on each target.
from himalaya.kernel_ridge import KernelRidgeCV

###############################################################################
# Interestingly, ``himalaya`` implements different computational backends,
# including two backends that use GPU for faster computations. The two
# available GPU backends are "torch_cuda" and "cupy". (Each backend is only
# available if you installed the corresponding package with CUDA enabled. Check
# the ``pytorch``/``cupy`` documentation for install instructions.)
#
# Here we use the "torch_cuda" backend, but if the import fails we continue
# with the default "numpy" backend. The "numpy" backend is expected to be
# slower since it only uses the CPU.
from himalaya.backend import set_backend
backend = set_backend("torch_cuda", on_error="warn")
print(backend)

###############################################################################
# To speed up model fitting on GPU, we use single precision float numbers.
# (This step probably does not change significantly the performances on non-GPU
# backends.)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

###############################################################################
# Since the scale of the regularization hyperparameter ``alpha`` is unknown, we
# use a large logarithmic range, and we will check after the fit that best
# hyperparameters are not all on one range edge.
alphas = np.logspace(1, 20, 20)

###############################################################################
# We also indicate some batch sizes to limit the GPU memory.
kernel_ridge_cv = KernelRidgeCV(
    alphas=alphas, cv=cv,
    solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                       n_targets_batch_refit=100))

###############################################################################
# Finally, we use a ``scikit-learn`` ``Pipeline`` to link the different steps
# together. A ``Pipeline`` can be used as a regular estimator, calling
# ``pipeline.fit``, ``pipeline.predict``, etc. Using a ``Pipeline`` can be
# useful to clarify the different steps, avoid cross-validation mistakes, or
# automatically cache intermediate results. See the ``scikit-learn``
# `documentation <https://scikit-learn.org/stable/modules/compose.html>`_ for
# more information.
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(
    scaler,
    delayer,
    kernel_ridge_cv,
)

###############################################################################
# We can display the ``scikit-learn`` pipeline with an HTML diagram.
from sklearn import set_config
set_config(display='diagram')  # requires scikit-learn 0.23
pipeline

###############################################################################
# Fit the model
# -------------
#
# We fit on the train set..

_ = pipeline.fit(X_train, Y_train)

###############################################################################
# ..and score on the test set. Here the scores are the :math:`R^2` scores, with
# values in :math:`]-\infty, 1]`. A value of :math:`1` means the predictions
# are perfect.
#
# Note that since ``himalaya`` is specifically implementing multiple targets
# models, the ``score`` method differs from ``scikit-learn`` API and returns
# one score per target/voxel.
scores = pipeline.score(X_test, Y_test)
print("(n_voxels,) =", scores.shape)

###############################################################################
# If we fit the model on GPU, scores are returned on GPU using an array object
# specfic to the backend we used (such as a ``torch.Tensor``). Thus, we need to
# move them into ``numpy`` arrays on CPU, to be able to use them for example in
# a ``matplotlib`` figure.
scores = backend.to_numpy(scores)

###############################################################################
# Plot the model performances
# ---------------------------
#
# To visualize the model performances, we can plot them on a flattened
# surface of the brain, using a mapper that is specific to the subject brain.
# (Check previous example to see how to use the mapper to Freesurfer average
# surface.)
import matplotlib.pyplot as plt
from voxelwise_tutorials.viz import plot_flatmap_from_mapper

mapper_file = os.path.join(directory, "mappers", f"{subject}_mappers.hdf")
ax = plot_flatmap_from_mapper(scores, mapper_file, vmin=0, vmax=0.4)
plt.show()

###############################################################################
# We can see that the "wordnet" features successfully predict a part of the
# brain activity, with :math:`R^2` scores as high as 0.4. Note that these
# scores are generalization scores, since they are computed on a test set not
# seen during the mode fitting. Since we fitted a model independently on each
# voxel, we can show the generalization performances at the maximal resolution,
# the voxel.
#
# The best performances are located in visual semantic areas like EBA, or FFA.
# This is expected since the wordnet features encode semantic information about
# the visual stimulus. For more discussions about these results, we refer the
# reader to the original publication [1]_.

###############################################################################
# Plot the selected hyperparameters
# ---------------------------------
#
# Since the scale of alphas is unknown, we plot the optimal alphas selected by
# the solver over cross-validation. This plot is helpful to refine the alpha
# grid if the range is too small or too large.
#
# Note that some voxels might be at the maximum regularization value in the
# grid search. These are voxels where the model has no predictive power, thus
# the optimal regularization parameter is large to lead to a prediction equal
# to zero. We do not need to extend the alpha range for these voxels.

from himalaya.viz import plot_alphas_diagnostic
best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)
plot_alphas_diagnostic(best_alphas=best_alphas, alphas=alphas)
plt.show()

###############################################################################
# Visualize the regression coefficients
# -------------------------------------
#
# Here, we go back to the main model on all voxels. Since our model is linear,
# we can use the (primal) regression coefficients to interpret the model. The
# basic intuition is that the model will use larger coefficients on features
# that have more predictive power.
#
# Since we know the meaning of each feature, we can interpret the large
# regression coefficients. In the case of wordnet features, we can even build
# a graph that represents the features linked by a semantic relationship.

###############################################################################
# We first get the (primal) ridge regression coefficients from the fitted
# model.
primal_coef = pipeline[-1].get_primal_coef()
primal_coef = backend.to_numpy(primal_coef)
print("(n_delays * n_features, n_voxels) =", primal_coef.shape)

###############################################################################
# Here, we are only interested in the voxels with good generalization
# performances. We select an arbitrary threshold of 0.05 (R^2 score).
primal_coef_selection = primal_coef[:, scores > 0.05]

###############################################################################
# Then, we aggregate the coefficients across the different delays.

# split the ridge coefficients per delays
delayer = pipeline.named_steps['delayer']
primal_coef_per_delay = delayer.reshape_by_delays(primal_coef_selection,
                                                  axis=0)
print("(n_delays, n_features, n_voxels) =", primal_coef_per_delay.shape)

# average over delays
average_coef = np.mean(primal_coef_per_delay, axis=0)
print("(n_features, n_voxels) =", average_coef.shape)

###############################################################################
# Even after averaging over delays, the coefficient matrix is still too large
# to understand it. Therefore, we use principal component analysis (PCA) to
# reduce the dimensionality of the matrix.
from sklearn.decomposition import PCA

pca = PCA(n_components=4)
pca.fit(average_coef.T)
components = pca.components_
print("(n_components, n_features) =", components.shape)

###############################################################################
# We can check the ratio of explained variance by each principal component.
# We see that the first four components already explain a large part of the
# coefficients variance.
print("PCA explained variance =", pca.explained_variance_ratio_)

###############################################################################
# Similarly to [1]_, we correct the coefficients of features linked by a
# semantic relationship. Indeed, in the wordnet features, if a clip was labeled
# with `wolf`, the authors automatically added the categories `canine`,
# `carnivore`, `placental mammal`, `mamma`, `vertebrate`, `chordate`,
# `organism`, and `whole`. The authors thus argue that the same correction
# needs to be done on the coefficients.

from voxelwise_tutorials.wordnet import load_wordnet
from voxelwise_tutorials.wordnet import correct_coefficients
_, wordnet_categories = load_wordnet(directory=directory)
components = correct_coefficients(components.T, wordnet_categories).T
components -= components.mean(axis=1)[:, None]
components /= components.std(axis=1)[:, None]

###############################################################################
# Finally, we plot the first principal component on the wordnet graph. In such
# graph, links indicate "is a" relationships (e.g. an `athlete` "is a"
# `person`). Each marker represents a single noun (circle) or verb (square).
# The area of each marker indicates the principal component magnitude, and the
# color indicates the sign (red is positive, blue is negative).

from voxelwise_tutorials.wordnet import plot_wordnet_graph
from voxelwise_tutorials.wordnet import apply_cmap

first_component = components[0]
node_sizes = np.abs(first_component)
node_colors = apply_cmap(first_component, vmin=-2, vmax=2, cmap='coolwarm',
                         n_colors=2)

plot_wordnet_graph(node_colors=node_colors, node_sizes=node_sizes)
plt.show()

###############################################################################
# According to the authors of [1]_, "this principal component distinguishes
# between categories with high stimulus energy (e.g. moving objects like
# `person` and `vehicle`) and those with low stimulus energy (e.g. stationary
# objects like `sky` and `city`)".
#
# Our result is slightly different than in [1]_, since we only use one subject,
# and the voxel selection is slightly different. We also use a different
# regularization parameter in each voxels, while in [1]_ all voxels use the
# same regularization parameter. Here, we do not aim at reproducing exactly the
# results in [1]_, but we rather describe the general approach.

###############################################################################
# To project the principal component on the cortical surface, we first need to
# transform the primal weights of all voxels using the fitted PCA.

# split the ridge coefficients per delays
primal_coef_per_delay = delayer.reshape_by_delays(primal_coef, axis=0)
print("(n_delays, n_features, n_voxels) =", primal_coef_per_delay.shape)
del primal_coef

# average over delays
average_coef = np.mean(primal_coef_per_delay, axis=0)
print("(n_features, n_voxels) =", average_coef.shape)
del primal_coef_per_delay

# transform with the fitted PCA
average_coef_transformed = pca.transform(average_coef.T).T
print("(n_components, n_voxels) =", average_coef_transformed.shape)
del average_coef

# We make sure vmin = -vmax, so that the colormap is centered on 0.
vmax = np.percentile(np.abs(average_coef_transformed), 99.9)

# plot the primal weights projected on the first principal component.
ax = plot_flatmap_from_mapper(average_coef_transformed[0], mapper_file,
                              vmin=-vmax, vmax=vmax, cmap='coolwarm')
plt.show()

###############################################################################
# This flatmap shows in which brain regions the model has the largest
# projection on the first component. Again, this result is different from the
# one in [1]_, and should only be considered as reproducing the general
# approach.

###############################################################################
# Following [1]_, we also plot the next three principal components on the
# wordnet graph, mapping the three vectors to RGB colors.

from voxelwise_tutorials.wordnet import scale_to_rgb_cube

next_three_components = components[1:4].T
node_sizes = np.linalg.norm(next_three_components, axis=1)
node_colors = scale_to_rgb_cube(next_three_components)
print("(n_nodes, n_channels) =", node_colors.shape)

plot_wordnet_graph(node_colors=node_colors, node_sizes=node_sizes)
plt.show()

###############################################################################
# According to the authors of [1]_, "this graph shows that categories thought
# to be semantically related (e.g. athletes and walking) are represented
# similarly in the brain".

###############################################################################
# Finally, we project these principal components on the cortical surface.

from voxelwise_tutorials.viz import plot_3d_flatmap_from_mapper

voxel_colors = scale_to_rgb_cube(average_coef_transformed[1:4].T, clip=3).T
print("(n_channels, n_voxels) =", voxel_colors.shape)

ax = plot_3d_flatmap_from_mapper(voxel_colors[0], voxel_colors[1],
                                 voxel_colors[2], mapper_file=mapper_file,
                                 vmin=0, vmax=1, vmin2=0, vmax2=1, vmin3=0,
                                 vmax3=1)
plt.show()

###############################################################################
# Again, our results are different from the ones in [1]_, for the same reasons
# mentioned earlier.

###############################################################################
# References
# ----------
#
# .. [1] Huth, A. G., Nishimoto, S., Vu, A. T., & Gallant, J. L. (2012).
#    A continuous semantic space describes the representation of thousands of
#    object and action categories across the human brain. Neuron, 76(6),
#    1210-1224.
#
# .. [2] Saunders, C., Gammerman, A., & Vovk, V. (1998).
#    Ridge regression learning algorithm in dual variables.

del pipeline
