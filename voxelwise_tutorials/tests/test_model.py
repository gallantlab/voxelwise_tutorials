import os

import pytest
import numpy as np

from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV

from voxelwise_tutorials.delayer import Delayer
from voxelwise_tutorials.io import load_hdf5_array
from voxelwise_tutorials.io import get_data_home
from voxelwise_tutorials.io import download_datalad
from voxelwise_tutorials.utils import explainable_variance
from voxelwise_tutorials.utils import generate_leave_one_run_out

# use "cupy" or "torch_cuda" for faster computation with GPU
backend = set_backend("numpy", on_error="warn")

# Download the dataset
subject = "S01"
feature_spaces = ["motion_energy", "wordnet"]
directory = get_data_home(dataset="shortclips")
for file_name in [
        "features/motion_energy.hdf",
        "features/wordnet.hdf",
        "mappers/S01_mappers.hdf",
        "responses/S01_responses.hdf",
]:
    download_datalad(file_name, destination=directory,
                     source="https://gin.g-node.org/gallantlab/shortclips")


def run_model(X_train, X_test, Y_train, Y_test, run_onsets):
    ##############
    # define model
    n_samples_train = Y_train.shape[0]
    cv = generate_leave_one_run_out(n_samples_train, run_onsets,
                                    random_state=0, n_runs_out=1)
    cv = check_cv(cv)

    alphas = np.logspace(-4, 15, 20)

    model = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        Delayer(delays=[1, 2, 3, 4]),
        KernelRidgeCV(
            kernel="linear", alphas=alphas, cv=cv,
            solver_params=dict(n_targets_batch=1000, n_alphas_batch=10)),
    )

    ###########
    # run model
    model.fit(X_train, Y_train)
    test_scores = model.score(X_test, Y_test)

    test_scores = backend.to_numpy(test_scores)
    # cv_scores = backend.to_numpy(model[-1].cv_scores_)

    return test_scores


@pytest.mark.parametrize('feature_space', feature_spaces)
def test_model_fitting(feature_space):
    ###########################################
    # load the data

    # load X
    features_file = os.path.join(directory, 'features',
                                 feature_space + ".hdf")
    features = load_hdf5_array(features_file)
    X_train = features['X_train']
    X_test = features['X_test']

    # load Y
    responses_file = os.path.join(directory, 'responses',
                                  subject + "_responses.hdf")
    responses = load_hdf5_array(responses_file)
    Y_train = responses['Y_train']
    Y_test_repeats = responses['Y_test']
    run_onsets = responses['run_onsets']

    #############################################
    # select voxels based on explainable variance
    ev = explainable_variance(Y_test_repeats)
    mask = ev > 0.4
    assert mask.sum() > 0
    Y_train = Y_train[:, mask]
    Y_test = Y_test_repeats[:, :, mask].mean(0)

    ###########################################
    # fit a ridge model and compute test scores
    test_scores = run_model(X_train, X_test, Y_train, Y_test, run_onsets)
    assert np.percentile(test_scores, 95) > 0.05
    assert np.percentile(test_scores, 99) > 0.15
    assert np.percentile(test_scores, 100) > 0.35
