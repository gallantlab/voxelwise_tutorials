import numpy as np
import pytest

from voxelwise_tutorials.utils import generate_leave_one_run_out, zscore_runs


def test_generate_leave_one_run_out_disjoint():
    n_samples = 40
    run_onsets = [0, 10, 20, 30]

    for train, val in generate_leave_one_run_out(n_samples, run_onsets):
        assert len(train) > 0
        assert len(val) > 0
        assert not np.any(np.isin(train, val))
        assert not np.any(np.isin(val, train))


@pytest.mark.parametrize("run_onsets",
                         [[0, 10, 20, 30, 40], [0, 10, 20, 20, 30]])
def test_generate_leave_one_run_out_empty_runs(run_onsets):
    n_samples = 40
    with pytest.raises(ValueError):
        list(generate_leave_one_run_out(n_samples, run_onsets))


def test_zscore_runs():
    # Create sample data
    data = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [10, 11, 12],
        [11, 12, 13],
        [12, 13, 14],
        [13, 14, 15]
    ], dtype=np.float64)
    
    run_onsets = [0, 4]  # Two runs: first 4 samples and last 4 samples
    
    # Apply zscore_runs
    zscored_data = zscore_runs(data, run_onsets)
    
    # Check shape preserved
    assert zscored_data.shape == data.shape
    
    # Check that each run has mean 0 and std 1
    run1 = zscored_data[:4]
    run2 = zscored_data[4:]
    
    # For each run and each feature column, mean should be close to 0 and std close to 1
    for run in [run1, run2]:
        assert np.allclose(run.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(run.std(axis=0), 1, atol=1e-10)
    
    # Test with integer data to ensure dtype handling works correctly
    int_data = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ], dtype=np.int32)
    
    int_result = zscore_runs(int_data, [0])
    assert int_result.dtype == np.int32