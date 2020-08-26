import numpy as np
import pytest
from voxelwise_tutorials.utils import generate_leave_one_run_out


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
