import pytest
import numpy as np

import sklearn.kernel_ridge
import sklearn.utils.estimator_checks

from voxelwise_tutorials.delayer import Delayer


@sklearn.utils.estimator_checks.parametrize_with_checks([Delayer()])
def test_check_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize('delays', [None, [0]])
def test_no_delays(delays):
    X = np.random.randn(10, 3)
    Xt = Delayer(delays=delays).fit_transform(X)
    np.testing.assert_array_equal(Xt, X)


@pytest.mark.parametrize('delays', [[0], [0, 1], [0, -1, 2]])
def test_zero_delay_identity(delays):
    X = np.random.randn(10, 3)
    Xt = Delayer(delays=delays).fit_transform(X)
    np.testing.assert_array_equal(Xt[:, :X.shape[1]], X)


@pytest.mark.parametrize('delays', [[1], [1, 2], [-1, 0, 2]])
def test_nonzero_delay(delays):
    X = np.random.randn(10, 3)
    Xt = Delayer(delays=delays).fit_transform(X)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(Xt[:, :X.shape[1]], X)


@pytest.mark.parametrize('delays', [[1], [1, 2], [-1, 0, 2]])
def test_reshape_by_delays(delays):
    X = np.random.randn(10, 3)
    trans = Delayer(delays=delays)
    Xt = trans.fit_transform(X)
    Xtt = trans.reshape_by_delays(Xt)

    assert Xtt.shape == (len(delays), X.shape[0], X.shape[1])
