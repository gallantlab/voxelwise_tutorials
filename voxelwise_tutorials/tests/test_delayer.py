import sklearn.kernel_ridge
import sklearn.utils.estimator_checks

from voxelwise_tutorials.delayer import Delayer


@sklearn.utils.estimator_checks.parametrize_with_checks([Delayer()])
def test_check_estimator(estimator, check):
    check(estimator)
