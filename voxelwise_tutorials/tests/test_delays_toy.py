import matplotlib.pyplot as plt
import numpy as np
import pytest

from voxelwise_tutorials import delays_toy


def test_simulate_bold():
    """Test that simulate_bold function works correctly"""
    # Create a simple stimulus array with an onset
    stimulus = np.zeros(50)
    stimulus[10:15] = 1  # activation for 5 time points

    # Test with default parameters
    bold = delays_toy.simulate_bold(stimulus)

    # Check that output has the same length as input
    assert len(bold) == len(stimulus)

    # Check that there is activation (values > 0) after stimulus onset
    assert np.any(bold[10:] > 0)

    # Test with different TR
    bold_tr1 = delays_toy.simulate_bold(stimulus, TR=1.0)
    assert len(bold_tr1) == len(stimulus)


def test_create_voxel_data():
    """Test that create_voxel_data function works correctly"""
    # Test with default parameters
    X, Y, times = delays_toy.create_voxel_data()

    # Check shapes
    assert X.shape == (50,)
    assert Y.shape == (50,)
    assert times.shape == (50,)

    # Check that activation happens at expected time
    onset_idx = int(30 / 2.0)  # 30 seconds at TR=2.0
    assert np.all(
        X[onset_idx : onset_idx + 5] == 1
    )  # Should be active for 5 TRs (10s duration)

    # Test with different parameters
    X2, Y2, times2 = delays_toy.create_voxel_data(
        n_trs=100, TR=1.0, onset=20, duration=5
    )
    assert X2.shape == (100,)
    assert Y2.shape == (100,)
    assert times2.shape == (100,)

    # Test error cases
    with pytest.raises(ValueError):
        delays_toy.create_voxel_data(n_trs=50, onset=120)  # onset > n_trs * TR

    with pytest.raises(ValueError):
        delays_toy.create_voxel_data(duration=-5)  # negative duration


def test_plot_delays_toy():
    """Test that plot_delays_toy function works correctly"""
    # Create sample data
    X, Y, times = delays_toy.create_voxel_data()

    # Test with 1D array (single delay)
    axs = delays_toy.plot_delays_toy(X, Y, times)
    assert len(axs) == 2  # Should have 2 subplots
    plt.close()

    # Test with 2D array (multiple delays)
    X_delayed = np.column_stack([X, np.roll(X, 1), np.roll(X, 2)])
    axs = delays_toy.plot_delays_toy(X_delayed, Y, times)
    assert len(axs) == 4  # Should have 4 subplots (1 for Y, 3 for X)
    plt.close()

    # Test with highlight
    axs = delays_toy.plot_delays_toy(X, Y, times, highlight=30)
    plt.close()
