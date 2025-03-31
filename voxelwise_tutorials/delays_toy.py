import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma


def simulate_bold(stimulus_array, TR=2.0, duration=32.0):
    """Simulate BOLD signal by convolving a stimulus time series with a canonical HRF.

    Parameters:
    -----------
    stimulus_array : array-like
        Binary array representing stimulus onset (1) and absence (0)
    TR : float, optional
        Repetition time of the scanner in seconds (default: 2.0s)
    duration : float, optional
        Total duration to model the HRF in seconds (default: 32.0s)

    Returns:
    --------
    bold_signal : ndarray
        The simulated BOLD signal after convolution
    """
    # Create time array based on TR
    t = np.arange(0, duration, TR)

    # Define canonical double-gamma HRF
    # Parameters for the canonical HRF (based on SPM defaults)
    peak_delay = 6.0  # delay of peak in seconds
    undershoot_delay = 16.0  # delay of undershoot
    peak_disp = 1.0  # dispersion of peak
    undershoot_disp = 1.0  # dispersion of undershoot
    peak_amp = 1.0  # amplitude of peak
    undershoot_amp = 0.1  # amplitude of undershoot

    # Compute positive and negative gamma functions
    pos_gamma = peak_amp * gamma.pdf(t, peak_delay / peak_disp, scale=peak_disp)
    neg_gamma = undershoot_amp * gamma.pdf(
        t, undershoot_delay / undershoot_disp, scale=undershoot_disp
    )

    # Canonical HRF = positive gamma - negative gamma
    hrf = pos_gamma - neg_gamma

    # Normalize HRF to have sum = 1
    hrf = hrf / np.sum(hrf)

    # Perform convolution
    bold_signal = np.convolve(stimulus_array, hrf, mode="full")

    # Return only the part of the signal that matches the length of the input
    return bold_signal[: len(stimulus_array)]


def create_voxel_data(n_trs=50, TR=2.0, onset=30, duration=10, random_seed=42):
    """Create a toy dataset with a single voxel and a single feature.

    Parameters
    ----------
    n_trs : int
        Number of time points (TRs).
    TR : float
        Repetition time in seconds.
    activation_t : float
        Time point of activation onset in seconds.
    activation_duration : float
        Duration of activation in seconds.
    random_seed : int
        Seed for random number generation.

    Returns
    -------
    X : array of shape (n_trs,)
        The generated feature.
    Y : array of shape (n_trs,)
        The generated voxel data.
    times : array of shape (n_trs,)
        The time points corresponding to the voxel data.
    """
    if onset > n_trs * TR:
        raise ValueError("onset must be less than n_trs * TR")
    if duration > n_trs * TR:
        raise ValueError("duration must be less than n_trs * TR")
    if duration < 0:
        raise ValueError("duration must be greater than 0")
    if onset < 0:
        raise ValueError("onset must be greater than 0")
    rng = np.random.RandomState(random_seed)
    # figure out slices of activation
    activation = slice(int(onset / TR), int((onset + duration) / TR))
    X = np.zeros(n_trs)
    # add some arbitrary value to our feature
    X[activation] = 1
    Y = simulate_bold(X, TR=TR)
    # add some noise
    Y += rng.randn(n_trs) * 0.1
    times = np.arange(n_trs) * TR
    return X, Y, times


def plot_delays_toy(X_delayed, Y, times, highlight=None):
    """Creates a figure showing a BOLD response and delayed versions of a stimulus.

    Parameters
    ----------
    X_delayed : ndarray, shape (n_timepoints, n_delays)
        The delayed stimulus, where each column corresponds to a different delay.
    Y : ndarray, shape (n_timepoints,)
        The BOLD response time series.
    times : ndarray, shape (n_timepoints,)
        Time points in seconds.
    highlight : float or None, optional
        Time point to highlight in the plot (default: 30 seconds).

    Returns
    -------
    axs : ndarray
        Array of matplotlib axes objects containing the plots.
    """
    if X_delayed.ndim == 1:
        X_delayed = X_delayed[:, np.newaxis]
    n_delays = X_delayed.shape[1]
    n_rows = n_delays + 1
    TR = times[1] - times[0]

    fig, axs = plt.subplots(
        n_rows,
        1,
        figsize=(6, n_rows),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axs[0].plot(times, Y, color="r")
    axs[0].set_title("BOLD response")
    if len(axs) == 2:
        axs[1].set_title("Feature")
        axs[1].plot(times, X_delayed, color="k")
    else:
        for i, (ax, xx) in enumerate(zip(axs.flat[1:], X_delayed.T)):
            ax.plot(times, xx, color="k")
            ax.set_title(
                "$x(t - {0:.0f})$ (feature delayed by {1} sample{2})".format(
                    i * TR, i, "" if i == 1 else "s"
                )
            )
    if highlight is not None:
        for ax in axs.flat:
            ax.axvline(highlight, color="gray")
            ax.set_yticks([])
    _ = axs[-1].set_xlabel("Time [s]")
    # add more margin at the top of the y axis
    ylim = axs[0].get_ylim()
    axs[0].set_ylim(ylim[0], ylim[1] * 1.2)
    return axs
