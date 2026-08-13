# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.16",
#     "matplotlib==3.11.1",
#     "numpy==2.5.2",
#     "voxelwise-tutorials==0.2.3",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from voxelwise_tutorials.io import get_data_home, load_hdf5_array
    from voxelwise_tutorials.utils import explainable_variance

    return (
        explainable_variance,
        get_data_home,
        load_hdf5_array,
        mo,
        np,
        os,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Explainable variance for all shortclips subjects

    This focused notebook loads every available subject, computes voxelwise
    explainable variance from the 10 repeated test presentations, and compares
    subjects using a histogram, an empirical survivor plot, and summary statistics.
    """)
    return


@app.cell
def _(explainable_variance, get_data_home, load_hdf5_array, mo, np, os):
    directory = get_data_home(dataset="shortclips")
    responses_directory = os.path.join(directory, "responses")
    subjects = sorted(
        file_name.removesuffix("_responses.hdf")
        for file_name in os.listdir(responses_directory)
        if file_name.endswith("_responses.hdf")
    )
    mo.stop(
        not subjects,
        mo.md(f"No subject response files found in `{responses_directory}`."),
    )

    # Load one subject at a time: the five response files total roughly 9 GB,
    # while the resulting EV vectors are small enough to keep in memory.
    ev_by_subject = {}
    excluded_by_subject = {}
    shapes_by_subject = {}
    for _subject in subjects:
        file_name = os.path.join(
            responses_directory, f"{_subject}_responses.hdf"
        )
        y_test = load_hdf5_array(file_name, key="Y_test")
        shapes_by_subject[_subject] = y_test.shape
        _ev = explainable_variance(y_test)
        _finite = np.isfinite(_ev)
        ev_by_subject[_subject] = _ev[_finite]
        excluded_by_subject[_subject] = int((~_finite).sum())
        del y_test
    return ev_by_subject, excluded_by_subject, shapes_by_subject, subjects


@app.cell
def _(ev_by_subject, np, plt, subjects):
    bins = np.linspace(0, 1, 101)
    figure, axis = plt.subplots(figsize=(10, 6))

    for _subject in subjects:
        axis.hist(
            ev_by_subject[_subject],
            bins=bins,
            log=True,
            histtype="step",
            linewidth=1.7,
            label=f"{_subject} (n={ev_by_subject[_subject].size:,} voxels)",
        )

    axis.set(
        xlabel="Explainable variance",
        ylabel="Number of voxels (log scale)",
        title="Histogram of explainable variance — all subjects",
        xlim=(0, 1),
    )
    axis.grid(True, alpha=0.3)
    axis.legend(title="Subject")
    figure.tight_layout()
    figure
    return


@app.cell
def _(ev_by_subject, np, plt, subjects):
    survivor_figure, survivor_axis = plt.subplots(figsize=(10, 6))

    for _subject in subjects:
        _sorted_ev = np.sort(ev_by_subject[_subject])
        _survival_probability = (
            _sorted_ev.size - np.arange(_sorted_ev.size)
        ) / _sorted_ev.size
        survivor_axis.plot(
            _sorted_ev,
            _survival_probability,
            linewidth=1.7,
            label=_subject,
        )

    survivor_axis.set(
        xlabel="Explainable variance threshold",
        ylabel="Fraction of voxels with EV ≥ threshold (log scale)",
        title="Survivor plot of explainable variance — all subjects",
        xlim=(0, 1),
        yscale="log",
    )
    survivor_axis.grid(True, which="both", alpha=0.3)
    survivor_axis.legend(title="Subject")
    survivor_figure.tight_layout()
    survivor_figure
    return


@app.cell(hide_code=True)
def _(ev_by_subject, excluded_by_subject, mo, np, shapes_by_subject, subjects):
    _header = (
        "| Subject | Total voxels | Valid EV | Undefined | Mean | Median | "
        "Std. dev. | Min | Max | EV ≥ 0.1 | EV ≥ 0.2 | EV ≥ 0.3 |\n"
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
    )
    _table_rows = []
    for _subject in subjects:
        _values = ev_by_subject[_subject]
        _total = shapes_by_subject[_subject][2]
        _percentages = [
            100 * np.mean(_values >= _threshold)
            for _threshold in (0.1, 0.2, 0.3)
        ]
        _table_rows.append(
            f"| {_subject} | {_total:,} | {_values.size:,} | "
            f"{excluded_by_subject[_subject]:,} | {_values.mean():.4f} | "
            f"{np.median(_values):.4f} | {_values.std():.4f} | "
            f"{_values.min():.4f} | {_values.max():.4f} | "
            f"{_percentages[0]:.2f}% | {_percentages[1]:.2f}% | "
            f"{_percentages[2]:.2f}% |"
        )

    mo.md(
        "### Summary statistics by subject\n\n"
        + _header
        + "\n"
        + "\n".join(_table_rows)
        + "\n\nThreshold percentages use all finite voxelwise EV estimates."
    )
    return


@app.cell(hide_code=True)
def _(excluded_by_subject, mo, shapes_by_subject):
    _rows = "\n".join(
        f"- **{subject}:** {shape[0]} repeats × {shape[1]} samples × "
        f"{shape[2]:,} voxels"
        + (
            f" ({excluded_by_subject[subject]} undefined EV values excluded)"
            if excluded_by_subject[subject]
            else ""
        )
        for subject, shape in shapes_by_subject.items()
    )
    mo.md(f"### Included data\n\n{_rows}")
    return


if __name__ == "__main__":
    app.run()
