# SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
# SPDX-License-Identifier: Apache-2.0

# @file plot.py
# @author Jakob Schatzl
# @brief Plotting utilities for neural controller experiments
# @version 1.0
# @date 2026-11-17

import matplotlib.pyplot as plt
import numpy as np

def plot_response(
    ax,
    runs,
    p2,
    color,
    signal_column,
    signal_name,
):
    """
    Plot all individual runs in the background and their average
    in the foreground.
    """
    t = runs[0][:, 0]
    signal_all = np.asarray(
        [run[:, signal_column] for run in runs]
    )
    # Individual runs
    for signal in signal_all:
        ax.plot(
            t,
            signal,
            color=color,
            alpha=0.10,
            linewidth=1.0,
        )
    # Average run
    signal_average = signal_all.mean(axis=0)
    ax.plot(
        t,
        signal_average,
        color=color,
        linewidth=2.5,
        label=f"p2 = {p2}",
        zorder=10,
    )
    return t, signal_all, signal_average

def configure_plot(ax, title, y_label):
    """Apply common formatting to a plot."""

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.legend(title="Average runs")

def plot_all_experiments(experiment_data):
    """
    Create one figure for y(t) and one figure for u(t).
    """

    fig_y, ax_y = plt.subplots(figsize=(10, 6))
    fig_u, ax_u = plt.subplots(figsize=(10, 6))
    for experiment in experiment_data:
        plot_response(
            ax=ax_y,
            runs=experiment["runs"],
            p2=experiment["p2"],
            color=experiment["color"],
            signal_column=1,
            signal_name="y(t)",
        )
        plot_response(
            ax=ax_u,
            runs=experiment["runs"],
            p2=experiment["p2"],
            color=experiment["color"],
            signal_column=2,
            signal_name="u(t)",
        )
    configure_plot(
        ax_y,
        title="Plant output for different p2 values",
        y_label="y(t)",
    )
    configure_plot(
        ax_u,
        title="Controller output for different p2 values",
        y_label="u(t)",
    )
    fig_y.tight_layout()
    fig_u.tight_layout()
    return fig_y, fig_u