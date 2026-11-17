# SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
# SPDX-License-Identifier: Apache-2.0

# @file metrics.py
# @author Jakob Schatzl
# @brief Metrics calculation for neural controller experiments
# @version 1.0
# @date 2026-11-17

import numpy as np
import pandas as pd
import learnrate as lr
from matplotlib import colormaps
from neuralControllerClass import NeuralController
from pt2_rk4 import PT2

def calculate_metrics(
    data,
    setpoint,
    settling_tolerance=0.02,
):
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise TypeError(
            "calculate_metrics() expects one 2-D NumPy array "
            "with columns [t, y, u]. "
            f"Received type={type(data).__name__}, "
            f"shape={getattr(data, 'shape', None)}"
        )

    if data.shape[1] < 2:
        raise ValueError(
            "data must contain at least columns [t, y]"
        )

    t = data[:, 0]
    y = data[:, 1]

    if len(t) < 2:
        raise ValueError(
            "At least two samples are required"
        )

    if np.any(np.diff(t) <= 0):
        raise ValueError(
            "Time values must be strictly increasing"
        )

    error = setpoint - y
    ise = np.trapezoid(error**2, t)

    final_value = y[-1]

    if setpoint >= 0.0:
        peak_value = np.max(y)
        max_overshoot = max(
            0.0,
            peak_value - setpoint,
        )
    else:
        peak_value = np.min(y)
        max_overshoot = max(
            0.0,
            setpoint - peak_value,
        )

    if abs(setpoint) > 1e-12:
        overshoot_percent = (
            100.0
            * max_overshoot
            / abs(setpoint)
        )
    else:
        overshoot_percent = np.nan

    band = settling_tolerance * max(
        abs(setpoint),
        1e-12,
    )

    inside_band = np.abs(y - setpoint) <= band
    settling_time = np.inf

    for i in range(len(t)):
        if inside_band[i] and np.all(inside_band[i:]):
            settling_time = t[i]
            break

    return {
        "ise": ise,
        "final_value": final_value,
        "max_overshoot": max_overshoot,
        "overshoot_percent": overshoot_percent,
        "settling_time": settling_time,
    }


def add_setpoint_column(data, setpoint):
    t = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]

    r = np.full_like(t, setpoint)

    return np.column_stack((t, r, y, u))

def min_max_normalize(series):
    minimum = series.min()
    maximum = series.max()

    if np.isclose(minimum, maximum):
        return pd.Series(0.0, index=series.index)

    return (series - minimum) / (maximum - minimum)


def min_max_normalize(series):
    minimum = series.min()
    maximum = series.max()

    if np.isclose(minimum, maximum):
        return pd.Series(0.0, index=series.index)

    return (series - minimum) / (maximum - minimum)

def calculate_average_metrics(metrics_df):
    """
    Calculate mean and variance for each p2 value.
    """

    return (
        metrics_df
        .groupby("p2", as_index=False)
        .agg(
            ise_mean=("ise", "mean"),
            ise_variance=("ise", "var"),

            final_value_mean=("final_value", "mean"),
            final_value_variance=(
                "final_value",
                "var",
            ),

            max_overshoot_mean=(
                "max_overshoot",
                "mean",
            ),
            max_overshoot_variance=(
                "max_overshoot",
                "var",
            ),

            overshoot_percent_mean=(
                "overshoot_percent",
                "mean",
            ),
            overshoot_percent_variance=(
                "overshoot_percent",
                "var",
            ),

            settling_time_mean=(
                "settling_time",
                "mean",
            ),
            settling_time_variance=(
                "settling_time",
                "var",
            ),
        )
    )

def run_experiments(
    p2_values,
    repetitions,
    test_parameters,
    simulation_steps,
    initial_learning_rate,
    setpoint,
    option=True,
):
    """
    Execute all experiments and return plot data and metric results.
    """

    colors = colormaps["viridis"](
        np.linspace(0.1, 0.9, len(p2_values))
    )

    experiment_data = []
    metric_results = []

    for p2, color in zip(p2_values, colors):
        runs = []

        for _ in range(repetitions):
            data = test_parameters(
                hidden_layers=1,
                neurons=16,
                max_epochs=simulation_steps,
                learning_rate=initial_learning_rate,
                learning_rate_method=0,
                setpoint=setpoint,
                isJordan=option,
                initMethod=1
            )
            runs.append(data)

        calculate_and_export_average_run(
        runs=runs,
        filename=f"average_run_p2_{p2:.0f}.csv",
        )

        experiment_data.append({
            "p2": p2,
            "color": color,
            "runs": runs,
        })

        # Calculate metrics for every individual repetition
        for repetition, run in enumerate(runs, start=1):
            metrics = calculate_metrics(
                run,
                setpoint=setpoint,
            )

            metric_results.append({
                "p2": p2,
                "repetition": repetition,
                **metrics,
            })

    metrics_df = pd.DataFrame(metric_results)

    return experiment_data, metrics_df

def calculate_and_export_average_run(
    runs,
    filename="average_run.csv",
):
    """
    Calculate the average run from multiple repetitions
    and export it as a CSV file with columns t,y.
    """

    if len(runs) == 0:
        raise ValueError(
            "At least one run is required."
        )

    runs = [
        np.asarray(run, dtype=float)
        for run in runs
    ]

    first_run = runs[0]

    if first_run.ndim != 2 or first_run.shape[1] < 2:
        raise ValueError(
            "Each run must be a 2-D array with "
            "at least columns [t, y]."
        )

    expected_shape = first_run.shape

    for run in runs:
        if run.shape != expected_shape:
            raise ValueError(
                "All runs must have the same shape."
            )

    t = first_run[:, 0]

    y_values = np.stack(
        [run[:, 1] for run in runs],
        axis=0,
    )

    y_average = np.mean(
        y_values,
        axis=0,
    )

    average_run = np.column_stack(
        (t, y_average)
    )

    average_df = pd.DataFrame(
        average_run,
        columns=["t", "y"],
    )

    average_df.to_csv(
        filename,
        index=False,
    )

    return average_run

def create_short_test_report(average_metrics):
    lines = [
        ";".join([
            "Neurons",
            "ISE mean",
            "ISE variance",
            "Final value mean",
            "Maximum overshoot mean",
            "Overshoot mean [%]",
            "Overshoot variance [%^2]",
            "Settling time mean [s]",
            "Settling time variance [s^2]",
        ])
    ]

    for _, row in average_metrics.iterrows():
        lines.append(";".join([
            f"{row['p2']:.0f}",
            f"{row['ise_mean']:.6g}",
            f"{row['ise_variance']:.6g}",
            f"{row['final_value_mean']:.6f}",
            f"{row['max_overshoot_mean']:.6f}",
            f"{row['overshoot_percent_mean']:.3f}",
            f"{row['overshoot_percent_variance']:.6g}",
            f"{row['settling_time_mean']:.4f}",
            f"{row['settling_time_variance']:.6g}",
        ]))

    return "\n".join(lines)

def testParameters(
    hidden_layers: int,
    neurons: int,
    max_epochs: int,
    learning_rate: float,
    learning_rate_method: int,
    setpoint: float,
    isJordan: bool,
    initMethod: int
):
    dt = 0.001

    t = np.arange(max_epochs, dtype=float) * dt
    y = np.zeros(max_epochs, dtype=float)
    u = np.zeros(max_epochs, dtype=float)

    pt2 = PT2(
        K=1.0,
        T=0.5,
        D=1.0,
        dt=dt,
    )

    neuralController = NeuralController(
        hidden_layers,
        neurons,
        setpoint,
        isJordan,
        initMethod,
    )

    step_size = max_epochs // 8
    step_size_ld = max_epochs // 5
    gamma = 0.1
    min_learning_rate = learning_rate * gamma
    final_learning_rate = min_learning_rate
    reference_error = abs(setpoint)

    for i in range(max_epochs - 1):
        if learning_rate_method == 0:
            learning_rate_rel = lr.step_decay_lr(
                i,
                learning_rate,
                step_size,
                gamma,
            )

            learning_rate_rel = max(
                min_learning_rate,
                learning_rate_rel,
            )

        elif learning_rate_method == 1:
            learning_rate_rel = lr.linear_decay_lr(
                i,
                learning_rate * 0.7,
                final_learning_rate * 0.1,
                step_size_ld,
            )

        elif learning_rate_method == 2:
            error = abs(setpoint - y[i])

            learning_rate_rel = lr.error_dependent_lr(
                error,
                min_learning_rate,
                0.3*learning_rate,
                reference_error,
            )

        else:
            raise ValueError(
                "learning_rate_method must be 0, 1, or 2"
            )

        u[i + 1] = neuralController.run(
            y[i],
            learning_rate_rel,
            setpoint,
        )

        y[i + 1] = pt2.step(u[i + 1])

        if i % 1000 == 0:
            error = setpoint - y[i]

            print(
                f"Epoch: {i} | "
                f"Setpoint: {setpoint:.3f} | "
                f"Plant output: {y[i]:.6f} | "
                f"u: {u[i]:.6f} | "
                f"Error: {error:.6f} | "
                f"Learning rate: {learning_rate_rel:.8f}"
            )

    data = np.column_stack((t, y, u))

    del neuralController

    return data