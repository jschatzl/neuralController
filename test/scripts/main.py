# SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
# SPDX-License-Identifier: Apache-2.0

# @file main.py
# @author Jakob Schatzl
# @brief Main script for running experiments with the neural controller
# @version 1.0
# @date 2026-11-17

import gc
import matplotlib.pyplot as plt
import plot
import metrics

def main():
    gc.disable()

    p2_values = list(range(0, 1))
    # p2_values = [True, False]
    repetitions = 100
    setpoint = 0.5

    experiment_data, metrics_df = metrics.run_experiments(
        p2_values=p2_values,
        repetitions=repetitions,
        test_parameters=metrics.testParameters,
        simulation_steps=10000,
        initial_learning_rate=0.01,
        setpoint=setpoint,
        option=True,
    )

    plot.plot_all_experiments(experiment_data)

    average_metrics = metrics.calculate_average_metrics(
        metrics_df,
    )

    print("\nMetrics for every individual run:")
    print(metrics_df.to_string(index=False))

    print("\nMean metrics per p2:")
    print(average_metrics.to_string(index=False))

    report = metrics.create_short_test_report(
        average_metrics
    )

    print("\ntest report:")
    print(report)

    plt.show()
    

if __name__ == "__main__":
    main()
