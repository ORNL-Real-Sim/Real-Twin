import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import time

# Add current directory to sys.path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from onemodel_findpoints_online import online_optimization
from Run_SUMO_behavior_online_check_function_v2 import setup_sumo_environment, fitness_func2

def main():
    # Set up SUMO environment and get required data
    filePath, fPath_orig, EB_tt, WB_tt, EB_edge_list, WB_edge_list = setup_sumo_environment()

    def evaluation_function(x):
        solution = np.array(x).tolist()
        fitness = fitness_func2(solution, filePath, EB_tt, WB_tt, EB_edge_list, WB_edge_list)
        return fitness

    # Set up optimization parameters (adjust as needed)
    num_params = 6  # Change if your SUMO solution vector has a different length
    lower_bounds = np.array([1.0, 2.5, 4.0, 0.0, 0.25, 5.0])
    upper_bounds = np.array([3.0, 3.0, 5.3, 1.0, 1.25, 9.3])
    variable_bounds = (lower_bounds, upper_bounds)
    tolerance = 10
    target = 0
    random_points = 50000
    max_evaluations = 100
    kernel_type = 'RBF'

    # DataFrame to record all runs
    all_results = []
    all_points = []
    run_times = []
    param_mins = np.full(num_params, np.inf)
    param_maxs = np.full(num_params, -np.inf)

    # First, collect min/max for fitness and parameters across all runs
    fitness_global_min = np.inf
    fitness_global_max = -np.inf

    temp_evaluated_values = []
    temp_evaluated_points = []

    for run in range(10):
        print(f"\n--- Run {run+1}/10 ---")
        print("Starting SUMO-based Bayesian optimization...")
        print(f"Dimensions: {num_params}")
        print(f"Bounds: {variable_bounds}")
        print(f"Target: {target} with tolerance: {tolerance}")
        print(f"Maximum evaluations: {max_evaluations}")

        start_time = time.time()

        # Run the optimization
        f_best, x_best, evaluated_points, evaluated_values = online_optimization(
            num_params=num_params,
            variable_bounds=variable_bounds,
            evaluation_function=evaluation_function,
            tolerance=tolerance,
            target=target,
            random_points=random_points,
            max_evaluations=max_evaluations,
            kernel_type=kernel_type
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        run_times.append(elapsed_time)

        temp_evaluated_values.append(evaluated_values)
        temp_evaluated_points.append(evaluated_points)

        # Update global fitness min/max
        fitness_global_min = min(fitness_global_min, np.min(evaluated_values))
        fitness_global_max = max(fitness_global_max, np.max(evaluated_values))

        # Update global param min/max
        evaluated_points_np = np.array(evaluated_points)
        param_mins = np.minimum(param_mins, np.min(evaluated_points_np, axis=0))
        param_maxs = np.maximum(param_maxs, np.max(evaluated_points_np, axis=0))

        print("\nResults:")
        print(f"Best function value found: {f_best}")
        print(f"Best point found: {x_best}")
        print(f"Number of evaluations performed: {len(evaluated_values)}")
        print(f"Elapsed time for run {run+1}: {elapsed_time:.2f} seconds")

        # Record fitness and iteration for this run
        for iteration, (fitness, point) in enumerate(zip(evaluated_values, evaluated_points)):
            all_results.append({
                'run': run + 1,
                'iteration': iteration + 1,
                'fitness': fitness,
                'run_time_sec': elapsed_time if (iteration == len(evaluated_values) - 1) else ""
            })
            # Store parameter values for each iteration
            all_points.append({
                'run': run + 1,
                'iteration': iteration + 1,
                **{f'param_{i+1}': v for i, v in enumerate(point)},
                'fitness': fitness,
                'run_time_sec': elapsed_time if (iteration == len(evaluated_values) - 1) else ""
            })

        # After each run, print pass/fail for this run
        run_success = (f_best - target) <= tolerance
        if run_success:
            print(f"Run {run+1}: Test PASSED! ✓ Found point {x_best} with value {f_best}")
        else:
            print(f"Run {run+1}: Test FAILED! ✗ Could not reach target within tolerance. Best value: {f_best}")

    # Save all results to a DataFrame and CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("sumo_bayesopt_10runs_results.csv", index=False)
    print("\nSaved all fitness and iteration results to sumo_bayesopt_10runs_results.csv")

    points_df = pd.DataFrame(all_points)
    points_df.to_csv("sumo_bayesopt_10runs_points.csv", index=False)
    print("Saved all parameter values for each iteration to sumo_bayesopt_10runs_points.csv")

    # Create a DataFrame for best results per run (lowest fitness), including run time
    best_rows = []
    for run in range(1, 11):
        run_df = points_df[points_df['run'] == run]
        if not run_df.empty:
            best_idx = run_df['fitness'].idxmin()
            best_row = run_df.loc[best_idx].copy()
            # Ensure run_time_sec is filled for the best row
            best_row['run_time_sec'] = run_times[run-1]
            best_rows.append(best_row)
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv("sumo_bayesopt_best_per_run.csv", index=False)
    print("Saved best fitness/iteration/parameters for each run to sumo_bayesopt_best_per_run.csv")

    # Plot fitness vs iteration for each run with fixed y-limits
    for run in range(10):
        evaluated_values = temp_evaluated_values[run]
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(evaluated_values) + 1), evaluated_values, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title(f'Fitness Convergence for Run {run+1}')
        plt.ylim(fitness_global_min, fitness_global_max)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'fitness_vs_iteration_run{run+1}.png')
        plt.close()

    # Find the best run (lowest fitness)
    best_run_row = best_df.sort_values('fitness').iloc[0]
    best_run = best_run_row['run']
    best_fitness = best_run_row['fitness']
    print(f"\nBest run: {int(best_run)}")

    # Filter points for the best run
    best_run_fitness = results_df[results_df['run'] == best_run].sort_values('iteration')['fitness'].values

    # Plot fitness convergence for the best run and save (with fixed y-limits)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(best_run_fitness) + 1), best_run_fitness, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Fitness Convergence for Best Run')
    plt.ylim(fitness_global_min, fitness_global_max)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('best_run_fitness_convergence.png')
    plt.close()

    # Test passed/failed message for best run
    success = (best_fitness - target) <= tolerance
    if success:
        print("\nTest PASSED! ✓")
    else:
        print("\nTest FAILED! ✗")
        print(f"Could not reach target within tolerance. Best value: {best_fitness}")

if __name__ == "__main__":
    main()