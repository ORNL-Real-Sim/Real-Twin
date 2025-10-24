'''
##############################################################
# Created Date: Friday, October 24th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from Run_SUMO_behavior_online_check_function_v2 import setup_sumo_environment, fitness_func2
from onemodel_findpoints_online import online_optimization
from time import time
import numpy as np
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import time


class BayesianOptimization:
    """ Bayesian Optimization for calibrating turn inflow parameters. """

    def __init__(self, num_params: int, lower_bounds: np.array, upper_bounds: np.array, tolerance: int = 10, target: int = 0, random_points: int = 50000, max_evaluations: int = 100, kernel_type: str = 'RBF'):
        self.num_params = num_params
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.variable_bounds = (lower_bounds, upper_bounds)
        self.tolerance = tolerance
        self.target = target
        self.random_points = random_points
        self.max_evaluations = max_evaluations
        self.kernel_type = kernel_type
        self.all_results = []
        self.all_points = []
        self.run_times = []

    def optimize(self, total_run: int = 10):
        """ Perform Bayesian Optimization over multiple runs. """

        # TDD
        if not isinstance(total_run, int):
            raise TypeError("Total runs must be an integer.")
        if total_run <= 0:
            raise ValueError("Total runs must be a positive integer.")

        # Initialize min/max trackers
        param_mins = np.full(self.num_params, np.inf)
        param_maxs = np.full(self.num_params, -np.inf)
        fitness_global_min = np.inf
        fitness_global_max = -np.inf
        temp_evaluated_values = []
        temp_evaluated_points = []

        for run in range(total_run):
            print(f"\n--- Run {run + 1}/{total_run} ---")
            print("Starting SUMO-based Bayesian optimization...")
            print(f"Dimensions: {self.num_params}")
            print(f"lb: {self.lower_bounds}, ub: {self.upper_bounds}")
            print(f"Target: {self.target} with tolerance: {self.tolerance}")
            print(f"Maximum evaluations: {self.max_evaluations}")

            start_time = time.time()

            # Run the optimization
            f_best, x_best, evaluated_points, evaluated_values = online_optimization(
                num_params=self.num_params,
                variable_bounds=self.variable_bounds,
                evaluation_function=fitness_func2,
                tolerance=self.tolerance,
                target=self.target,
                random_points=self.random_points,
                max_evaluations=self.max_evaluations,
                kernel_type=self.kernel_type
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            self.run_times.append(elapsed_time)

            print("\nResults:")
            print(f"Best function value found: {f_best}")
            print(f"Best point found: {x_best}")
            print(f"Number of evaluations performed: {len(evaluated_values)}")
            print(f"Elapsed time for run {run + 1}: {elapsed_time:.2f} seconds")

            temp_evaluated_values.append(evaluated_values)
            temp_evaluated_points.append(evaluated_points)

            # Update global fitness min/max
            fitness_global_min = min(fitness_global_min, np.min(evaluated_values))
            fitness_global_max = max(fitness_global_max, np.max(evaluated_values))

            # Update global param min/max
            evaluated_points_np = np.array(evaluated_points)
            param_mins = np.minimum(param_mins, np.min(evaluated_points_np, axis=0))
            param_maxs = np.maximum(param_maxs, np.max(evaluated_points_np, axis=0))

            # Record fitness and iteration for this run
            for iteration, (fitness, point) in enumerate(zip(evaluated_values, evaluated_points)):
                self.all_results.append({
                    'run': run + 1,
                    'iteration': iteration + 1,
                    'fitness': fitness,
                    'run_time_sec': elapsed_time if (iteration == len(evaluated_values) - 1) else ""
                })
                # Store parameter values for each iteration
                self.all_points.append({
                    'run': run + 1,
                    'iteration': iteration + 1,
                    **{f'param_{i + 1}': v for i, v in enumerate(point)},
                    'fitness': fitness,
                    'run_time_sec': elapsed_time if (iteration == len(evaluated_values) - 1) else ""
                })

            # After each run, print pass/fail for this run
            run_success = (f_best - self.target) <= self.tolerance
            if run_success:
                print(f"Run {run + 1}: Test PASSED! ✓ Found point {x_best} with value {f_best}")
            else:
                print(f"Run {run + 1}: Test FAILED! ✗ Could not reach target within tolerance. Best value: {f_best}")

        # Save all results to a DataFrame and CSV
        results_df = pd.DataFrame(self.all_results)
        results_df.to_csv("sumo_bayesopt_10runs_results.csv", index=False)
        print("\nSaved all fitness and iteration results to sumo_bayesopt_10runs_results.csv")

        points_df = pd.DataFrame(self.all_points)
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
                best_row['run_time_sec'] = self.run_times[run - 1]
                best_rows.append(best_row)
        best_df = pd.DataFrame(best_rows)
        best_df.to_csv("sumo_bayesopt_best_per_run.csv", index=False)
        print("Saved best fitness/iteration/parameters for each run to sumo_bayesopt_best_per_run.csv")

        # Plot fitness vs iteration for each run with fixed y-limits
        for run in range(total_run):
            evaluated_values = temp_evaluated_values[run]
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(evaluated_values) + 1), evaluated_values, marker='o')
            plt.xlabel('Iteration')
            plt.ylabel('Fitness')
            plt.title(f'Fitness Convergence for Run {run + 1}')
            plt.ylim(fitness_global_min, fitness_global_max)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'fitness_vs_iteration_run{run + 1}.png')
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
        success = (best_fitness - self.target) <= self.tolerance
        if success:
            print("\nTest PASSED! ✓")
        else:
            print("\nTest FAILED! ✗")
            print(f"Could not reach target within tolerance. Best value: {best_fitness}")