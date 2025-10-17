import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import time

# Add current directory to sys.path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from onemodel_findpoints_online import online_optimization
from Run_SUMO_turn_online_check_function_v2 import (
    setup_sumo_environment, objective_function as fitness_func2
)

def main():
    # Set up SUMO environment and get required data
    NetworkName, SimName, SimulationStartTime, SimulationEndTime, CalibrationTarget, CablibrationInterval, DemandInterval, ubc = setup_sumo_environment()
    TurnDf = pd.read_excel(os.path.join(
        r"C:\Users\xh8\Downloads\transport\OnlineTests\SimulationModels\Sumo\BeforeTurn",
        'turn.xlsx'))
    InflowDf = pd.read_excel(os.path.join(
        r"C:\Users\xh8\Downloads\transport\OnlineTests\SimulationModels\Sumo\BeforeTurn",
        'inflow.xlsx'))

    def evaluation_function(x):
        solution = np.array(x).tolist()
        print(f"\n=== TESTING SOLUTION ===")
        print(f"Turn ratios [0-9]: {[f'{val:.3f}' for val in solution[:10]]}")
        print(f"Inflow counts [10-13]: {[f'{val:.1f}' for val in solution[10:]]}")
        print("Calling SUMO simulation...")

        try:
            fitness = fitness_func2(
                solution, TurnDf, InflowDf, NetworkName, SimulationStartTime, SimulationEndTime,
                CablibrationInterval, DemandInterval, SimName, CalibrationTarget
            )
            print(f"SUCCESS - Fitness result: {fitness:.3f}")
        except Exception as e:
            print(f"ERROR in fitness evaluation: {e}")
            fitness = float('inf')  # Return high penalty for failed evaluations

        print("=== SOLUTION COMPLETE ===\n")
        return fitness

    # Set up optimization parameters (adjust as needed)
    num_params = 14
    lower_bounds = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    upper_bounds = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 200., 200., 200., 200.])
    variable_bounds = (lower_bounds, upper_bounds)
    tolerance = 3
    target = 0
    random_points = 4000
    max_evaluations = 100
    kernel_type = 'RBF'

    all_rows = []
    best_rows = []
    run_times = []

    for run in range(10):
        print(f"\n--- Run {run+1}/10 ---")
        print("Starting SUMO-based Bayesian optimization (turn scenario)...")
        print(f"Dimensions: {num_params}")
        print(f"Bounds: {variable_bounds}")
        print(f"Target: {target} with tolerance: {tolerance}")
        print(f"Maximum evaluations: {max_evaluations}")

        start_time = time.time()
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

        print("\nResults:")
        print(f"Best function value found: {f_best}")
        print(f"Best point found: {x_best}")
        print(f"Number of evaluations performed: {len(evaluated_values)}")
        print(f"Elapsed time for run {run+1}: {elapsed_time:.2f} seconds")
        run_success = (f_best - target) <= tolerance
        if run_success:
            print(f"Run {run+1}: Test PASSED! ✓ Found point {x_best} with value {f_best}")
        else:
            print(f"Run {run+1}: Test FAILED! ✗ Could not reach target within tolerance. Best value: {f_best}")

        for iteration, (fitness, point) in enumerate(zip(evaluated_values, evaluated_points)):
            row = {
                'run': run + 1,
                'iteration': iteration + 1,
                'fitness': fitness,
                'run_time_sec': elapsed_time if (iteration == len(evaluated_values) - 1) else ""
            }
            for i, v in enumerate(point):
                row[f'param_{i+1}'] = v
            all_rows.append(row)

        best_idx = int(np.argmin(evaluated_values))
        best_row = {
            'run': run + 1,
            'iteration': best_idx + 1,
            'fitness': evaluated_values[best_idx],
            'run_time_sec': elapsed_time
        }
        for i, v in enumerate(np.array(evaluated_points)[best_idx]):
            best_row[f'param_{i+1}'] = v
        best_rows.append(best_row)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(evaluated_values) + 1), evaluated_values, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title(f'Fitness vs Iteration for Run {run+1}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'fitness_vs_iteration_turn_run{run+1}.png')
        plt.close()

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv("sumo_bayesopt_turn_all_iterations.csv", index=False)
    print("\nSaved all parameter values and fitness for each run/iteration to sumo_bayesopt_turn_all_iterations.csv")

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv("sumo_bayesopt_turn_best_per_run.csv", index=False)
    print("Saved best fitness/iteration/parameters for each run to sumo_bayesopt_turn_best_per_run.csv")

    print("\nSummary of bests for each run:")
    print(best_df)

    overall_best_idx = best_df['fitness'].idxmin()
    overall_best = best_df.loc[overall_best_idx]
    print(f"\nOverall best: Run {int(overall_best['run'])}, Iteration {int(overall_best['iteration'])}, Fitness {overall_best['fitness']}")
    print(f"Parameters: {[overall_best[f'param_{i+1}'] for i in range(num_params)]}")

    success = (overall_best['fitness'] - target) <= tolerance
    if success:
        print("\nTest PASSED! ✓")
    else:
        print("\nTest FAILED! ✗")

if __name__ == "__main__":
    main()