import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV with best fitness and parameter values for each run
best_df = pd.read_csv("sumo_bayesopt_10runs_results.csv")
points_df = pd.read_csv("sumo_bayesopt_10runs_points.csv")

# Get best fitness and corresponding parameters for each run
best_per_run = best_df.loc[best_df.groupby('run')['fitness'].idxmin()].sort_values('run')
param_cols = [col for col in points_df.columns if col.startswith('param_')]

# Merge best fitness with parameter values
best_params = []
for idx, row in best_per_run.iterrows():
    run = row['run']
    iteration = row['iteration']
    params_row = points_df[(points_df['run'] == run) & (points_df['iteration'] == iteration)]
    if not params_row.empty:
        best_params.append(params_row[param_cols].values[0])
    else:
        best_params.append([np.nan]*len(param_cols))
best_params = np.array(best_params)

# Plot boxplot for best fitness across runs
plt.figure(figsize=(7, 5))
plt.boxplot(best_per_run['fitness'], vert=True, patch_artist=True, labels=['Best Fitness'])
plt.ylabel('Best Fitness')
plt.title('Distribution of Best Fitness Across 10 Runs')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('boxplot_best_fitness_across_runs.png')
plt.close()
print("Saved boxplot of best fitness across runs to boxplot_best_fitness_across_runs.png")

# Plot boxplot for each parameter (show bounds)
lower_bounds = np.array([1.0, 2.5, 4.0, 0.0, 0.25, 5.0])
upper_bounds = np.array([3.0, 3.0, 5.3, 1.0, 1.25, 9.3])

plt.figure(figsize=(10, 6))
bplot = plt.boxplot([best_params[:, i] for i in range(len(param_cols))],
                    labels=[f'param_{i+1}' for i in range(len(param_cols))],
                    patch_artist=True, showmeans=True)
for i, (low, high) in enumerate(zip(lower_bounds, upper_bounds)):
    plt.hlines([low, high], i+0.7, i+1.3, colors='red', linestyles='dashed', linewidth=1)
plt.xlabel('Parameter')
plt.ylabel('Value')
plt.title('Distribution of Best Parameter Values Across 10 Runs')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('boxplot_best_params_across_runs.png')
plt.close()
print("Saved boxplot of best parameter values across runs to boxplot_best_params_across_runs.png")

# Print quantiles for each parameter
quantiles = [0, 0.25, 0.5, 0.75, 1.0]
print("\nParameter quantiles (min, 25%, 50%, 75%, max):")
for i, col in enumerate(param_cols):
    q = np.quantile(best_params[:, i], quantiles)
    print(f"{col}: {q}")

# --- Boxplot for time taken for all iterations in each run ---
# Extract run_time_sec for each run and iteration
points_df['run_time_sec'] = pd.to_numeric(points_df['run_time_sec'], errors='coerce')
run_times_per_run = []
num_iterations_per_run = []

for run in sorted(points_df['run'].unique()):
    run_times = points_df[points_df['run'] == run]['run_time_sec'].dropna().values
    if len(run_times) == 0:
        continue
    run_times_per_run.append(run_times)
    num_iterations_per_run.append(len(points_df[points_df['run'] == run]))

plt.figure(figsize=(10, 6))
plt.boxplot(run_times_per_run, labels=[f'Run {i+1}\n({n} iters)' for i, n in enumerate(num_iterations_per_run)], patch_artist=True)
plt.ylabel('Time (sec)')
plt.xlabel('Run (number of iterations)')
plt.title('Distribution of Time Taken per Run (last iteration time per run)')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('boxplot_time_per_run.png')
plt.close()
print("Saved boxplot of time taken for all iterations in each run to boxplot_time_per_run.png")