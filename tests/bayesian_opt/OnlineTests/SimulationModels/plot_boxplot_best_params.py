import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV with best fitness and parameter values for each run
best_df = pd.read_csv("sumo_bayesopt_turn_best_per_run.csv")

# Collect parameter columns
param_cols = [col for col in best_df.columns if col.startswith('param_')]
param_data = best_df[param_cols]

# Define parameter bounds (first 10: [0,1], last 4: [0,200])
lower_bounds = [0.0]*10 + [0.0]*4
upper_bounds = [1.0]*10 + [200.0]*4

# Make boxplot for parameters 1-10
plt.figure(figsize=(12, 6))
plt.boxplot([param_data[f'param_{i+1}'] for i in range(10)], labels=[f'param_{i+1}' for i in range(10)])
for i in range(10):
    plt.hlines([lower_bounds[i], upper_bounds[i]], i+0.7, i+1.3, colors='red', linestyles='dashed', linewidth=1)
plt.xlabel('Parameter')
plt.ylabel('Value')
plt.title('Distribution of Best Parameter Values (Parameters 1-10) Across 10 Runs')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('boxplot_best_params_1_10_turn.png')
plt.close()
print("Saved boxplot for parameters 1-10 to boxplot_best_params_1_10_turn.png")

# Make boxplot for parameters 11-14
plt.figure(figsize=(8, 6))
plt.boxplot([param_data[f'param_{i+1}'] for i in range(10, 14)], labels=[f'param_{i+1}' for i in range(10, 14)])
for i in range(10, 14):
    plt.hlines([lower_bounds[i], upper_bounds[i]], i-9+0.7, i-9+1.3, colors='red', linestyles='dashed', linewidth=1)
plt.xlabel('Parameter')
plt.ylabel('Value')
plt.title('Distribution of Best Parameter Values (Parameters 11-14) Across 10 Runs')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('boxplot_best_params_11_14_turn.png')
plt.close()
print("Saved boxplot for parameters 11-14 to boxplot_best_params_11_14_turn.png")

# --- Boxplot for time taken for each run ---
if 'run_time_sec' in best_df.columns:
    plt.figure(figsize=(8, 5))
    plt.boxplot(best_df['run_time_sec'], vert=True, patch_artist=True, labels=['Run Time (sec)'])
    for i, t in enumerate(best_df['run_time_sec']):
        plt.text(1, t, f"{t:.1f}", ha='center', va='bottom', fontsize=8, color='blue')
    plt.ylabel('Time (sec)')
    plt.title('Distribution of Time Taken for Each Run (Best Result)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('boxplot_time_per_run_turn.png')
    plt.close()
    print("Saved boxplot of time taken for each run to boxplot_time_per_run_turn.png")
else:
    print("Column 'run_time_sec' not found in best results CSV. No time plot generated.")