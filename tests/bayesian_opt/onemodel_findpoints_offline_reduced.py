import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, ExpSineSquared, Matern, DotProduct, RationalQuadratic
from scipy.stats import norm
import argparse
from itertools import combinations
from scipy.spatial.distance import pdist, squareform

def get_furthest_points(points, k):
    """
    Return k points that are furthest from each other in space using a greedy approach.
    
    Args:
        points: Array of shape (n, d) where n is the number of points and d is the dimension
        k: Number of points to select
        
    Returns:
        Indices of the k selected points
    """
    n = len(points)
    
    if k > n:
        raise ValueError("k cannot be greater than the number of points provided.")
    
    if k == n:
        return list(range(n))
    
    # Calculate all pairwise distances once
    # Using scipy's pdist which is optimized for this purpose
    distances = squareform(pdist(points))
    
    # Start with the two points that are furthest apart
    max_dist = np.max(distances)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    
    selected = [i, j]
    
    # Greedily add points that are furthest from the already selected points
    while len(selected) < k:
        # For each unselected point, find its minimum distance to any selected point
        min_distances = np.zeros(n)
        for p in range(n):
            if p not in selected:
                # Find minimum distance to any selected point
                min_distances[p] = min(distances[p, s] for s in selected)
            else:
                min_distances[p] = -1  # Mark as already selected
        
        # Choose the point with the maximum minimum distance
        next_point = np.argmax(min_distances)
        selected.append(next_point)
    
    return selected

def read_and_normalize_high_fidelity_data(high_fidelity_file, num_params):
    # Read high fidelity data
    high_fidelity_data = pd.read_csv(high_fidelity_file)

    fields = [f'col{i+1}' for i in range(num_params)]  # Assuming the first num_params columns are the fields
    response = 'col' + str(num_params+1)  # Assuming the last column is the response variable
    #response = str("objective")  # Assuming the objective is the response variable
    
    # Use specified fields as x variables and response as y variable
    x_high_fidelity = high_fidelity_data[fields].values
    f_high_fidelity = high_fidelity_data[response].values

    # Normalize x variables to standard Gaussian (mean=0, std=1)
    x_mean = np.mean(x_high_fidelity, axis=0)
    x_std = np.std(x_high_fidelity, axis=0)
    x_high_fidelity = (x_high_fidelity - x_mean) / x_std

    # Normalize y variable to standard Gaussian (mean=0, std=1)
    f_high_fidelity_mean = np.mean(f_high_fidelity)
    f_high_fidelity_std = np.std(f_high_fidelity)
    f_high_fidelity = (f_high_fidelity - f_high_fidelity_mean) / f_high_fidelity_std

    return x_high_fidelity, f_high_fidelity, x_mean, x_std, f_high_fidelity_mean, f_high_fidelity_std

def fit_high_fidelity_model(x_high_fidelity, f_high_fidelity, kernel_type='RBF'):
    """Fit a Gaussian Process model on the high fidelity data."""
    # Choose kernel based on the kernel_type input
    if kernel_type == 'RBF':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    elif kernel_type == 'Matern':
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
    elif kernel_type == 'RationalQuadratic':
        kernel = ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
    elif kernel_type == 'ExpSineSquared':
        kernel = ConstantKernel(1.0) * ExpSineSquared(length_scale=1.0, periodicity=1.0)
    elif kernel_type == 'Combined':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + ConstantKernel(1.0) * WhiteKernel(noise_level=1)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(x_high_fidelity, f_high_fidelity)
    return gp

def dynamic_optimization(x_high_fidelity, f_high_fidelity, gp, known_idx, max_high_fidelity_evals=6, beta=1):
    f_best = np.max(f_high_fidelity[known_idx])
    x_best = x_high_fidelity[np.argmax(f_high_fidelity[known_idx])]
    index_best = known_idx[np.argmax(f_high_fidelity[known_idx])]

    for eval_iter in range(max_high_fidelity_evals):
        beta = min(1, max(beta * (max_high_fidelity_evals - eval_iter - 1) / (max_high_fidelity_evals - eval_iter), 0))
        candidate_idx = np.array([i for i in range(x_high_fidelity.shape[0]) if i not in known_idx])
        if candidate_idx.size == 0:
            print("No more candidate points available.")
            break
        mu, sigma = gp.predict(x_high_fidelity[candidate_idx], return_std=True)
        pred_high_fidelity = mu
        sigma_safe = np.maximum(sigma, 1e-8)

        Z = np.zeros_like(pred_high_fidelity)
        mask = sigma_safe > 1e-8
        Z[mask] = (pred_high_fidelity[mask] - f_best) / sigma_safe[mask]

        improvement = np.maximum(pred_high_fidelity - f_best, 0)
        EI = np.zeros_like(pred_high_fidelity)
        EI[mask] = improvement[mask] * norm.cdf(Z[mask]) + sigma_safe[mask] * norm.pdf(Z[mask])

        EI_percentile = (EI - EI.min()) / (EI.max() - EI.min() + 1e-10) if EI.size > 1 else np.ones_like(EI)
        uncertainty_percentile = (sigma_safe - sigma_safe.min()) / (sigma_safe.max() - sigma_safe.min() + 1e-10) if sigma_safe.size > 1 else np.ones_like(sigma_safe)
        
        acquisition = (1 - beta) * EI_percentile + beta * uncertainty_percentile
        best_candidate_idx = candidate_idx[np.argmax(acquisition)]

        new_high_fidelity_val = f_high_fidelity[best_candidate_idx]  # this line must be replaced with an actual function evaluation

        known_idx.append(best_candidate_idx)
        known_idx = sorted(list(set(known_idx)))

        if new_high_fidelity_val > f_best:
            f_best = new_high_fidelity_val
            x_best = x_high_fidelity[best_candidate_idx]
            index_best = best_candidate_idx

    return f_best, x_best, index_best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('high_fidelity_file', type=str, help='Path to high_fidelity_results CSV file.')
    parser.add_argument('--num_points', type=int, default=4, help='Number of points to select that are furthest apart.')
    parser.add_argument('--num_params', type=int, default=11, help='Number of parameters in the data.')
    parser.add_argument('--max_evals', type=int, default=6, help='Maximum number of evaluations.')
    parser.add_argument('--obj', type=str, choices=['max', 'min'], help='Specify whether to maximize or minimize the objective function.')
    parser.add_argument('--kernel', type=str, default='RBF', choices=['RBF', 'Matern', 'RationalQuadratic', 'ExpSineSquared', 'Combined'], help='Kernel type for Gaussian Process.')

    args = parser.parse_args()

    # Step 1: Read and normalize high-fidelity data
    x_high_fidelity, f_high_fidelity, x_mean, x_std, f_high_fidelity_mean, f_high_fidelity_std = read_and_normalize_high_fidelity_data(args.high_fidelity_file, args.num_params)
    if args.obj == 'min':
        f_high_fidelity = -f_high_fidelity
    # print best index
    best_index = np.argmax(f_high_fidelity)
    print(f"Best index in high fidelity data: {best_index}")
    print(f"Best value in high fidelity data: {f_high_fidelity[best_index]}")

    # Step 2: Choose the num_points that are the furthest apart in x_high_fidelity, excluding the best point
    # Exclude the best point from selection
    x_high_fidelity_nobest = np.delete(x_high_fidelity, best_index, axis=0)
    selected_indices = get_furthest_points(x_high_fidelity_nobest, args.num_points)
    x_selected = x_high_fidelity[selected_indices]
    f_selected = f_high_fidelity[selected_indices]

    # Step 3: Fit Gaussian Process model using the selected points
    gp = fit_high_fidelity_model(x_selected, f_selected, kernel_type=args.kernel)

    best_values = []
    best_xs = []
    best_indices = []
    # Step 5: Dynamic optimization loop
    for maxeval in range(1, args.max_evals + 1, 1):
        f_best, x_best , index_best = dynamic_optimization(x_high_fidelity, f_high_fidelity, gp, selected_indices, max_high_fidelity_evals=maxeval)
        best_values.append(f_best)
        best_xs.append(x_best)
        best_indices.append(index_best)
        print(f"Best point found at index {index_best} with value {f_best}")
        print("Number of evaluations: ", maxeval+args.num_points)
        if best_index == index_best:
            print("Premature stopping: best point found.")
            break
        
if __name__ == "__main__":
    main()
