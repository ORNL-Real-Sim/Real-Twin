import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, ExpSineSquared, Matern, DotProduct, RationalQuadratic
from scipy.stats import norm
import generate_random_points as grp
import matplotlib.pyplot as plt


def fit_high_fidelity_model(x_high_fidelity, f_high_fidelity, kernel_type='RBF'):
    """Fit a Gaussian Process model on the high fidelity data."""
    # Choose kernel based on the kernel_type input
    if kernel_type == 'RBF':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    elif kernel_type == 'Matern':
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
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

def dynamic_online_optimization(x_norm, f_norm, x_mean, x_std, f_mean, f_std, evaluation_function, gp, candidate_points, target, tolerance, max_evals, kernel='RBF', beta=1, minimize=True):
    """
    Performs dynamic optimization to minimize or maximize the objective function.
    
    Parameters:
    -----------
    x_norm : array-like
        Normalized feature data points that have been evaluated
    f_norm : array-like
        Normalized function values corresponding to x_norm
    x_mean : array-like
        Mean values used for feature normalization
    x_std : array-like
        Standard deviation values used for feature normalization
    f_mean : float
        Mean value used for objective function normalization
    f_std : float
        Standard deviation used for objective function normalization
    evaluation_function : callable
        Function that evaluates new points in unnormalized space
    gp : object
        Gaussian Process model fitted on normalized data
    candidate_points : array-like
        Normalized candidate points to choose from for evaluation
    target : float
        Target value to reach (in unnormalized space)
    tolerance : float
        Tolerance around the target (in unnormalized space)
    max_evals : int
        Maximum number of high-fidelity evaluations to perform
    beta : float, optional
        Exploration-exploitation trade-off parameter (default=1)
    minimize : bool, optional
        If True, minimize the objective function; if False, maximize it (default=True)
    
    Returns:
    --------
    f_best : float
        Best function value found (unnormalized)
    x_best : array-like
        Point corresponding to best function value (unnormalized)
    evaluated_points : list
        List of points evaluated during optimization (unnormalized)
    evaluated_values : list
        List of function values for evaluated points (unnormalized)
    """
    # Initialize with the current best value based on optimization direction
    if minimize:
        f_best_norm = np.min(f_norm)
        best_index = np.argmin(f_norm)
    else:
        f_best_norm = np.max(f_norm)
        best_index = np.argmax(f_norm)
    
    x_best_norm = x_norm[best_index]
    
    # Convert best value to unnormalized space
    f_best = f_best_norm * f_std + f_mean
    x_best = x_best_norm * x_std + x_mean
    
    # Keep track of evaluated points
    evaluated_points = []
    evaluated_values = []
    

    beta_modifier = 0.8
    # Convert target to normalized space for comparison
    target_norm = (target - f_mean) / f_std
    
    for eval_iter in range(max_evals):
        # Check if we've already reached the target within tolerance
        if (f_best - target) <= tolerance:
            print(f"Target {target} reached within tolerance {tolerance} after {eval_iter} evaluations.")
            break
        
        # Adjust beta based on iterations left
        beta = min(1, max(beta * (max_evals - eval_iter - 1) / (max_evals - eval_iter), 0))
        
        # Get predictions for all candidate points
        mu, sigma = gp.predict(candidate_points, return_std=True)
        sigma_safe = np.maximum(sigma, 1e-8)  # Avoid division by zero

        # Calculate Z based on optimization direction
        Z = np.zeros_like(mu)
        mask = sigma_safe > 1e-8
        
        if minimize:
            # For minimization, we want values below f_best
            Z[mask] = (f_best_norm - mu[mask]) / sigma_safe[mask]
            # Expected improvement for minimization
            improvement = np.maximum(f_best_norm - mu, 0)
        else:
            # For maximization, we want values above f_best
            Z[mask] = (mu[mask] - f_best_norm) / sigma_safe[mask]
            # Expected improvement for maximization
            improvement = np.maximum(mu - f_best_norm, 0)
        
        # Calculate expected improvement
        EI = np.zeros_like(mu)
        EI[mask] = improvement[mask] * norm.cdf(Z[mask]) + sigma_safe[mask] * norm.pdf(Z[mask])

        # Normalize metrics to [0,1] range for combining
        EI_percentile = (EI - EI.min()) / (EI.max() - EI.min() + 1e-10) if EI.size > 1 else np.ones_like(EI)
        uncertainty_percentile = (sigma_safe - sigma_safe.min()) / (sigma_safe.max() - sigma_safe.min() + 1e-10) if sigma_safe.size > 1 else np.ones_like(sigma_safe)
        
        # Combine metrics with exploration-exploitation trade-off
        acquisition = (1 - beta) * EI_percentile + beta * uncertainty_percentile
        
        # Select best candidate point
        best_candidate_idx = np.argmax(acquisition)
        x_new_norm = candidate_points[best_candidate_idx]
        
        # Convert to unnormalized space for evaluation
        x_new = x_new_norm * x_std + x_mean
        
        # Evaluate the function at the new point
        f_new = evaluation_function(x_new)
        
        # Convert back to normalized space for GP
        f_new_norm = (f_new - f_mean) / f_std
        
        # Record evaluated points
        evaluated_points.append(x_new)
        evaluated_values.append(f_new)
        
        # Update best value based on optimization direction
        if (minimize and f_new < f_best) or (not minimize and f_new > f_best):
            f_best = f_new
            f_best_norm = f_new_norm
            x_best = x_new
            x_best_norm = x_new_norm
            best_index = len(x_norm) + eval_iter  # Adjusted index for new points
        
        # Remove selected point from candidates to avoid selecting it again
        candidate_points = np.delete(candidate_points, best_candidate_idx, axis=0)
        
        # Check if we've run out of candidate points
        if candidate_points.shape[0] == 0:
            print("No more candidate points available.")
            break

        # Update GP with the new point
        x_norm = np.vstack((x_norm, x_new_norm))
        f_norm = np.append(f_norm, f_new_norm)
        gp = fit_high_fidelity_model(x_norm, f_norm, kernel_type=kernel)

        # update beta if close to target
        if (f_best - target) <= 5*tolerance:
            beta *= beta_modifier
    
    return f_best, x_best, evaluated_points, evaluated_values


def online_optimization(num_params, variable_bounds, evaluation_function, tolerance=1, target=0, random_points = 10000, max_evaluations=100, kernel_type='RBF', obj='min'):
    #num_initial_points = int(np.ceil(2+num_params**(4/3)))
    num_initial_points = int(np.ceil(2 + num_params*(4/3)))
    print(f"Number of initial points: {num_initial_points}")
    if num_initial_points + 1 > max_evaluations:
        raise ValueError("Not enough evaluations for initial points.")
    # Generate initial points
    x_random = grp.generateRandom(variable_bounds[0], x_maxs=variable_bounds[1], num_points=random_points)
    # Normalize the data
    x_mean = np.mean(x_random, axis=0)
    x_std = np.std(x_random, axis=0)
    x_random = (x_random - x_mean) / x_std
    # Select num_initial_points that are furthest apart
    selected_indices = grp.get_furthest_points(x_random, num_initial_points)
    x_initial = x_random[selected_indices]
    # Evaluate initial unnormalized points
    x_initial_unnorm = x_initial * x_std + x_mean
    f_initial_unnorm = np.array([evaluation_function(x) for x in x_initial_unnorm])
    # Normalize the function values
    f_mean = np.mean(f_initial_unnorm)
    f_std = np.std(f_initial_unnorm)
    f_initial = (f_initial_unnorm - f_mean) / f_std
    # Fit GP model on initial points
    gp = fit_high_fidelity_model(x_initial, f_initial, kernel_type=kernel_type)
    # Prepare candidate points for optimization
    candidate_points = x_random.copy()
    # Remove initial points from candidate points
    candidate_points = np.delete(candidate_points, selected_indices, axis=0)
    # Perform dynamic optimization
    f_best, x_best, evaluated_points, evaluated_values = dynamic_online_optimization(
        x_initial, f_initial, x_mean, x_std, f_mean, f_std,
        evaluation_function, gp, candidate_points, target, tolerance,
        max_evals=max_evaluations - num_initial_points, kernel=kernel_type
    )
    # add the initial points to the evaluated points and values
    evaluated_points = x_initial_unnorm.tolist() + evaluated_points
    evaluated_values = f_initial_unnorm.tolist() + evaluated_values
    return f_best, x_best, evaluated_points, evaluated_values

def test_online_optimization():
    """
    Test the online_optimization function with the 2-norm of x as the evaluation function.
    The 2-norm has a minimum at the origin (0, 0, ..., 0).
    """
    # Define the evaluation function as the 2-norm of x
    def evaluation_function(x):
        """Return the 2-norm (Euclidean norm) of x."""
        return np.linalg.norm(x, 2)
    
    # Set up test parameters
    num_params = 14 
    variable_bounds = ( np.ones(num_params) * -5, np.ones(num_params) * 4)  # Bounds for each parameter
    tolerance = 1  # Acceptable distance from the target
    target = 0  # Target value (minimum of 2-norm is 0 at the origin)
    random_points = 50000  # Number of random candidate points
    max_evaluations = 200  # Maximum number of function evaluations
    kernel_type = 'RBF'  # Type of kernel for the Gaussian Process
    
    print("Starting online optimization test with 2-norm...")
    print(f"Dimensions: {num_params}")
    print(f"Bounds: {variable_bounds}")
    print(f"Target: {target} with tolerance: {tolerance}")
    print(f"Maximum evaluations: {max_evaluations}")
    
    # Run the optimization algorithm
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
    
    # Convert evaluated points and values to numpy arrays for easier manipulation
    evaluated_points = np.array(evaluated_points)
    evaluated_values = np.array(evaluated_values)
    
    print("\nResults:")
    print(f"Best function value found: {f_best}")
    print(f"Best point found: {x_best}")
    print(f"Distance from origin: {np.linalg.norm(x_best)}")
    print(f"Number of evaluations performed: {len(evaluated_values)}")
    
    # Check if the optimization succeeded
    success = (f_best - target) <= tolerance
    print(f"Optimization success: {success}")
    
    
    return success, f_best, x_best, evaluated_points, evaluated_values


if __name__ == "__main__":
    # Optional: Visualize the function surface
    
    # Run the test
    success, f_best, x_best, evaluated_points, evaluated_values = test_online_optimization()
    
    if success:
        print("\nTest PASSED! ✓")
        print(f"Found point {x_best} with value {f_best}")
    else:
        print("\nTest FAILED! ✗")
        print(f"Could not reach target within tolerance. Best value: {f_best}")
