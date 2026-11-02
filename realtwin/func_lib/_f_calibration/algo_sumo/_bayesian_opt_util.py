##############################################################################
# Copyright (c) 2024-, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################

"""Gaussian-process search adapted from the GitLab Real-Twin BO workflow."""

from collections.abc import Callable

import numpy as np


def online_optimization(
    num_params: int,
    variable_bounds: tuple,
    evaluation_function: Callable,
    tolerance: float = 3,
    target: float = 0,
    random_points: int = 4000,
    max_evaluations: int = 100,
    kernel_type: str = "RBF",
    *,
    seed: int = 812,
    verbose: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Minimize a scalar objective using a finite pool of bounded candidates.

    Args:
        num_params: Number of decision variables, including fixed variables.
        variable_bounds: Lower and upper bounds in the objective's units.
        evaluation_function: Callable accepting a parameter vector and returning
            a finite scalar or a single-element sequence.
        tolerance: Stop when the best fitness is at most target + tolerance.
        target: Desired minimum in the objective's units.
        random_points: Candidate pool size; must cover max_evaluations.
        max_evaluations: Per-run objective-call limit, including initialization.
        kernel_type: RBF, Matern, RationalQuadratic, ExpSineSquared, or Combined.
        seed: Nonnegative seed for candidate generation and model fitting.
        verbose: Print each evaluation and the best observed fitness.

    Returns:
        Best fitness, best parameters, all evaluated parameters, and fitnesses.

    Raises:
        ImportError: If the optional BO dependencies are unavailable.
        ValueError: If bounds, settings, or objective values are invalid.
    """
    for name, value in (
        ("num_params", num_params),
        ("random_points", random_points),
        ("max_evaluations", max_evaluations),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if random_points < max_evaluations:
        raise ValueError("random_points must be at least max_evaluations.")
    if not callable(evaluation_function):
        raise ValueError("evaluation_function must be callable.")
    target, tolerance = float(target), float(tolerance)
    if not np.isfinite(target) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "target must be finite and tolerance must be finite and nonnegative."
        )

    bounds = np.asarray(variable_bounds, dtype=float)
    if bounds.shape != (2, num_params) or not np.isfinite(bounds).all():
        raise ValueError(
            "variable_bounds must contain finite lower and upper bounds for every parameter."
        )
    lower, upper = bounds
    if np.any(lower > upper):
        raise ValueError(
            "Each lower bound must be less than or equal to its upper bound."
        )

    # Load only when BO is selected, so existing calibration needs no new imports.
    try:
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF,
            ConstantKernel,
            Matern,
            RationalQuadratic,
            WhiteKernel,
        )
    except ImportError as exc:
        raise ImportError(
            "Bayesian optimization requires the BO extra. Install it in your "
            'Python environment with: python -m pip install "realtwin[bo]" '
            '(from a checkout: python -m pip install -e ".[bo]").'
        ) from exc

    kernels = {
        "RBF": RBF(length_scale_bounds=(1e-5, 1e6)),
        "Matern": Matern(nu=2.5, length_scale_bounds=(1e-5, 1e6)),
        "RationalQuadratic": RationalQuadratic(length_scale_bounds=(1e-5, 1e6)),
        "ExpSineSquared": RBF(length_scale_bounds=(1e-5, 1e6)),
        "Combined": RBF(length_scale_bounds=(1e-5, 1e6)) + WhiteKernel(),
    }
    if kernel_type not in kernels:
        raise ValueError(
            f"Unsupported kernel_type {kernel_type!r}; choose from {list(kernels)}."
        )

    rng = np.random.default_rng(seed)
    active = upper > lower
    active_count = int(active.sum())
    # Unit bounds keep inflow counts and turn ratios on comparable scales.
    candidates = rng.random((random_points if active_count else 1, active_count))
    model_points = candidates
    if kernel_type == "ExpSineSquared":
        # An RBF on per-variable sine/cosine coordinates is a product periodic
        # kernel with period one in unit bounds. Unlike applying sin to a
        # multidimensional Euclidean distance, its covariance is positive definite.
        angles = 2 * np.pi * candidates
        model_points = np.concatenate((np.sin(angles), np.cos(angles)), axis=1)
    budget = min(max_evaluations, len(candidates))
    initial_count = min(int(np.ceil(2 + active_count * 4 / 3)), budget)

    # Greedy space-filling initialization uses linear memory, avoiding the
    # reference implementation's candidate-by-candidate distance matrix.
    initial_indices = []
    nearest_distance = np.full(len(candidates), np.inf)
    next_index = int(rng.integers(len(candidates)))
    for _ in range(initial_count):
        initial_indices.append(next_index)
        distance = np.sum((candidates - candidates[next_index]) ** 2, axis=1)
        nearest_distance = np.minimum(nearest_distance, distance)
        nearest_distance[initial_indices] = -np.inf
        next_index = int(np.argmax(nearest_distance))

    available = np.ones(len(candidates), dtype=bool)
    evaluated_indices = []
    evaluated_points = []
    evaluated_values = []
    for iteration in range(budget):
        if iteration < initial_count:
            next_index = initial_indices[iteration]
        else:
            model = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * kernels[kernel_type],
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=int(seed % (2**32)),
            )
            model.fit(model_points[evaluated_indices], np.asarray(evaluated_values))
            remaining_indices = np.flatnonzero(available)
            mean, std = model.predict(model_points[remaining_indices], return_std=True)
            improvement = min(evaluated_values) - mean
            expected_improvement = np.maximum(improvement, 0)
            uncertain = std > 1e-12
            z_score = improvement[uncertain] / std[uncertain]
            expected_improvement[uncertain] = improvement[uncertain] * norm.cdf(
                z_score
            ) + std[uncertain] * norm.pdf(z_score)
            expected_improvement = np.maximum(expected_improvement, 0)
            ei_scale = np.ptp(expected_improvement)
            std_scale = np.ptp(std)
            normalized_ei = (expected_improvement - expected_improvement.min()) / max(
                ei_scale, 1e-12
            )
            normalized_std = (std - std.min()) / max(std_scale, 1e-12)
            # Gradually shift from uncertainty toward expected improvement.
            exploration = (budget - iteration - 1) / (budget - initial_count)
            acquisition = (
                1 - exploration
            ) * normalized_ei + exploration * normalized_std
            next_index = int(remaining_indices[np.argmax(acquisition)])

        point = lower.copy()
        point[active] += candidates[next_index] * (upper[active] - lower[active])
        # Simulation objectives may modify their arguments; retain the proposed
        # parameters so the model and exported history describe the same input.
        value = np.asarray(evaluation_function(point.copy()), dtype=float)
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError(
                f"BO objective must return one finite fitness value; received {value!r} "
                f"at evaluation {iteration + 1}. Check the simulation output."
            )
        fitness = float(value.reshape(-1)[0])
        evaluated_indices.append(next_index)
        evaluated_points.append(point)
        evaluated_values.append(fitness)
        available[next_index] = False
        best_fitness = min(evaluated_values)
        if verbose:
            print(
                f"  :BO evaluation {iteration + 1}/{budget}: fitness={fitness:.6g}, best={best_fitness:.6g}",
                flush=True,
            )
        if best_fitness <= target + tolerance:
            break

    best_index = int(np.argmin(evaluated_values))
    return (
        evaluated_values[best_index],
        evaluated_points[best_index].copy(),
        np.asarray(evaluated_points),
        np.asarray(evaluated_values),
    )
