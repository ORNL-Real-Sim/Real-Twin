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

"""Shared Bayesian optimization and simulator-specific calibration exports."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from mealpy.utils.agent import Agent
from mealpy.utils.target import Target
import numpy as np
import pandas as pd

from ._bayesian_opt_util import online_optimization


class BayesianOptimization:
    """Run the GitLab Real-Twin BO workflow with reproducible independent runs.

    Args:
        scenario_config: Configuration passed to the simulation objective.
        algo_config: Calibration settings containing an optional bo_config.
        verbose: Print run progress.
        bounds: Explicit lower/upper bounds from the calibration adapter. When
            omitted, derive them from scenario variable counts or behavior ranges.
        simulator: Simulator name used in CSV filenames; defaults to sumo.
    """

    def __init__(
        self,
        scenario_config: dict,
        algo_config: dict | None = None,
        verbose: bool = False,
        *,
        bounds: tuple | None = None,
        simulator: str = "sumo",
    ):
        self.scenario_config = scenario_config
        self.algo_config = algo_config or {}
        self.verbose = verbose
        if simulator not in {"sumo", "aimsun"}:
            raise ValueError("simulator must be 'sumo' or 'aimsun'.")
        self.simulator = simulator
        if bounds is None:
            if "behavior_parameters_ranges" in scenario_config:
                ranges = scenario_config["behavior_parameters_ranges"]
                if isinstance(ranges, dict):
                    ranges = ranges.values()
                ranges = list(ranges)
                bounds = ([pair[0] for pair in ranges], [pair[1] for pair in ranges])
            else:
                n_turn = scenario_config["N_TurnVariable"]
                n_inflow = scenario_config["N_InflowVariable"]
                bounds = (
                    [0] * scenario_config["N_Variable"],
                    [1] * n_turn + [scenario_config.get("max_inflow", 200)] * n_inflow,
                )
        self.bounds = bounds
        self.n_variable = len(bounds[0])

    def solve(
        self, obj_func: Callable, *, objective_kwargs: dict | None = None
    ) -> Agent:
        """Minimize obj_func(parameters, scenario_config=...) across BO runs.

        The returned agent provides solution and target.fitness, matching the
        existing calibration algorithms. Simulation adapters apply this solution
        once more after the search to restore generated files. objective_kwargs
        can supply a different objective contract, such as Aimsun's input_config;
        by default, the objective receives scenario_config.
        """
        settings = self.algo_config.get("bo_config", {})
        if not isinstance(settings, dict):
            raise ValueError("bo_config must be a dictionary.")
        self.total_run = settings.get("total_run", 1)
        if (
            isinstance(self.total_run, bool)
            or not isinstance(self.total_run, int)
            or not 1 <= self.total_run <= 1000
        ):
            raise ValueError("total_run must be an integer between 1 and 1000.")
        seed = settings.get("seed", 812)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if not callable(obj_func):
            raise ValueError("obj_func must be callable.")

        if objective_kwargs is None:
            objective_kwargs = {"scenario_config": self.scenario_config}
        objective = partial(obj_func, **objective_kwargs)

        all_points = []
        best_rows = []
        self.temp_evaluated_values = []
        for run in range(1, self.total_run + 1):
            if self.verbose:
                print(
                    f"  :Starting Bayesian optimization run {run}/{self.total_run}",
                    flush=True,
                )
            start_time = perf_counter()
            _, _, points, values = online_optimization(
                num_params=self.n_variable,
                variable_bounds=self.bounds,
                evaluation_function=objective,
                tolerance=settings.get("tolerance", 3),
                target=settings.get("target", 0),
                random_points=settings.get("random_points", 4000),
                max_evaluations=settings.get("max_evaluations", 100),
                kernel_type=settings.get("kernel_type", "RBF"),
                seed=seed + run - 1,
                verbose=self.verbose,
            )
            elapsed = perf_counter() - start_time
            self.temp_evaluated_values.append(values)
            run_rows = [
                {
                    "run": run,
                    "iteration": iteration,
                    **{
                        f"param_{index + 1}": value for index, value in enumerate(point)
                    },
                    "fitness": fitness,
                    "run_time_sec": elapsed if iteration == len(values) else np.nan,
                }
                for iteration, (point, fitness) in enumerate(
                    zip(points, values), start=1
                )
            ]
            all_points.extend(run_rows)
            best_row = run_rows[int(np.argmin(values))].copy()
            best_row["run_time_sec"] = elapsed
            best_rows.append(best_row)

        self.points_df = pd.DataFrame(all_points)
        self.results_df = self.points_df[
            ["run", "iteration", "fitness", "run_time_sec"]
        ].copy()
        self.best_df = pd.DataFrame(best_rows)
        best = self.best_df.loc[self.best_df["fitness"].idxmin()]
        self.best_fitness = float(best["fitness"])
        self.best_solution = np.array(
            [best[f"param_{index + 1}"] for index in range(self.n_variable)]
        )
        self.best_run_fitness = self.temp_evaluated_values[int(best["run"]) - 1].copy()
        self.g_best = Agent(
            solution=self.best_solution.copy(), target=Target(self.best_fitness)
        )
        return self.g_best

    def run_vis(self, output_dir: str = "./") -> bool:
        """Save per-evaluation CSVs, per-run best parameters, and convergence plots."""
        if not hasattr(self, "g_best"):
            raise RuntimeError("Call solve() before exporting BO results.")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.simulator}_bayesopt_{self.total_run}runs"
        self.results_df.to_csv(output_path / f"{prefix}_results.csv", index=False)
        self.points_df.to_csv(output_path / f"{prefix}_points.csv", index=False)
        self.best_df.to_csv(output_path / f"{prefix}_best_per_run.csv", index=False)

        plots = [
            (
                values,
                f"Fitness convergence for run {run}",
                f"fitness_vs_iteration_run{run}.png",
            )
            for run, values in enumerate(self.temp_evaluated_values, start=1)
        ]
        plots.append(
            (
                self.best_run_fitness,
                "Fitness convergence for best run",
                "best_run_fitness_convergence.png",
            )
        )
        for values, title, filename in plots:
            figure, axes = plt.subplots(figsize=(8, 5))
            try:
                iterations = np.arange(1, len(values) + 1)
                axes.plot(iterations, values, marker="o", label="Evaluated fitness")
                axes.plot(
                    iterations, np.minimum.accumulate(values), label="Best so far"
                )
                axes.set(xlabel="Evaluation", ylabel="Fitness", title=title)
                axes.grid(True)
                axes.legend()
                figure.tight_layout()
                figure.savefig(output_path / filename)
            finally:
                plt.close(figure)
        return True
