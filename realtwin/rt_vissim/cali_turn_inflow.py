##############################################################################
# Copyright (c) 2024-, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
# This file is part of RealTwin and is distributed under a GPL license.      #
# See the LICENSE file in the top-level directory.                            #
##############################################################################

"""Turn/inflow calibration using the package's GA/SA/TS/BO contracts."""

from realtwin.rt_vissim.calibrate import Evaluator, calibration_termination, make_optimiser


class TurnInflowCaliVissim:
    """Optimize demand with a live evaluator and effective per-stage settings.

    Args:
        input_config: Configuration containing VISSIM.turn_inflow, including
            scenario_config, turn_inflow and algorithm configuration blocks.
        evaluator: Loaded Vissim session's normalized demand objective.
        verbose: Print BO progress. Evaluation history is always recorded.
    """

    def __init__(self, *, input_config: dict, evaluator: Evaluator,
                 verbose: bool = True):
        self.turn_inflow_cfg = input_config["VISSIM"]["turn_inflow"]
        self.evaluator = evaluator
        self.verbose = verbose
        self.after: dict | None = None

    def _run_mealpy(self, algorithm: str) -> tuple:
        """Run the configured model and apply its best demand once more."""
        from mealpy import FloatVar

        settings = self.turn_inflow_cfg[f"{algorithm}_config"]
        lower, upper = self.evaluator.problem.bounds()
        model = make_optimiser(algorithm, settings)
        best = model.solve(
            {"obj_func": self.evaluator, "minmax": "min", "log_to": None,
             "bounds": FloatVar(lb=lower, ub=upper)},
            termination=calibration_termination(self.turn_inflow_cfg, "turn_inflow", settings),
            seed=self.turn_inflow_cfg.get("turn_inflow", {}).get("optimizer_seed", 812),
        )
        self.after = self.evaluator.run(best.solution.copy())
        return best, model

    def run_GA(self) -> tuple:
        """Run genetic algorithm calibration (the default)."""
        return self._run_mealpy("ga")

    def run_SA(self) -> tuple:
        """Run simulated annealing calibration."""
        return self._run_mealpy("sa")

    def run_TS(self) -> tuple:
        """Run tabu search calibration."""
        return self._run_mealpy("ts")

    def run_BO(self) -> tuple:
        """Reuse shared BO on normalized bounds and apply the best candidate."""
        from realtwin.func_lib._f_calibration.algo_sumo._bayesian_opt import BayesianOptimization

        model = BayesianOptimization(
            scenario_config=self.turn_inflow_cfg.get("scenario_config", {}),
            algo_config=self.turn_inflow_cfg,
            bounds=self.evaluator.problem.bounds(),
            simulator="vissim", verbose=self.verbose,
        )
        best = model.solve(self.evaluator, objective_kwargs={})
        self.after = self.evaluator.run(best.solution.copy())
        return best, model
