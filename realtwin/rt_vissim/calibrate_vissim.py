##############################################################################
# Copyright (c) 2024-, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
# This file is part of RealTwin and is distributed under a GPL license.      #
# See the LICENSE file in the top-level directory.                            #
##############################################################################

"""Dispatch separate Vissim calibration adapters through the shared workflow."""

import math

from realtwin.func_lib._f_calibration._calibration_workflow import run_calibration_algorithm
from realtwin.rt_vissim.cali_turn_inflow import TurnInflowCaliVissim
from realtwin.rt_vissim.cali_behavior import BehaviorCaliVissim


def validate_bo_settings(settings: dict) -> None:
    """Check BO settings before starting a licensed Vissim session."""
    for name, maximum in (("max_evaluations", None), ("random_points", None),
                          ("total_run", 1000), ("seed", None)):
        value = settings[name]
        minimum = 0 if name == "seed" else 1
        if (isinstance(value, bool) or not isinstance(value, int)
                or value < minimum or (maximum is not None and value > maximum)):
            raise ValueError(f"bo_config.{name} must be an integer >= {minimum}"
                             + (f" and <= {maximum}" if maximum else ""))
    if settings["random_points"] < settings["max_evaluations"]:
        raise ValueError("bo_config.random_points must be at least max_evaluations")
    target, tolerance = float(settings["target"]), float(settings["tolerance"])
    if not math.isfinite(target) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("BO target must be finite and tolerance finite and nonnegative")
    if settings["kernel_type"] not in {
        "RBF", "Matern", "RationalQuadratic", "ExpSineSquared", "Combined"
    }:
        raise ValueError(f"Unsupported BO kernel_type {settings['kernel_type']!r}")


def cali_vissim(*, input_config: dict, evaluator,
                sel_algo: dict | None = None, verbose: bool = True,
                stage: str = "turn_inflow") -> tuple:
    """Dispatch Vissim calibration through the shared package workflow.

    Args:
        input_config: Effective VISSIM.turn_inflow or VISSIM.behavior configuration.
        evaluator: Objective bound to the loaded Vissim model and observations.
        sel_algo: Algorithm names per stage; GA is the default. The public
            RealTwinVISSIM.calibrate method validates stage flags and names.
        verbose: Print BO progress.
        stage: Select the independent turn_inflow or behavior adapter. The
            public method runs enabled stages in that order using saved models.

    Returns:
        Optimizer model, effective algorithm name and final best-candidate score.
        The caller saves the calibrated network and exports its results.
    """
    requested = sel_algo if isinstance(sel_algo, dict) else {}
    if stage not in {"turn_inflow", "behavior"}:
        raise ValueError(f"Unknown Vissim calibration stage: {stage!r}")
    algorithm = requested.get(stage, "ga")
    if isinstance(algorithm, str):
        algorithm = algorithm.lower()
    adapter = TurnInflowCaliVissim if stage == "turn_inflow" else BehaviorCaliVissim
    calibrator = adapter(
        input_config=input_config, evaluator=evaluator, verbose=verbose)
    model, algorithm = run_calibration_algorithm(calibrator, algorithm)
    return model, algorithm, calibrator.after
