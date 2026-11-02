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

"""Shared calibration policy and optimizer dispatch for SUMO and Aimsun."""

from rich.console import Console


def select_calibration_algorithms(
    calibration: dict, sel_algo: dict | None, *, console: Console
) -> dict[str, str] | None:
    """Validate enabled stages and return their normalized algorithm names.

    Disabled stages retain the GA placeholder expected by the dispatchers.
    Return None, with the existing console diagnostic, when calibration is
    disabled or an enabled stage has an unsupported algorithm.
    """
    enabled_stages = {
        stage: calibration.get(stage, {}).get("is_calibration", False)
        for stage in ("turn_inflow", "behavior")
    }
    if not any(enabled_stages.values()):
        console.print(
            "  [dim cyan]:Calibration is skipped in the input configuration file."
        )
        return None

    if sel_algo is not None and not isinstance(sel_algo, dict):
        console.print(
            "  [bold red]:sel_algo must be a dictionary; using GA for enabled stages."
        )
    requested_algorithms = sel_algo if isinstance(sel_algo, dict) else {}
    algorithms = {}
    for stage, enabled in enabled_stages.items():
        algorithm = requested_algorithms.get(stage, "ga") if enabled else "ga"
        if not isinstance(algorithm, str) or algorithm.lower() not in {
            "ga",
            "sa",
            "ts",
            "bo",
        }:
            console.print(
                f"  [bold red]:Unsupported {stage} algorithm {algorithm!r}; use GA, SA, TS, or BO."
            )
            return None
        algorithms[stage] = algorithm.lower()
    return algorithms


def apply_calibration_overrides(
    stage_configs: dict, *, turn_inflow: dict | None, behavior: dict | None
) -> None:
    """Apply per-stage settings without modifying shared nested dictionaries.

    Merge dictionary values one level deep; replace other values directly.
    Stage initialization and route precedence remain with each simulator.
    """
    for stage, updates in (("turn_inflow", turn_inflow), ("behavior", behavior)):
        stage_config = stage_configs[stage]
        for key, value in (updates or {}).items():
            if isinstance(value, dict) and isinstance(stage_config.get(key), dict):
                stage_config[key] = {**stage_config[key], **value}
            else:
                stage_config[key] = value


def run_calibration_algorithm(calibrator, algorithm: str) -> tuple:
    """Run one adapter's optimizer and return its model and effective algorithm.

    Adapters expose run_GA, run_SA, run_TS, and run_BO; each applies its best
    candidate before returning. Direct dispatcher calls retain their existing
    case-sensitive GA fallback. Callers own result paths, exports, and cleanup.
    """
    match algorithm:
        case "ga":
            _, model = calibrator.run_GA()
        case "sa":
            _, model = calibrator.run_SA()
        case "ts":
            _, model = calibrator.run_TS()
        case "bo":
            _, model = calibrator.run_BO()
        case _:
            print(
                f"  :Error: unsupported algorithm {algorithm}, using genetic algorithm as default."
            )
            _, model = calibrator.run_GA()
            algorithm = "ga"
    return model, algorithm
