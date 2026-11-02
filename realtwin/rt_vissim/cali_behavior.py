##############################################################################
# Copyright (c) 2024-, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
# This file is part of RealTwin and is distributed under a GPL license.      #
# See the LICENSE file in the top-level directory.                            #
##############################################################################

"""Vissim driving-behavior calibration against route travel times in seconds."""

from __future__ import annotations

import csv
import math
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from realtwin.rt_vissim.calibrate import calibration_termination, make_optimiser, speed_up

# Continuous COM attributes only. W99cc1Distr is a distribution reference,
# not a headway in seconds, and cannot be optimized as a floating-point value.
PARAMETER_LIMITS = {
    "W74ax": (0, None), "W74bxAdd": (None, None), "W74bxMult": (None, None),
    "W99cc0": (0, None), "W99cc2": (0, None), "W99cc3": (None, -0.001),
    "W99cc4": (None, 0), "W99cc5": (0, None), "W99cc6": (0, None),
    "W99cc7": (0, None), "W99cc8": (0, None), "W99cc9": (0, None),
}


def resolve_behavior_config(calibration: dict) -> dict:
    """Apply explicit Vissim behavior settings without changing shared settings.

    ``Calibration.behavior.vissim`` replaces the corresponding shared behavior
    values. Native parameter ranges replace the whole range mapping because
    SUMO/Aimsun and Vissim attributes do not share an optimization space.
    """
    resolved = deepcopy(calibration)
    behavior = resolved.get("behavior", {})
    if not isinstance(behavior, dict):
        raise ValueError("Calibration.behavior must be a mapping")
    native = behavior.pop("vissim", {})
    if not isinstance(native, dict):
        raise ValueError("Calibration.behavior.vissim must be a mapping")
    behavior.update(native)
    resolved["behavior"] = behavior
    return resolved


@dataclass
class BehaviorProblem:
    """Explicit native parameter bounds, applied to selected driving behaviors."""

    names: list[str]
    ranges: list[tuple[float, float]]
    driving_behavior_ids: list[int]

    @property
    def size(self) -> int:
        return len(self.names)

    def bounds(self) -> tuple[list[float], list[float]]:
        return [0.0] * self.size, [1.0] * self.size

    def physical(self, solution) -> list[float]:
        """Decode normalized optimizer values to the native Vissim units."""
        return [low + float(value) * (high - low)
                for value, (low, high) in zip(solution, self.ranges)]


def export_behavior_info(input_config: dict, net_path: Path) -> dict:
    """Validate parameter names, model IDs and travel-time targets before COM.

    Routes use the Aimsun contract: (name, start link, end link, observed
    travel time in seconds). Vissim measures from position zero on the start
    link to the end of the end link, unless behavior.route_positions supplies
    explicit marker positions in metres. Native ranges must be supplied explicitly;
    SUMO/Aimsun parameter names are not silently translated.
    """
    cfg = resolve_behavior_config(input_config["VISSIM"]["behavior"])
    behavior = cfg.get("behavior", {})
    ranges = behavior.get("params_ranges", {})
    if not isinstance(ranges, dict) or not ranges:
        raise ValueError(
            "Vissim behavior.params_ranges must map native COM names to [min, max]. "
            "Set Calibration.behavior.vissim.params_ranges or pass "
            "update_behavior_algo={'behavior': {'params_ranges': {'W74ax': [1.0, 3.0]}, "
            "'driving_behavior_ids': [1]}} for a Wiedemann 74 model.")
    bounds = []
    for name, pair in ranges.items():
        if name not in PARAMETER_LIMITS:
            raise ValueError(f"Unsupported Vissim behavior parameter {name!r}; use native "
                             f"continuous attributes: {', '.join(PARAMETER_LIMITS)}")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"behavior.params_ranges.{name} must contain [min, max]")
        low, high = map(float, pair)
        minimum, maximum = PARAMETER_LIMITS[name]
        if (not math.isfinite(low) or not math.isfinite(high) or low >= high
                or (minimum is not None and low < minimum)
                or (maximum is not None and high > maximum)):
            raise ValueError(f"Invalid Vissim bounds for {name}: {pair}")
        bounds.append((low, high))
    ids = behavior.get("driving_behavior_ids")
    if (not isinstance(ids, list) or not ids
            or any(isinstance(no, bool) or not isinstance(no, int) or no <= 0 for no in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("behavior.driving_behavior_ids must list unique positive Vissim IDs")
    root = ET.parse(net_path).getroot()
    driving = {int(item.get("no")): item for item in root.findall("drivingBehaviors/*")}
    for no in ids:
        if no not in driving:
            raise ValueError(f"Driving behavior {no} is absent from {net_path.name}")
        model = driving[no].get("carFollowModType")
        for name in ranges:
            required = "WIEDEMANN74" if name.startswith("W74") else "WIEDEMANN99"
            if model != required:
                raise ValueError(f"{name} requires {required}; driving behavior {no} uses {model}")
    routes = cfg.get("sel_behavior_routes", behavior.get("sel_behavior_routes"))
    if not isinstance(routes, (list, tuple)) or not routes:
        raise ValueError("Behavior calibration requires sel_behavior_routes with observed seconds")
    links = {int(item.get("no")) for item in root.findall("links/*")}
    targets, names = [], set()
    for route in routes:
        if not isinstance(route, (list, tuple)) or len(route) != 4:
            raise ValueError("Each behavior route must be [name, start_link, end_link, observed_seconds]")
        name, start, end, observed = route
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Behavior route names must be nonempty and unique")
        if any(isinstance(no, bool) or not isinstance(no, int) or no not in links
               for no in (start, end)):
            raise ValueError(f"Route {name!r} requires existing integer Vissim start/end link IDs")
        observed = float(observed)
        if not math.isfinite(observed) or observed <= 0:
            raise ValueError(f"Route {name!r} observed travel time must be positive seconds")
        targets.append((name, start, end, observed))
        names.add(name)
    penalty = float(behavior.get("missing_route_penalty", 1.0e4))
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("behavior.missing_route_penalty must be finite positive seconds")
    positions = behavior.get("route_positions", {})
    if not isinstance(positions, dict) or set(positions) - names:
        raise ValueError("behavior.route_positions must map configured route names to marker positions")
    for name, markers in positions.items():
        if not isinstance(markers, dict) or set(markers) - {"start", "end"}:
            raise ValueError(f"route_positions.{name} must map start/end to positions in metres")
        for position in markers.values():
            if not math.isfinite(float(position)) or float(position) < 0:
                raise ValueError(f"route_positions.{name} must contain finite nonnegative metres")
    return {"problem": BehaviorProblem(list(ranges), bounds, ids),
            "routes": targets, "missing_route_penalty": penalty, "route_positions": positions,
            "reference_source": behavior.get("reference_source", {"kind": "user_supplied"})}


class BehaviorEvaluator:
    """Reset, assign native parameters and minimize route travel-time MAE."""

    def __init__(self, session, info: dict, *, start: int, end: int,
                 seed: int = 42, resolution: int = 10, history: Path | None = None):
        self.session = session
        self.problem = info["problem"]
        self.routes = info["routes"]
        self.penalty = info["missing_route_penalty"]
        self.seed = seed
        self.count = 0
        self.best = None
        self.history = history
        sim = session.vissim.Simulation
        for name, value in (("SimPeriod", end - start), ("SimBreakAt", end - start),
                            ("StartTm", start), ("SimRes", resolution), ("RandSeed", seed)):
            sim.SetAttValue(name, value)
        speed_up(session)
        evaluation = session.vissim.Evaluation
        for name, value in (("VehTravTmsCollectData", True), ("VehTravTmsFromTime", 0),
                            ("VehTravTmsToTime", end - start), ("VehTravTmsInterval", end - start)):
            evaluation.SetAttValue(name, value)
        self.driving = [session.net.DrivingBehaviors.ItemByKey(no)
                        for no in self.problem.driving_behavior_ids]
        measurements = session.net.VehicleTravelTimeMeasurements
        existing = [int(item.AttValue("No")) for item in measurements.GetAll()]
        next_no = max(existing, default=0) + 1
        self.measures = []
        self.route_positions = {}
        for offset, (name, start_link, end_link, observed) in enumerate(self.routes):
            first = session.net.Links.ItemByKey(start_link)
            last = session.net.Links.ItemByKey(end_link)
            length = float(last.AttValue("Length2D"))
            if not math.isfinite(length) or length <= 0:
                raise ValueError(f"Route {name!r} end link must have positive length")
            markers = info["route_positions"].get(name, {})
            start_position = float(markers.get("start", 0))
            end_position = float(markers.get("end", length))
            if (start_position > float(first.AttValue("Length2D")) or end_position > length
                    or (start_link == end_link and end_position <= start_position)):
                raise ValueError(f"Route {name!r} markers must lie within their links and have positive distance")
            measure = measurements.AddVehicleTravelTimeMeasurement(
                next_no + offset, first, start_position, last, end_position)
            measure.SetAttValue("Name", name)
            distance = measure.AttValue("Dist")
            if distance is None or not math.isfinite(float(distance)) or float(distance) <= 0:
                raise ValueError(f"Route {name!r} has no continuous path between its start/end markers")
            self.measures.append(measure)
            self.route_positions[name] = {"start": start_position, "end": end_position}
        if history:
            history.parent.mkdir(parents=True, exist_ok=True)
            with history.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(["n", "seconds", "mae_seconds", "missing_routes"]
                                           + self.problem.names)

    def run(self, solution=None) -> dict:
        """Score all routes; missing completed vehicles incur a finite penalty."""
        sim = self.session.vissim.Simulation
        sim.Stop()
        sim.SetAttValue("RandSeed", self.seed)
        if solution is not None:
            values = self.problem.physical(solution)
            for driving in self.driving:
                for name, value in zip(self.problem.names, values):
                    driving.SetAttValue(name, value)
        sim.RunContinuous()
        rows, errors = [], []
        for (name, _, _, observed), measure in zip(self.routes, self.measures):
            count = measure.AttValue("Vehs(Current,Last,All)")
            travel = measure.AttValue("TravTm(Current,Last,All)")
            valid = (count is not None and float(count) > 0 and travel is not None
                     and math.isfinite(float(travel)) and float(travel) > 0)
            modelled = float(travel) if valid else None
            error = abs(modelled - observed) if valid else self.penalty + observed
            errors.append(error)
            rows.append({"route": name, "observed_seconds": observed,
                         "modelled_seconds": modelled, "vehicles": float(count or 0),
                         "absolute_error_seconds": error})
        return {"mae_seconds": sum(errors) / len(errors),
                "missing_routes": sum(row["modelled_seconds"] is None for row in rows),
                "routes": rows}

    def __call__(self, solution) -> float:
        began = time.time()
        score = self.run(solution)
        self.count += 1
        if self.best is None or score["mae_seconds"] < self.best[0]:
            self.best = (score["mae_seconds"], [float(v) for v in solution], score)
        if self.history:
            with self.history.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow([self.count, round(time.time() - began, 1),
                                            score["mae_seconds"], score["missing_routes"]]
                                           + self.problem.physical(solution))
        print(f"  :eval {self.count:>3}  travel-time MAE {score['mae_seconds']:.2f} s  "
              f"missing routes {score['missing_routes']}  best {self.best[0]:.2f} s", flush=True)
        return score["mae_seconds"]


class BehaviorCaliVissim:
    """GA/SA/TS/BO adapter with the same contract as the other simulators."""

    def __init__(self, *, input_config: dict, evaluator: BehaviorEvaluator,
                 verbose: bool = True):
        self.behavior_cfg = input_config["VISSIM"]["behavior"]
        self.evaluator = evaluator
        self.verbose = verbose
        self.after: dict | None = None

    def _run_mealpy(self, algorithm: str) -> tuple:
        from mealpy import FloatVar

        settings = self.behavior_cfg[f"{algorithm}_config"]
        lower, upper = self.evaluator.problem.bounds()
        model = make_optimiser(algorithm, settings)
        best = model.solve(
            {"obj_func": self.evaluator, "minmax": "min", "log_to": None,
             "bounds": FloatVar(lb=lower, ub=upper)},
            termination=calibration_termination(self.behavior_cfg, "behavior", settings),
            seed=self.behavior_cfg.get("behavior", {}).get("optimizer_seed", 812),
        )
        self.after = self.evaluator.run(best.solution.copy())
        return best, model

    def run_GA(self) -> tuple:
        return self._run_mealpy("ga")

    def run_SA(self) -> tuple:
        return self._run_mealpy("sa")

    def run_TS(self) -> tuple:
        return self._run_mealpy("ts")

    def run_BO(self) -> tuple:
        from realtwin.func_lib._f_calibration.algo_sumo._bayesian_opt import BayesianOptimization

        model = BayesianOptimization(
            scenario_config=self.behavior_cfg.get("scenario_config", {}),
            algo_config=self.behavior_cfg, bounds=self.evaluator.problem.bounds(),
            simulator="vissim", verbose=self.verbose,
        )
        best = model.solve(self.evaluator, objective_kwargs={})
        self.after = self.evaluator.run(best.solution.copy())
        return best, model
