"""Characterize calibration control flow before consolidating shared policy."""

from copy import deepcopy
import importlib
from pathlib import Path

import pytest

from realtwin import RealTwinAimsun, RealTwinSUMO


@pytest.fixture(params=["SUMO", "AIMSUN"])
def dispatch_case(request, monkeypatch, tmp_path):
    simulator = request.param
    module_name = (
        "realtwin.func_lib._f_calibration.calibration_sumo"
        if simulator == "SUMO"
        else "realtwin.rt_aimsun.calibrate_aimsun"
    )
    module = importlib.import_module(module_name)
    calls = []
    scenario = {"sel_behavior_routes": {"route": {"time": 30}}}
    config = {
        "Calibration": {"scenario_config": scenario},
        simulator: {
            "model_fname": str(tmp_path / "network.ang"),
            "turn_inflow": {
                "turn_inflow": {"is_calibration": True},
                "scenario_config": scenario,
            },
            "behavior": {
                "behavior": {"is_calibration": True},
                "sel_behavior_routes": scenario["sel_behavior_routes"],
            },
        },
    }

    class Calibration:
        def __init__(self, stage):
            self.stage = stage
            self.model = object()
            calls.append((stage, "init"))

        def optimize(self, algorithm):
            calls.append((self.stage, algorithm))
            if config.get("fail_at") == "solve":
                raise RuntimeError("simulation failed")
            return object(), self.model

        def run_GA(self):
            return self.optimize("ga")

        def run_SA(self):
            return self.optimize("sa")

        def run_TS(self):
            return self.optimize("ts")

        def run_BO(self):
            return self.optimize("bo")

        def run_vis(self, output_dir, model):
            assert model is self.model
            calls.append((self.stage, "export", output_dir))
            if config.get("fail_at") == "export":
                raise RuntimeError("export failed")

        def _clean_up(self):
            calls.append((self.stage, "cleanup"))

    turn_class = "TurnInflowCali" if simulator == "SUMO" else "TurnInflowCaliAimsun"
    behavior_class = "BehaviorCali" if simulator == "SUMO" else "BehaviorCaliAimsun"
    monkeypatch.setattr(
        module, turn_class, lambda *args, **kwargs: Calibration("turn_inflow")
    )
    monkeypatch.setattr(
        module, behavior_class, lambda *args, **kwargs: Calibration("behavior")
    )
    if simulator == "SUMO":
        monkeypatch.setattr(
            module, "prepare_scenario_config_turn_inflow", lambda config: scenario
        )
        monkeypatch.setattr(
            module, "prepare_scenario_config_behavior", lambda config: scenario
        )
    else:

        def export_metadata(input_config, **kwargs):
            calls.append(("metadata", kwargs))
            return {"calibration_info_path": "calibration_info.json"}

        monkeypatch.setattr(module, "export_turn_inflow_info", export_metadata)
    return simulator, getattr(module, f"cali_{simulator.lower()}"), config, calls


@pytest.mark.parametrize("requested", ["ga", "sa", "ts", "bo", "invalid", "BO", []])
def test_dispatch_preserves_algorithm_output_paths_and_stage_order(
    dispatch_case, requested
):
    simulator, dispatch, config, calls = dispatch_case
    assert dispatch(
        sel_algo={"turn_inflow": requested, "behavior": requested},
        input_config=config,
        verbose=False,
        export_option="preserved",
    )
    algorithm = requested if requested in ("ga", "sa", "ts", "bo") else "ga"
    expected = []
    if simulator == "AIMSUN":
        expected.append(("metadata", {"export_option": "preserved"}))
    for stage in ("turn_inflow", "behavior"):
        output_dir = f"{stage}_{algorithm}_result"
        if simulator == "AIMSUN" and algorithm == "bo":
            output_dir = str(Path(config[simulator]["model_fname"]).parent / output_dir)
        expected.extend(
            [(stage, "init"), (stage, algorithm), (stage, "export", output_dir)]
        )
        if stage == "turn_inflow" or simulator == "AIMSUN":
            expected.append((stage, "cleanup"))
    assert calls == expected


@pytest.mark.parametrize("failure", ["solve", "export"])
def test_dispatch_failure_stops_later_work(dispatch_case, failure):
    _, dispatch, config, calls = dispatch_case
    config["fail_at"] = failure
    with pytest.raises(RuntimeError, match="failed"):
        dispatch(sel_algo={"turn_inflow": "bo", "behavior": "bo"}, input_config=config)
    assert not any(call[0] == "behavior" or call[1] == "cleanup" for call in calls)


@pytest.mark.parametrize(
    "twin_class,module_name,simulator",
    [
        (RealTwinSUMO, "realtwin.rt_sumo.rt_sumo", "SUMO"),
        (RealTwinAimsun, "realtwin.rt_aimsun.rt_aimsun", "AIMSUN"),
    ],
)
@pytest.mark.parametrize("requested", [None, "not a dictionary", {}])
def test_public_defaults_overrides_and_route_precedence(
    twin_class, module_name, simulator, requested, monkeypatch
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, f"cali_{simulator.lower()}", lambda **kwargs: True)
    twin = twin_class.__new__(twin_class)
    twin.verbose = False
    base = {
        "turn_inflow": {"is_calibration": True},
        "behavior": {"is_calibration": True},
        "bo_config": {"seed": 42, "max_evaluations": 10},
    }
    original = deepcopy(base)
    twin.input_config = {
        "Calibration": base,
        "demo_data": False,
        simulator: {"exe_path": "simulator", "behavior": {"stale": True}},
    }
    routes = {"argument_route": {"time": 30}}
    route_override = {"argument_route": {"time": 25}, "override_route": {"time": 20}}
    assert twin.calibrate(
        sel_algo=requested,
        sel_behavior_routes=routes,
        update_turn_inflow_algo={"bo_config": {"max_evaluations": 2}},
        update_behavior_algo={
            "bo_config": None,
            "sel_behavior_routes": route_override,
        },
    )
    settings = twin.input_config[simulator]
    assert settings["exe_path"] == "simulator"
    assert "stale" not in settings["behavior"]
    assert settings["turn_inflow"]["bo_config"] == {"seed": 42, "max_evaluations": 2}
    assert settings["behavior"]["bo_config"] is None
    expected_routes = {**routes, **route_override} if simulator == "SUMO" else routes
    assert settings["behavior"]["sel_behavior_routes"] == expected_routes
    assert base == original
    # Unmodified nested values keep the existing shallow-copy contract.
    assert settings["turn_inflow"]["behavior"] is base["behavior"]
