"""Aimsun BO contracts and independent Aimsun/SUMO calibration stages."""

from copy import deepcopy
import importlib
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from realtwin import RealTwinAimsun, RealTwinSUMO
from realtwin.func_lib._f_calibration.calibration_sumo import (
    prepare_scenario_config_behavior,
)
from realtwin.rt_aimsun import cali_behavior, cali_turn_inflow
from realtwin.rt_aimsun.calibrate_aimsun import cali_aimsun

PUBLIC_APIS = [
    (RealTwinSUMO, "realtwin.rt_sumo.rt_sumo", "SUMO", "cali_sumo"),
    (RealTwinAimsun, "realtwin.rt_aimsun.rt_aimsun", "AIMSUN", "cali_aimsun"),
]


def make_twin(twin_class, simulator, turn_enabled, behavior_enabled):
    twin = twin_class.__new__(twin_class)
    twin.verbose = False
    twin.input_config = {
        "demo_data": False,
        simulator: {},
        "Calibration": {
            "scenario_config": {},
            "turn_inflow": {"is_calibration": turn_enabled, "max_inflow": 120},
            "behavior": {"is_calibration": behavior_enabled},
            "bo_config": {
                "seed": 24,
                "random_points": 32,
                "max_evaluations": 10,
                "target": -1,
                "tolerance": 0,
            },
        },
    }
    return twin


def routes_for(simulator):
    if simulator == "AIMSUN":
        return [("test_route", 10, 20, 30.0)]
    return {"test_route": {"route_list": ["edge1"], "time": 30.0}}


@pytest.mark.parametrize("twin_class,module_name,simulator,dispatcher", PUBLIC_APIS)
@pytest.mark.parametrize(
    "enabled", [(True, False), (False, True), (True, True), (False, False)]
)
@pytest.mark.parametrize("include_disabled", [False, True])
def test_public_stage_flags_and_partial_selectors(
    twin_class,
    module_name,
    simulator,
    dispatcher,
    enabled,
    include_disabled,
    monkeypatch,
):
    calls = []

    def calibration(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(importlib.import_module(module_name), dispatcher, calibration)
    twin = make_twin(twin_class, simulator, *enabled)
    original = deepcopy(twin.input_config["Calibration"])
    algorithms = {
        stage: "BO"
        for stage, is_enabled in zip(("turn_inflow", "behavior"), enabled)
        if is_enabled
    }
    if include_disabled:
        algorithms.update(
            {
                stage: None
                for stage, is_enabled in zip(("turn_inflow", "behavior"), enabled)
                if not is_enabled
            }
        )
    completed = twin.calibrate(
        sel_algo=algorithms,
        sel_behavior_routes=routes_for(simulator),
        update_turn_inflow_algo={"bo_config": {"max_evaluations": 2}},
        update_behavior_algo={"bo_config": {"max_evaluations": 3}},
    )
    assert completed is any(enabled)
    assert len(calls) == int(any(enabled))
    assert twin.input_config["Calibration"] == original
    if calls:
        for stage, is_enabled, budget in zip(
            ("turn_inflow", "behavior"), enabled, (2, 3)
        ):
            assert calls[0]["sel_algo"][stage] == ("bo" if is_enabled else "ga")
            settings = calls[0]["input_config"][simulator][stage]["bo_config"]
            assert settings["max_evaluations"] == budget
            assert settings["seed"] == 24


@pytest.mark.parametrize("twin_class,module_name,simulator,dispatcher", PUBLIC_APIS)
@pytest.mark.parametrize("active_stage", ["turn_inflow", "behavior"])
def test_invalid_enabled_algorithm_fails_before_dispatch(
    twin_class,
    module_name,
    simulator,
    dispatcher,
    active_stage,
    monkeypatch,
):
    monkeypatch.setattr(
        importlib.import_module(module_name),
        dispatcher,
        lambda **kwargs: pytest.fail("Invalid algorithm must not execute calibration."),
    )
    twin = make_twin(
        twin_class, simulator, active_stage == "turn_inflow", active_stage == "behavior"
    )
    assert (
        twin.calibrate(
            sel_algo={active_stage: "invalid"},
            sel_behavior_routes=routes_for(simulator),
        )
        is False
    )


@pytest.mark.parametrize("twin_class,module_name,simulator,dispatcher", PUBLIC_APIS)
def test_calibration_failure_is_propagated(
    twin_class, module_name, simulator, dispatcher, monkeypatch
):
    monkeypatch.setattr(
        importlib.import_module(module_name), dispatcher, lambda **kwargs: False
    )
    twin = make_twin(twin_class, simulator, True, False)
    assert twin.calibrate(sel_algo={"turn_inflow": "BO"}) is False


def test_aimsun_missing_routes_does_not_disable_future_behavior(monkeypatch):
    calls = []

    def calibration(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(
        importlib.import_module("realtwin.rt_aimsun.rt_aimsun"),
        "cali_aimsun",
        calibration,
    )
    twin = make_twin(RealTwinAimsun, "AIMSUN", False, True)
    original = deepcopy(twin.input_config["Calibration"])
    assert twin.calibrate(sel_algo={"behavior": "BO"}) is False
    assert twin.input_config["Calibration"] == original
    assert not calls
    assert (
        twin.calibrate(
            sel_algo={"behavior": "BO"}, sel_behavior_routes=routes_for("AIMSUN")
        )
        is True
    )
    assert len(calls) == 1
    assert twin.input_config["Calibration"] == original


def test_aimsun_can_read_behavior_routes_from_configuration(monkeypatch):
    calls = []

    def calibration(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(
        importlib.import_module("realtwin.rt_aimsun.rt_aimsun"),
        "cali_aimsun",
        calibration,
    )
    twin = make_twin(RealTwinAimsun, "AIMSUN", False, True)
    twin.input_config["Calibration"]["behavior"]["sel_behavior_routes"] = routes_for(
        "AIMSUN"
    )
    assert twin.calibrate(sel_algo={"behavior": "BO"}) is True
    assert calls[0]["input_config"]["AIMSUN"]["behavior"][
        "sel_behavior_routes"
    ] == routes_for("AIMSUN")


def configure_aimsun_adapters(tmp_path):
    model_dir = tmp_path / "Model"
    model_dir.mkdir()
    model_file = model_dir / "network.ang"
    model_file.touch()
    config = make_twin(RealTwinAimsun, "AIMSUN", True, True).input_config
    config["AIMSUN"] = {
        "model_fname": str(model_file),
        "exe_path": "aconsole",
        "aimsun_file": {"step7.2": "assign.py"},
        "turn_inflow": {
            **deepcopy(config["Calibration"]),
            "num_variables": 3,
            "n_turn_vars": 2,
            "n_inflow_vars": 1,
            "approach_groups": [{"from": 10, "tos": [20, 30]}],
            "inflow_sections": [40],
        },
        "behavior": deepcopy(config["Calibration"]),
    }
    config["AIMSUN"]["behavior"]["behavior"]["params_ranges"] = {
        # Deliberately reversed input order: the adapter must preserve its canonical order.
        "MaxDec": [7, 9],
        "MinHeadway": [0.5, 1],
        "SensitivityFactor": [0.2, 0.8],
        "NormalDec": [4, 5],
        "MaxAcc": [2, 3],
        "MinDist": [1, 2],
    }
    for stage in ("turn_inflow", "behavior"):
        config["AIMSUN"][stage]["bo_config"].update(
            {"max_evaluations": 3, "total_run": 2}
        )
    return config


@pytest.mark.parametrize(
    "turn_enabled,behavior_enabled", [(True, False), (False, True), (True, True)]
)
def test_aimsun_bo_dispatch_bounds_exports_and_best_application(
    turn_enabled, behavior_enabled, monkeypatch, tmp_path
):
    pytest.importorskip("sklearn", reason="BO execution requires realtwin[bo].")
    config = configure_aimsun_adapters(tmp_path)
    config["AIMSUN"]["turn_inflow"]["turn_inflow"]["is_calibration"] = turn_enabled
    config["AIMSUN"]["behavior"]["behavior"]["is_calibration"] = behavior_enabled
    exports = []
    turn_calls = []
    behavior_calls = []
    cleanup = []

    def export_turn(input_config, **kwargs):
        exports.append(True)
        return {"calibration_info_path": "calibration_info.json"}

    def turn_objective(solution, input_config):
        turn_calls.append(solution.copy())
        turns, inflows = cali_turn_inflow.assignNewTurn(solution, input_config)
        assert turns["turn"].sum() == pytest.approx(100)
        assert inflows["flow"].between(0, 120).all()
        return [float(np.sum(solution**2))]

    def apply_behavior(solution, input_config):
        behavior_calls.append(solution.copy())

    monkeypatch.setattr(
        importlib.import_module("realtwin.rt_aimsun.calibrate_aimsun"),
        "export_turn_inflow_info",
        export_turn,
    )
    monkeypatch.setattr(
        cali_turn_inflow, "fitness_func_turn_inflow_aimsun", turn_objective
    )
    monkeypatch.setattr(
        cali_behavior,
        "export_behavior_info",
        lambda input_config: {
            "subpath_targets": [("test_route", 1, 30.0)],
            "SUBPATH_IDS": {"test_route": "1"},
        },
    )
    monkeypatch.setattr(cali_behavior, "newParameter", apply_behavior)
    monkeypatch.setattr(cali_behavior, "runAimsun", lambda input_config: None)
    monkeypatch.setattr(
        cali_behavior,
        "resultFitness",
        lambda input_config: (float(np.sum(behavior_calls[-1] ** 2)), {}),
    )
    monkeypatch.setattr(
        cali_turn_inflow.TurnInflowCaliAimsun,
        "_clean_up",
        lambda self: cleanup.append("turn_inflow"),
    )
    monkeypatch.setattr(
        cali_behavior.BehaviorCaliAimsun,
        "_clean_up",
        lambda self: cleanup.append("behavior"),
    )

    assert cali_aimsun(
        sel_algo={"turn_inflow": "bo", "behavior": "bo"},
        input_config=config,
        verbose=False,
    )
    assert len(exports) == 1
    assert len(turn_calls) == (7 if turn_enabled else 0)
    assert len(behavior_calls) == (7 if behavior_enabled else 0)
    assert cleanup == [
        stage
        for stage, enabled in [
            ("turn_inflow", turn_enabled),
            ("behavior", behavior_enabled),
        ]
        if enabled
    ]
    model_dir = Path(config["AIMSUN"]["model_fname"]).parent
    for stage, enabled in [
        ("turn_inflow", turn_enabled),
        ("behavior", behavior_enabled),
    ]:
        output = model_dir / f"{stage}_bo_result"
        assert output.exists() is enabled
        if not enabled:
            continue
        points = pd.read_csv(output / "aimsun_bayesopt_2runs_points.csv")
        best_rows = pd.read_csv(output / "aimsun_bayesopt_2runs_best_per_run.csv")
        best = best_rows.loc[best_rows["fitness"].idxmin()]
        parameters = points.filter(regex="^param_").columns.tolist()
        assert len(points) == 6
        assert np.all(
            (points[parameters].to_numpy() >= 0) & (points[parameters].to_numpy() <= 1)
        )
        proposed = best[parameters].to_numpy(dtype=float)
        if stage == "turn_inflow":
            np.testing.assert_allclose(turn_calls[-1], proposed)
        else:
            lower = np.asarray(config["AIMSUN"]["behavior"]["params_lb"])
            upper = np.asarray(config["AIMSUN"]["behavior"]["params_ub"])
            np.testing.assert_allclose(
                behavior_calls[-1], lower + proposed * (upper - lower)
            )
            assert np.all(
                (np.asarray(behavior_calls) >= lower)
                & (np.asarray(behavior_calls) <= upper)
            )
        assert (output / "best_run_fitness_convergence.png").is_file()
        assert not list(output.glob("sumo*"))


def test_behavior_csv_matches_aimsun_assignment_script_order(monkeypatch, tmp_path):
    config = configure_aimsun_adapters(tmp_path)
    config["AIMSUN"]["behavior"]["params_names"] = [
        "MinDist",
        "MaxAcc",
        "NormalDec",
        "SensitivityFactor",
        "MinHeadway",
        "MaxDec",
    ]
    monkeypatch.setattr(cali_behavior, "run_aconsole", lambda command: (0, ""))
    cali_behavior.newParameter(np.array([1.5, 2.6, 4.5, 0.4, 0.8, 8.5]), config)
    values = np.loadtxt(
        tmp_path / "Model" / "DrivingBehaviorParameter.csv", delimiter=","
    )
    np.testing.assert_array_equal(values, [1.5, 2.6, 4.5, 8.5, 0.8, 0.4])


def test_behavior_ranges_can_be_reused_and_accept_shared_names(monkeypatch, tmp_path):
    config = configure_aimsun_adapters(tmp_path)
    config["AIMSUN"]["behavior"]["behavior"]["params_ranges"] = {
        "sigma": [0.3, 0.7],
        "min_gap": [1.3, 1.9],
    }
    monkeypatch.setattr(
        cali_behavior,
        "export_behavior_info",
        lambda input_config: {"subpath_targets": [], "SUBPATH_IDS": {}},
    )
    first = cali_behavior.BehaviorCaliAimsun(config, verbose=False)
    second = cali_behavior.BehaviorCaliAimsun(config, verbose=False)
    assert first.behavior_cfg["params_ranges"] == second.behavior_cfg["params_ranges"]
    assert first.behavior_cfg["params_ranges"]["MinDist"] == [1.3, 1.9]
    assert first.behavior_cfg["params_ranges"]["SensitivityFactor"] == [0.3, 0.7]


@pytest.mark.parametrize(
    "ranges", [{"unknown": [0, 1]}, {"sigma": [1, 0]}, {"sigma": [0, np.nan]}]
)
def test_invalid_behavior_bounds_fail_before_model_export(
    ranges, monkeypatch, tmp_path
):
    config = configure_aimsun_adapters(tmp_path)
    config["AIMSUN"]["behavior"]["behavior"]["params_ranges"] = ranges
    monkeypatch.setattr(
        cali_behavior,
        "export_behavior_info",
        lambda **kwargs: pytest.fail("No model export expected"),
    )
    with pytest.raises(ValueError):
        cali_behavior.BehaviorCaliAimsun(config, verbose=False)


def test_sumo_behavior_can_start_from_prepared_scenario_without_turn_calibration(
    tmp_path,
):
    sumo_directory = tmp_path / "SUMO"
    sumo_directory.mkdir()
    for suffix in (".net.xml", ".rou.xml", ".turn.xml", ".flow.xml"):
        (sumo_directory / f"network{suffix}").write_text("<root/>", encoding="utf-8")
    config = {
        "output_dir": str(tmp_path),
        "Network": {"NetworkName": "network"},
        "Calibration": {
            "scenario_config": {
                "sim_start_time": 100,
                "sim_end_time": 200,
                "calibration_seed": 42,
            }
        },
    }
    scenario = prepare_scenario_config_behavior(config)
    output = Path(scenario["dir_behavior"])
    assert not (sumo_directory / "turn_inflow").exists()
    assert not (output / "EdgeData.xml").exists()
    tree = ET.parse(output / "network.sumocfg")
    assert tree.find("./time/begin").attrib["value"] == "100"
    assert tree.find("./time/end").attrib["value"] == "200"
    assert tree.find("./random/seed").attrib["value"] == "42"
    assert tree.find("./input/additional-files").attrib["value"] == "Edge.add.xml"
    assert (
        ET.parse(output / "Edge.add.xml").find("./edgeData").attrib["file"]
        == "EdgeData.xml"
    )


def test_sumo_behavior_missing_inputs_reports_preparation_step(tmp_path):
    config = {
        "output_dir": str(tmp_path),
        "Network": {"NetworkName": "missing"},
        "Calibration": {"scenario_config": {}},
    }
    with pytest.raises(
        FileNotFoundError, match=r"prepare_simulation\(\).*missing.net.xml"
    ):
        prepare_scenario_config_behavior(config)


@pytest.mark.parametrize("module", [cali_turn_inflow, cali_behavior])
def test_console_keeps_required_output_when_child_exits_abnormally(module):
    # Aimsun can save the model and then crash before Python flushes its output.
    return_code, output = module.run_aconsole(
        [
            sys.executable,
            "-c",
            "import os; print('SUBPATH_ID route=17'); os._exit(7)",
        ]
    )
    assert return_code == 7
    assert "SUBPATH_ID route=17" in output


@pytest.mark.parametrize("module", [cali_turn_inflow, cali_behavior])
def test_console_merges_utf8_output_and_preserves_parent_environment(
    module, monkeypatch
):
    monkeypatch.setenv("REALTWIN_CONSOLE_TEST", "inherited")
    monkeypatch.setenv("PYTHONUNBUFFERED", "0")
    return_code, output = module.run_aconsole(
        [
            sys.executable,
            "-c",
            "import os; "
            "os.write(1, 'caf\\u00e9\\n'.encode('utf-8')); "
            "os.write(2, b'warning\\xff\\n'); "
            "print(os.environ['REALTWIN_CONSOLE_TEST'])",
        ]
    )
    assert return_code == 0
    assert output == "caf\u00e9\nwarning\ufffd\ninherited\n"
    assert os.environ["PYTHONUNBUFFERED"] == "0"


def test_aimsun_behavior_visualization_remains_optional():
    adapter = cali_behavior.BehaviorCaliAimsun.__new__(cali_behavior.BehaviorCaliAimsun)
    assert adapter.run_vis("unused", object()) is True
