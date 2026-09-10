"""BO search, calibration dispatch, and result-export regressions."""

import builtins
from copy import deepcopy
import importlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="BO tests require the realtwin[bo] extra.")

from realtwin import RealTwinSUMO
from realtwin.func_lib._f_calibration import calibration_sumo
from realtwin.func_lib._f_calibration.algo_sumo import cali_behavior, cali_turn_inflow
from realtwin.func_lib._f_calibration.algo_sumo._bayesian_opt import (
    BayesianOptimization,
)
from realtwin.func_lib._f_calibration.algo_sumo._bayesian_opt_util import (
    online_optimization,
)


@pytest.mark.parametrize(
    "kernel", ["RBF", "Matern", "RationalQuadratic", "ExpSineSquared", "Combined"]
)
def test_search_uses_gp_and_records_best_bounded_point(kernel):
    calls = []

    def objective(point):
        calls.append(point.copy())
        return [float(np.sum((point - 0.3) ** 2))]

    best, solution, points, values = online_optimization(
        1,
        ([0], [1]),
        objective,
        target=-1,
        tolerance=0,
        random_points=32,
        max_evaluations=7,
        kernel_type=kernel,
    )

    assert len(calls) == 7
    np.testing.assert_allclose(points, calls)
    np.testing.assert_allclose(values, np.sum((points - 0.3) ** 2, axis=1))
    assert np.all((points >= 0) & (points <= 1))
    assert len(np.unique(points, axis=0)) == len(points)
    assert best == values.min()
    np.testing.assert_array_equal(solution, points[values.argmin()])


@pytest.mark.parametrize("budget", [1, 2, 8])
def test_constant_fitness_and_small_budgets(budget):
    best, _, points, values = online_optimization(
        2,
        ([0, 10], [1, 10]),
        lambda point: 5.0,
        target=0,
        tolerance=0,
        random_points=budget,
        max_evaluations=budget,
    )
    assert best == 5
    assert len(values) == budget
    assert np.isfinite(points).all()
    assert np.all(points[:, 1] == 10)


def test_fully_fixed_bounds_evaluate_once():
    best, solution, points, values = online_optimization(
        2,
        ([2, 3], [2, 3]),
        lambda point: sum(point),
        target=0,
        tolerance=0,
        random_points=10,
        max_evaluations=10,
    )
    assert best == 5
    assert len(points) == len(values) == 1
    np.testing.assert_array_equal(solution, [2, 3])


def test_target_stops_during_initialization():
    _, _, points, values = online_optimization(
        3,
        ([0] * 3, [1] * 3),
        lambda point: np.array([0.5]),
        target=0,
        tolerance=1,
        random_points=20,
        max_evaluations=20,
    )
    assert len(points) == len(values) == 1


def test_seed_reproduces_search_and_objective_mutation_cannot_corrupt_history():
    def objective(point):
        fitness = float(np.sum(point**2))
        point[:] = -100
        return fitness

    options = dict(
        num_params=1,
        variable_bounds=([0], [1]),
        evaluation_function=objective,
        target=-1,
        tolerance=0,
        random_points=20,
        max_evaluations=6,
    )
    first = online_optimization(**options, seed=12)
    second = online_optimization(**options, seed=12)
    different = online_optimization(**options, seed=13)

    np.testing.assert_array_equal(first[2], second[2])
    np.testing.assert_array_equal(first[3], second[3])
    assert not np.array_equal(first[2], different[2])
    np.testing.assert_allclose(first[3], np.sum(first[2] ** 2, axis=1))


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"max_evaluations": 0}, "max_evaluations"),
        ({"max_evaluations": True}, "max_evaluations"),
        ({"random_points": 1}, "random_points"),
        ({"random_points": 2.5}, "random_points"),
        ({"num_params": 0}, "num_params"),
        ({"variable_bounds": ([2], [1])}, "lower bound"),
        ({"variable_bounds": ([0, 0], [1, 1])}, "variable_bounds"),
        ({"variable_bounds": ([0], [np.inf])}, "variable_bounds"),
        ({"tolerance": -1}, "tolerance"),
        ({"target": np.nan}, "target"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.2}, "seed"),
        ({"kernel_type": "unsupported"}, "kernel_type"),
    ],
)
def test_invalid_settings_fail_before_objective(updates, message):
    def objective(point):
        pytest.fail("Invalid settings must fail before starting a simulation.")

    options = dict(
        num_params=1,
        variable_bounds=([0], [1]),
        evaluation_function=objective,
        random_points=10,
        max_evaluations=5,
    )
    options.update(updates)
    with pytest.raises(ValueError, match=message):
        online_optimization(**options)


@pytest.mark.parametrize("value", [np.nan, np.inf, [1, 2], []])
def test_rejects_nonfinite_or_multiple_objectives(value):
    with pytest.raises(ValueError, match="one finite fitness"):
        online_optimization(
            1,
            ([0], [1]),
            lambda point: value,
            random_points=2,
            max_evaluations=2,
        )


def test_missing_optional_dependency_has_install_command(monkeypatch):
    original_import = builtins.__import__

    def import_without_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ModuleNotFoundError("No module named sklearn")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sklearn)
    with pytest.raises(ImportError, match=r"realtwin\[bo\]"):
        online_optimization(1, ([0], [1]), lambda point: 0)


def test_repeated_runs_export_all_points_and_global_best(tmp_path):
    optimizer = BayesianOptimization(
        {},
        {
            "bo_config": {
                "total_run": 2,
                "seed": 17,
                "random_points": 16,
                "max_evaluations": 5,
                "target": -1,
                "tolerance": 0,
            }
        },
        bounds=([0], [1]),
    )
    best = optimizer.solve(lambda point, scenario_config: [float(point[0] ** 2)])
    assert len(optimizer.points_df) == 10
    assert len(optimizer.best_df) == 2
    assert best.target.fitness == optimizer.points_df["fitness"].min()
    assert not np.array_equal(
        optimizer.points_df.query("run == 1")["param_1"].values,
        optimizer.points_df.query("run == 2")["param_1"].values,
    )
    np.testing.assert_allclose(best.solution, optimizer.best_solution)
    assert optimizer.run_vis(str(tmp_path / "exports")) is True

    exported = tmp_path / "exports"
    points = pd.read_csv(exported / "sumo_bayesopt_2runs_points.csv")
    best_rows = pd.read_csv(exported / "sumo_bayesopt_2runs_best_per_run.csv")
    results = pd.read_csv(exported / "sumo_bayesopt_2runs_results.csv")
    assert list(points.columns) == [
        "run",
        "iteration",
        "param_1",
        "fitness",
        "run_time_sec",
    ]
    assert len(results) == len(points) == 10
    assert best_rows["fitness"].tolist() == pytest.approx(
        points.groupby("run")["fitness"].min().tolist()
    )
    assert (best_rows["run_time_sec"] >= 0).all()
    assert len(list(exported.glob("*.png"))) == 3
    assert all(path.stat().st_size > 1000 for path in exported.glob("*.png"))


@pytest.mark.parametrize(
    "settings",
    [{"total_run": 0}, {"total_run": True}, {"total_run": 1001}, {"seed": -1}],
)
def test_invalid_run_settings_do_not_call_objective(settings):
    optimizer = BayesianOptimization({}, {"bo_config": settings}, bounds=([0], [1]))
    with pytest.raises(ValueError):
        optimizer.solve(
            lambda point, scenario_config: pytest.fail("Unexpected evaluation")
        )


def test_behavior_bounds_do_not_require_turn_counts():
    optimizer = BayesianOptimization(
        {"behavior_parameters_ranges": {"gap": [1, 3], "tau": [0.2, 1]}}
    )
    assert optimizer.bounds == ([1, 0.2], [3, 1])
    assert optimizer.n_variable == 2


def test_turn_adapter_uses_custom_objective_and_restores_global_best():
    calls = []

    def objective(point, scenario_config):
        calls.append(point.copy())
        return [float(np.sum(point**2))]

    adapter = cali_turn_inflow.TurnInflowCali(
        {"N_Variable": 2, "N_TurnVariable": 1, "N_InflowVariable": 1, "max_inflow": 40},
        {
            "bo_config": {
                "total_run": 2,
                "random_points": 8,
                "max_evaluations": 3,
                "target": -1,
                "tolerance": 0,
            }
        },
        verbose=False,
        fitness_func=objective,
    )
    best, optimizer = adapter.run_BO()
    assert len(calls) == len(optimizer.points_df) + 1 == 7
    np.testing.assert_array_equal(calls[-1], best.solution)
    assert best.target.fitness == min(float(np.sum(point**2)) for point in calls[:-1])
    assert np.all(optimizer.points_df["param_1"] <= 1)
    assert np.all(optimizer.points_df["param_2"] <= 40)


def test_behavior_adapter_uses_configured_bounds_and_restores_best(monkeypatch):
    calls = []

    def objective(point, scenario_config, error_func="rmse"):
        calls.append(point.copy())
        return float(np.sum(point**2))

    monkeypatch.setattr(cali_behavior, "fitness_func", objective)
    adapter = cali_behavior.BehaviorCali(
        {},
        {
            "behavior": {"params_ranges": {"gap": [1.5, 2], "tau": [0.5, 0.8]}},
            "bo_config": {
                "random_points": 8,
                "max_evaluations": 3,
                "target": -1,
                "tolerance": 0,
            },
        },
        verbose=False,
    )
    best, optimizer = adapter.run_BO()
    assert len(calls) == 4
    np.testing.assert_array_equal(calls[-1], best.solution)
    assert optimizer.points_df["param_1"].between(1.5, 2).all()
    assert optimizer.points_df["param_2"].between(0.5, 0.8).all()


@pytest.mark.parametrize(
    "turn_enabled, behavior_enabled", [(True, True), (True, False), (False, True)]
)
def test_public_bo_pipeline_exports_enabled_stages_and_applies_overrides(
    turn_enabled,
    behavior_enabled,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    turn_directory = tmp_path / "turn_inflow"
    behavior_directory = tmp_path / "behavior"
    turn_directory.mkdir()
    behavior_directory.mkdir()
    base_scenario = {"N_Variable": 1, "N_TurnVariable": 1, "N_InflowVariable": 0}
    turn_scenario = {**base_scenario, "dir_turn_inflow": str(turn_directory)}
    behavior_scenario = {"dir_behavior": str(behavior_directory)}
    monkeypatch.setattr(
        calibration_sumo,
        "prepare_scenario_config_turn_inflow",
        lambda config: turn_scenario,
    )
    monkeypatch.setattr(
        calibration_sumo,
        "prepare_scenario_config_behavior",
        lambda config: behavior_scenario,
    )
    monkeypatch.setattr(cali_turn_inflow.TurnInflowCali, "_clean_up", lambda self: None)

    def objective(point, scenario_config, **kwargs):
        stage_directory = scenario_config.get(
            "dir_turn_inflow", scenario_config.get("dir_behavior")
        )
        monkeypatch.chdir(stage_directory)
        return float(np.sum(point**2))

    monkeypatch.setattr(cali_turn_inflow, "fitness_func_turn_inflow", objective)
    monkeypatch.setattr(cali_behavior, "fitness_func", objective)
    twin = RealTwinSUMO.__new__(RealTwinSUMO)
    twin.verbose = False
    twin.input_config = {
        "demo_data": False,
        "SUMO": {},
        "Calibration": {
            "scenario_config": {},
            "turn_inflow": {"is_calibration": turn_enabled},
            "behavior": {"is_calibration": behavior_enabled},
            "bo_config": {
                "random_points": 16,
                "max_evaluations": 8,
                "seed": 31,
                "target": -1,
                "tolerance": 0,
            },
        },
    }
    original = deepcopy(twin.input_config["Calibration"])
    assert (
        twin.calibrate(
            sel_algo={"turn_inflow": "BO", "behavior": "bo"},
            sel_behavior_routes={"route_1": {"time": 12, "route_list": ["edge1"]}},
            update_turn_inflow_algo={"bo_config": {"max_evaluations": 2}},
            update_behavior_algo={"bo_config": {"max_evaluations": 3}},
        )
        is True
    )

    for stage, enabled, budget in [
        ("turn_inflow", turn_enabled, 2),
        ("behavior", behavior_enabled, 3),
    ]:
        stage_config = twin.input_config["SUMO"][stage]["bo_config"]
        assert stage_config["max_evaluations"] == budget
        assert stage_config["seed"] == 31
        path = (
            tmp_path / stage / f"{stage}_bo_result" / "sumo_bayesopt_1runs_points.csv"
        )
        assert path.exists() is enabled
        if enabled:
            assert len(pd.read_csv(path)) == budget
    assert twin.input_config["Calibration"]["bo_config"] == original["bo_config"]


def test_existing_default_selection_is_preserved(monkeypatch):
    calls = []

    def calibration(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(
        importlib.import_module("realtwin.rt_sumo.rt_sumo"), "cali_sumo", calibration
    )
    twin = RealTwinSUMO.__new__(RealTwinSUMO)
    twin.verbose = False
    twin.input_config = {
        "demo_data": False,
        "SUMO": {},
        "Calibration": {
            "turn_inflow": {"is_calibration": True},
            "behavior": {"is_calibration": True},
        },
    }
    assert twin.calibrate() is True
    assert calls[0]["sel_algo"] == {"turn_inflow": "ga", "behavior": "ga"}


def test_periodic_kernel_supports_six_behavior_parameters():
    best, _, points, values = online_optimization(
        6,
        ([0] * 6, [1] * 6),
        lambda point: np.sum((point - 0.3) ** 2),
        target=-1,
        tolerance=0,
        random_points=40,
        max_evaluations=13,
        kernel_type="ExpSineSquared",
    )
    assert len(values) == 13
    assert np.isfinite(values).all()
    assert best == values.min()
    assert np.all((points >= 0) & (points <= 1))


def test_behavior_preparation_can_use_existing_turn_files_without_turn_calibration(
    tmp_path,
):
    sumo_path = tmp_path / "SUMO"
    turn_path = sumo_path / "turn_inflow"
    turn_path.mkdir(parents=True)
    files = [
        "chatt.sumocfg",
        "chatt.net.xml",
        "Edge.add.xml",
        "chatt.rou.xml",
        "chatt.turn.xml",
        "chatt.flow.xml",
        "EdgeData.xml",
    ]
    for filename in files:
        (turn_path / filename).write_text("<root/>", encoding="utf-8")
    configuration = {
        "output_dir": str(tmp_path),
        "Network": {"NetworkName": "chatt"},
        "Calibration": {
            "scenario_config": {"sim_start_time": 28800, "sim_end_time": 32400}
        },
    }
    scenario = calibration_sumo.prepare_scenario_config_behavior(configuration)
    assert scenario["network_name"] == "chatt"
    assert scenario["sim_name"] == "chatt.sumocfg"
    assert all((sumo_path / "behavior" / filename).is_file() for filename in files)
