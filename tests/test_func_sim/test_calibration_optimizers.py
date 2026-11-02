"""Optimizer construction, adapter side effects, and result export contracts."""

import importlib
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
from mealpy import GA

ADAPTERS = [
    (
        "realtwin.func_lib._f_calibration.algo_sumo.cali_turn_inflow",
        "TurnInflowCali",
        "SUMO",
        "turn_inflow",
    ),
    (
        "realtwin.func_lib._f_calibration.algo_sumo.cali_behavior",
        "BehaviorCali",
        "SUMO",
        "behavior",
    ),
    (
        "realtwin.rt_aimsun.cali_turn_inflow",
        "TurnInflowCaliAimsun",
        "AIMSUN",
        "turn_inflow",
    ),
    ("realtwin.rt_aimsun.cali_behavior", "BehaviorCaliAimsun", "AIMSUN", "behavior"),
]
GA_MODELS = ["BaseGA", "EliteSingleGA", "EliteMultiGA", "MultiGA", "SingleGA"]


@pytest.fixture(
    params=ADAPTERS,
    ids=["sumo-turn", "sumo-behavior", "aimsun-turn", "aimsun-behavior"],
)
def adapter_case(request, monkeypatch):
    module_name, class_name, simulator, stage = request.param
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name)
    adapter = adapter_class.__new__(adapter_class)
    settings = {}
    adapter.turn_inflow_cfg = settings
    adapter.behavior_cfg = settings
    adapter.input_config = {"AIMSUN": {stage: settings}}
    adapter.scenario_config = {"name": "test"}
    adapter.term_dict = {"max_epoch": 20}
    adapter.init_solution = [0.5]
    initial = np.array([[0.5]])
    applied = []
    events = []

    def objective(solution, **kwargs):
        events.append("apply")
        applied.append((solution, kwargs))
        return 1.0

    def generate_initial(values, population):
        assert values is adapter.init_solution
        events.append(("initial", population))
        return initial

    adapter.problem_dict = {"obj_func": objective}
    monkeypatch.setattr(adapter, "_generate_initial_solutions", generate_initial)
    fitness_name = (
        "fitness_func"
        if stage == "behavior"
        else "fitness_func_turn_inflow"
        if simulator == "SUMO"
        else "fitness_func_turn_inflow_aimsun"
    )
    monkeypatch.setattr(module, fitness_name, objective)
    return adapter, settings, simulator, stage, initial, applied, events


@pytest.mark.parametrize("model_name", [None, *GA_MODELS, "unknown", []])
@pytest.mark.parametrize("custom", [False, True])
def test_ga_configuration_and_adapter_side_effects(
    adapter_case, model_name, custom, monkeypatch
):
    adapter, settings, simulator, stage, initial, applied, events = adapter_case
    ga_config = (
        {
            "epoch": 80,
            "pop_size": 4,
            "pc": 0.8,
            "pm": 0.2,
            "selection": "tournament",
            "k_way": 0.4,
            "crossover": "one_point",
            "mutation": "inversion",
            "elite_best": 0.2,
            "elite_worst": 0.4,
        }
        if custom
        else {}
    )
    if model_name is not None:
        ga_config["model_selection"] = model_name
    settings["ga_config"] = ga_config
    original = deepcopy(ga_config)
    best = SimpleNamespace(solution=np.array([0.25]))
    constructed = []

    class Optimizer:
        def __init__(self, selected_model, **parameters):
            events.append("construct")
            self.selected_model = selected_model
            self.parameters = parameters
            constructed.append(self)

        def solve(self, problem, **kwargs):
            events.append("solve")
            assert problem is adapter.problem_dict
            assert kwargs["termination"] is adapter.term_dict
            assert ("starting_solutions" in kwargs) is (
                simulator == "SUMO" and stage == "behavior"
            )
            if "starting_solutions" in kwargs:
                assert kwargs["starting_solutions"] is initial
            return best

    for name in GA_MODELS:
        monkeypatch.setattr(GA, name, partial(Optimizer, name))
    returned_best, model = adapter.run_GA(label="keep")
    expected_model = model_name if model_name in GA_MODELS else "BaseGA"
    assert returned_best is best
    assert constructed == [model]
    assert model.selected_model == expected_model
    expected = {
        "epoch": 80 if custom else 1000,
        "pop_size": 10 if custom else 50,
        "pc": 0.8 if custom else 0.75,
        "pm": 0.2 if custom else 0.1,
        "label": "keep",
    }
    if expected_model != "BaseGA":
        expected.update(
            {
                "selection": "tournament" if custom else "roulette",
                "k_way": 0.4 if custom else 0.2,
                "crossover": "one_point" if custom else "uniform",
                "mutation": "inversion" if custom else "swap",
            }
        )
    if expected_model.startswith("Elite"):
        expected.update(
            {
                "elite_best": 0.2 if custom else 0.1,
                "elite_worst": 0.4 if custom else 0.3,
            }
        )
    assert model.parameters == expected
    assert ga_config == original
    assert adapter.term_dict["max_epoch"] == (
        expected["epoch"] if stage == "turn_inflow" else 20
    )
    assert events == (
        ([("initial", expected["pop_size"])] if stage == "behavior" else [])
        + ["construct", "solve", "apply"]
    )
    assert len(applied) == 1
    assert applied[0][0] is best.solution
    objective_kwargs = (
        {
            "scenario_config": adapter.scenario_config,
            **({"error_func": "rmse"} if stage == "behavior" else {}),
        }
        if simulator == "SUMO"
        else {"input_config": adapter.input_config}
        if stage == "turn_inflow"
        else {}
    )
    assert applied[0][1] == objective_kwargs


def test_ga_missing_configuration_fails_before_side_effects(adapter_case):
    adapter, _, _, _, _, applied, events = adapter_case
    with pytest.raises(ValueError, match="ga_config is not provided"):
        adapter.run_GA()
    assert not events
    assert not applied


def test_ga_duplicate_keyword_is_rejected(adapter_case):
    adapter, settings, _, _, _, applied, events = adapter_case
    settings["ga_config"] = {}
    with pytest.raises(TypeError, match="multiple values.*epoch"):
        adapter.run_GA(epoch=2)
    assert "solve" not in events
    assert not applied


@pytest.mark.parametrize(
    "module_name,class_name,simulator,stage",
    ADAPTERS[:3],
    ids=["sumo-turn", "sumo-behavior", "aimsun-turn"],
)
@pytest.mark.parametrize(
    "mode", ["history", "history-error", "delegate", "delegate-false", "delegate-error"]
)
def test_result_export_contract(
    module_name, class_name, simulator, stage, mode, tmp_path
):
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name)
    adapter = adapter_class.__new__(adapter_class)
    output_dir = tmp_path / "results"
    calls = []

    def export(**kwargs):
        assert output_dir.is_dir()
        calls.append(kwargs)
        if mode.endswith("error"):
            raise RuntimeError("export failed")
        return mode != "delegate-false"

    if mode.startswith("delegate"):
        model = SimpleNamespace(run_vis=export)
    else:
        model = SimpleNamespace(
            history=SimpleNamespace(save_global_objectives_chart=export)
        )
    if mode == "delegate-error":
        with pytest.raises(RuntimeError, match="export failed"):
            adapter.run_vis(str(output_dir), model)
    else:
        assert adapter.run_vis(str(output_dir), model) is (
            mode not in {"history-error", "delegate-false"}
        )
    assert calls == (
        [{"output_dir": str(output_dir)}]
        if mode.startswith("delegate")
        else [{"filename": f"{output_dir}/global_objectives"}]
    )


@pytest.mark.parametrize("model_name", GA_MODELS)
def test_ga_variants_run_with_real_mealpy(model_name, monkeypatch):
    from mealpy import FloatVar

    evaluations = []

    def objective(solution, **kwargs):
        fitness = float(np.sum((solution - 0.3) ** 2))
        evaluations.append(fitness)
        return fitness

    # Exercise the public adapter with the real optimizer and an in-memory objective.
    module = importlib.import_module(ADAPTERS[0][0])
    adapter = module.TurnInflowCali.__new__(module.TurnInflowCali)
    adapter.turn_inflow_cfg = {
        "ga_config": {"model_selection": model_name, "epoch": 2, "pop_size": 10}
    }
    adapter.scenario_config = {}
    adapter.term_dict = {"max_epoch": 2}
    # Swap mutation needs multiple parameters, as in behavior calibration.
    adapter.problem_dict = {
        "bounds": FloatVar(lb=[0.0] * 6, ub=[1.0] * 6),
        "obj_func": objective,
        "minmax": "min",
        "log_to": None,
    }
    monkeypatch.setattr(module, "fitness_func_turn_inflow", objective)
    best, model = adapter.run_GA()
    assert type(model).__name__ == model_name
    assert np.all((best.solution >= 0) & (best.solution <= 1))
    assert best.target.fitness == pytest.approx(min(evaluations))
