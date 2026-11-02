"""Calibration configuration and COM handoff regressions without a Vissim licence."""

import importlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from realtwin.rt_vissim import cali_behavior, calibrate, rt_vissim
from realtwin.rt_vissim.network import VissimLink

stage = importlib.import_module("realtwin.rt_vissim.06_calibrate_turn_inflow")


@pytest.fixture
def twin(tmp_path):
    model = rt_vissim.RealTwinVISSIM.__new__(rt_vissim.RealTwinVISSIM)
    model.input_config = {
        "input_dir": str(tmp_path / "inputs"),
        "output_dir": str(tmp_path / "outputs"),
        "Network": {"NetworkName": "custom"},
        "VISSIM": {"progid": "Vissim.Vissim.2026"},
        "Calibration": {
            "scenario_config": {"sim_start_time": 28800, "sim_end_time": 32400},
            "turn_inflow": {"is_calibration": True, "max_inflow": 200},
            "behavior": {"is_calibration": False,
                         "params_ranges": {"W74ax": [1, 3]},
                         "driving_behavior_ids": [1],
                         "sel_behavior_routes": [["route", 1, 2, 30]]},
            "sa_config": {"epoch": 10, "step_size": 0.1},
        },
    }
    model.vissim_module = {"step6": "realtwin.rt_vissim.06_calibrate_turn_inflow"}
    network = tmp_path / "outputs" / "VISSIM" / "custom_demand_signals.inpx"
    network.parent.mkdir(parents=True)
    network.write_text('''<network><drivingBehaviors>
    <drivingBehavior no="1" carFollowModType="WIEDEMANN74"/>
    </drivingBehaviors><links><link no="1"/><link no="2"/></links></network>''')
    return model


@pytest.mark.parametrize("algorithm", ["GA", "SA", "TS", "BO"])
def test_calibrate_passes_live_config_and_user_paths(twin, monkeypatch, algorithm):
    shared = deepcopy(twin.input_config["Calibration"])
    observed = {}

    def run(command):
        observed["command"] = command
        path = Path(command[command.index("--config") + 1])
        observed["path"] = path
        observed["config"] = json.loads(path.read_text())

    monkeypatch.setattr(rt_vissim, "_run_vissim_stage", run)
    assert twin.calibrate(
        sel_algo={"turn_inflow": algorithm},
        update_turn_inflow_algo={
            "sa_config": {"epoch": 1},
            "scenario_config": {"sim_end_time": 29700},
            "turn_inflow": {"max_inflow": 500, "seed": 7},
        },
    ) is True
    command = observed["command"]
    assert command[1:3] == ["-m", twin.vissim_module["step6"]]
    assert command[command.index("--algo") + 1] == algorithm.lower()
    assert command[command.index("--matchup") + 1].endswith("MatchupTable.xlsx")
    assert command[command.index("--inpx") + 1].endswith("custom_demand_signals.inpx")
    assert Path(command[command.index("--traffic-dir") + 1]).parts[-2:] == ("inputs", "Traffic")
    effective = observed["config"]["Calibration"]
    assert effective["sa_config"] == {"epoch": 1, "step_size": 0.1}
    assert effective["turn_inflow"]["max_inflow"] == 500
    assert effective["scenario_config"]["sim_end_time"] == 29700
    assert observed["config"]["VISSIM"]["progid"] == "Vissim.Vissim.2026"
    assert twin.input_config["Calibration"] == shared
    assert not observed["path"].exists()


@pytest.mark.parametrize("algorithm", ["unknown"])
def test_unsupported_algorithm_does_not_launch(twin, monkeypatch, algorithm):
    monkeypatch.setattr(rt_vissim, "_run_vissim_stage", lambda cmd: pytest.fail("launched"))
    assert twin.calibrate(sel_algo={"turn_inflow": algorithm}) is False


@pytest.mark.parametrize("stage_name", ["turn_inflow", "behavior"])
def test_calibrate_uses_demand_network_when_control_disabled(twin, monkeypatch, stage_name):
    """Both calibration stages and behavior preflight must select the demand file."""
    twin.input_config["VISSIM"]["is_control_data"] = False
    for name in ("turn_inflow", "behavior"):
        twin.input_config["Calibration"][name]["is_calibration"] = name == stage_name
    network_dir = Path(twin.input_config["output_dir"]) / "VISSIM"
    signal_network = network_dir / "custom_demand_signals.inpx"
    demand_network = network_dir / "custom_demand.inpx"
    demand_network.write_bytes(signal_network.read_bytes())
    # Invalid stale signal data would fail the behavior preflight if selected.
    signal_network.write_text("<network/>", encoding="utf-8")
    commands = []
    monkeypatch.setattr(rt_vissim, "_run_vissim_stage", commands.append)

    assert twin.calibrate() is True

    assert len(commands) == 1
    command = commands[0]
    assert Path(command[command.index("--inpx") + 1]) == demand_network
    assert command[command.index("--stage") + 1] == stage_name


def test_enabled_flags_use_overrides(twin, monkeypatch):
    commands = []
    monkeypatch.setattr(rt_vissim, "_run_vissim_stage", commands.append)
    twin.input_config["Calibration"]["turn_inflow"]["is_calibration"] = False
    assert twin.calibrate() is False
    assert twin.calibrate(
        update_turn_inflow_algo={"turn_inflow": {"is_calibration": True}}
    ) is True
    assert len(commands) == 1
    assert twin.calibrate(
        update_turn_inflow_algo={"turn_inflow": {"is_calibration": False}}
    ) is False
    assert len(commands) == 1


def test_stages_are_independent_and_behavior_uses_saved_demand(twin, monkeypatch, tmp_path):
    commands = []
    snapshots = []
    calibrated = tmp_path / "custom_demand.inpx"

    def run(command):
        commands.append(command)
        snapshots.append(json.loads(Path(command[command.index("--config") + 1]).read_text()))
        result = Path(command[command.index("--result") + 1])
        result.write_text(json.dumps({"network": str(calibrated)}))

    monkeypatch.setattr(rt_vissim, "_run_vissim_stage", run)
    twin.input_config["Calibration"]["behavior"]["is_calibration"] = True
    shared = deepcopy(twin.input_config["Calibration"])
    routes = [["route", 1, 2, 30]]
    assert twin.calibrate(
        sel_algo={"turn_inflow": "SA", "behavior": "BO"}, sel_behavior_routes=routes,
        update_behavior_algo={"bo_config": {"max_evaluations": 7},
                              "behavior": {"seed": 13}},
    ) is True
    assert [cmd[cmd.index("--stage") + 1] for cmd in commands] == ["turn_inflow", "behavior"]
    assert commands[1][commands[1].index("--inpx") + 1] == str(calibrated)
    assert "--traffic-dir" not in commands[1]
    assert snapshots[1]["Calibration"]["sel_behavior_routes"] == routes
    assert snapshots[1]["Calibration"]["behavior"]["seed"] == 13
    assert "bo_config" not in snapshots[0]["Calibration"]
    assert twin.input_config["Calibration"] == shared
    commands.clear()
    assert twin.calibrate(
        update_turn_inflow_algo={"turn_inflow": {"is_calibration": False}}
    ) is True
    assert len(commands) == 1 and commands[0][commands[0].index("--stage") + 1] == "behavior"
    commands.clear()
    assert twin.calibrate(
        update_behavior_algo={"behavior": {"is_calibration": False}}
    ) is True
    assert len(commands) == 1 and commands[0][commands[0].index("--stage") + 1] == "turn_inflow"


def test_no_variable_handoff_ignores_previous_output(twin, monkeypatch, tmp_path):
    twin.input_config["Calibration"]["behavior"]["is_calibration"] = True
    commands = []

    def run(command):
        commands.append(command)
        source = command[command.index("--inpx") + 1]
        Path(command[command.index("--result") + 1]).write_text(json.dumps({"network": source}))

    monkeypatch.setattr(rt_vissim, "_run_vissim_stage", run)
    assert twin.calibrate() is True
    assert commands[1][commands[1].index("--inpx") + 1].endswith("custom_demand_signals.inpx")


def test_stage_failure_propagates_and_cleans_snapshot(twin, monkeypatch):
    captured = []

    def fail(command):
        captured.append(Path(command[command.index("--config") + 1]))
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(rt_vissim, "_run_vissim_stage", fail)
    with pytest.raises(subprocess.CalledProcessError):
        twin.calibrate()
    assert not captured[0].exists()


def test_vissim_behavior_settings_replace_shared_ranges_without_mutating_them():
    calibration = {
        "behavior": {"is_calibration": True, "params_ranges": {"min_gap": [1, 3]},
                     "vissim": {"params_ranges": {"W74ax": [1, 3]},
                                "driving_behavior_ids": [1]}}
    }
    original = deepcopy(calibration)

    resolved = cali_behavior.resolve_behavior_config(calibration)

    assert resolved["behavior"]["params_ranges"] == {"W74ax": [1, 3]}
    assert "vissim" not in resolved["behavior"]
    resolved["behavior"]["params_ranges"]["W74ax"][0] = 2
    assert calibration == original


def test_behavior_native_settings_and_call_overrides_reach_subprocess(twin, monkeypatch):
    behavior = twin.input_config["Calibration"]["behavior"]
    behavior.update({"is_calibration": True, "params_ranges": {"min_gap": [1, 3]},
                     "vissim": {"params_ranges": {"W74ax": [1, 3]},
                                "driving_behavior_ids": [1]}})
    twin.input_config["Calibration"]["turn_inflow"]["is_calibration"] = False
    original = deepcopy(twin.input_config["Calibration"])
    snapshots = []

    def run(command):
        snapshots.append(json.loads(Path(command[command.index("--config") + 1]).read_text()))

    monkeypatch.setattr(rt_vissim, "_run_vissim_stage", run)
    assert twin.calibrate(
        sel_algo={"behavior": "BO"}, sel_behavior_routes=[["argument", 1, 2, 35]],
        update_behavior_algo={"behavior": {"params_ranges": {"W74bxAdd": [2, 4]}}},
    ) is True
    assert snapshots[0]["Calibration"]["behavior"]["params_ranges"] == {"W74bxAdd": [2, 4]}
    assert snapshots[0]["Calibration"]["sel_behavior_routes"] == [["argument", 1, 2, 35]]
    assert twin.calibrate(sel_algo={"behavior": "BO"}) is True
    assert snapshots[1]["Calibration"]["behavior"]["params_ranges"] == {"W74ax": [1, 3]}
    assert twin.input_config["Calibration"] == original


@pytest.mark.parametrize("invalid_field", ["params_ranges", "sel_behavior_routes"])
def test_invalid_behavior_fails_in_parent_before_any_stage(twin, monkeypatch, invalid_field):
    behavior = twin.input_config["Calibration"]["behavior"]
    behavior["is_calibration"] = True
    behavior.pop(invalid_field)
    monkeypatch.setattr(rt_vissim, "_run_vissim_stage", lambda cmd: pytest.fail("launched"))

    with pytest.raises(ValueError, match=invalid_field):
        twin.calibrate(sel_algo={"turn_inflow": "BO", "behavior": "BO"})

    assert not list((Path(twin.input_config["output_dir"]) / "VISSIM").glob("calibration_*"))


def test_tutorial_supplies_native_vissim_behavior_parameters():
    path = Path(__file__).parents[2] / "tutorials" / "realtwin_config.yaml"
    calibration = stage.load_config(path)["Calibration"]
    behavior = cali_behavior.resolve_behavior_config(calibration)["behavior"]

    assert behavior["params_ranges"] == {"W74ax": [1.0, 3.0], "W74bxAdd": [1.0, 4.0]}
    assert behavior["driving_behavior_ids"] == [1]
    assert behavior["calibration_time_step"] == 0.1
    assert behavior["sel_behavior_routes"][0][1:3] == [9, 15]
    assert behavior["sel_behavior_routes"][0][3] > 0
    assert behavior["reference_source"]["kind"] == "simulated_baseline"
    assert calibration["behavior"]["initial_params"]["min_gap"] == 2.5


@pytest.mark.parametrize("text, expected", [("08:00", 28800), ("08:00:01", 28801),
                                           ("28801", 28801), (32400, 32400)])
def test_clock_times(text, expected):
    assert stage.clock_to_seconds(text) == expected


@pytest.mark.parametrize("text", ["08:60", "08:00:60", "-1", "nan", "28800.5"])
def test_bad_clock_times(text):
    with pytest.raises(ValueError):
        stage.clock_to_seconds(text)


def test_invalid_config_is_not_silently_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        stage.config_defaults(tmp_path / "missing.yaml")
    config = tmp_path / "bad.yaml"
    config.write_text("- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        stage.config_defaults(config)


def test_score_honors_configured_threshold():
    counted = {(1, 2): 100}
    modelled = {(1, 2): 150}
    assert calibrate.score(modelled, counted)["share"] == 1
    assert calibrate.score(modelled, counted, target_geh=3)["approach_share"] == 0


@pytest.mark.parametrize("stage_name", ["turn_inflow", "behavior"])
def test_stage_stop_limits_do_not_override_algorithm_epoch(stage_name):
    config = {"scenario_config": {"max_epoch": 100, "max_time": 600},
              stage_name: {"max_epoch": 2, "max_fe": 20, "max_early_stop": 5}}
    assert calibrate.calibration_termination(config, stage_name, {"epoch": 1}) == {
        "max_epoch": 1, "max_fe": 20, "max_time": 600, "max_early_stop": 5}
    assert calibrate.calibration_termination(config, stage_name, {"epoch": 10})["max_epoch"] == 2


def test_invalid_stop_limit_fails_before_com(behavior_inputs, behavior_session):
    network, calibration, config, _ = behavior_inputs
    calibration["behavior"]["max_epoch"] = 0
    config.write_text(json.dumps({"Calibration": calibration}))
    with pytest.raises(ValueError, match="max_epoch"):
        stage.main(["--stage", "behavior", "--inpx", str(network), "--config", str(config)])
    assert not behavior_session.instances


def test_split_uses_its_own_intervals(tmp_path):
    network = tmp_path / "network.inpx"
    network.write_text('''<network><vehicleInputs/>
    <vehicleRoutingDecisionsStatic><vehicleRoutingDecisionStatic no="1" link="1">
    <vehRoutSta><vehicleRouteStatic no="1" relFlow="2 0:50, 2 900:50">
    <linkSeq><intObjectRef key="2"/></linkSeq></vehicleRouteStatic>
    <vehicleRouteStatic no="2" relFlow="2 0:50, 2 900:50">
    <linkSeq><intObjectRef key="3"/></linkSeq></vehicleRouteStatic></vehRoutSta>
    </vehicleRoutingDecisionStatic></vehicleRoutingDecisionsStatic></network>''')
    links = {no: VissimLink(no=no, num_lanes=1, length=100) for no in (1, 2, 3)}
    movements = pd.DataFrame({"FromLinkNo_Vissim": [1, 1], "ToLinkNo_Vissim": [2, 3],
                              "Turn": ["thru", "right"], "JunctionID_OpenDrive": ["J_A", "J_A"]})
    problem, _ = calibrate.build_problem(network, links, movements, set(),
                                        needs_calibration={"J_A"})
    assert problem.size == 1
    assert problem.splits[0].intervals == 2


def test_counting_sites_keeps_parallel_lane_paths():
    links = {no: VissimLink(no=no) for no in (1, 2, 3, 4)}
    links.update({no: VissimLink(no=no, junction_key="J_A") for no in (5, 6, 7)})
    # Two lane paths for 1->2 and one for 1->3; no cross-movement counting.
    for number, source, target in ((10, 1, 5), (11, 5, 2), (12, 1, 6),
                                   (13, 6, 2), (14, 1, 7), (15, 7, 3)):
        links[number] = VissimLink(no=number, is_connector=True,
                                   from_link=source, to_link=target)
    sites, notes = calibrate.counting_sites(links, [(1, 2), (1, 3), (1, 4)])
    assert sites == {(1, 2): [5, 6], (1, 3): [7]}
    assert len(notes) == 1


def test_counting_sites_avoids_shared_internal_links():
    links = {no: VissimLink(no=no) for no in (1, 2, 3)}
    links[4] = VissimLink(no=4, junction_key="J_A")
    for number, source, target in ((10, 1, 4), (11, 4, 2), (12, 4, 3)):
        links[number] = VissimLink(no=number, is_connector=True,
                                   from_link=source, to_link=target)
    sites, notes = calibrate.counting_sites(links, [(1, 2)])
    # The unobserved 1->3 traffic shares link 4, so measure connector 11.
    assert sites == {(1, 2): [11]}
    assert not notes


@pytest.fixture
def fake_session(monkeypatch):
    """Mirror the COM ownership and prohibit editing demand before a reset."""
    class Attributes:
        def __init__(self):
            self.attributes = {}

        def SetAttValue(self, name, value):
            self.attributes[name] = value

    class Session:
        instances = []

        def __init__(self, progid=None, visible=False):
            self.events = []
            self.volume = 0
            self.running = False
            simulation = Attributes()

            def stop():
                self.events.append("stop")
                self.running = False

            def run():
                self.events.append("run")
                self.running = True

            def set_volume(name, value):
                assert not self.running
                self.events.append("apply")
                self.volume = value

            simulation.Stop = stop
            simulation.RunContinuous = run
            self.vissim = SimpleNamespace(Simulation=simulation, Evaluation=Attributes())
            self.net = SimpleNamespace(VehicleInputs=SimpleNamespace(
                ItemByKey=lambda no: SimpleNamespace(SetAttValue=set_volume)))
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def load_net(self, path):
            pass

        def save_net_as(self, path):
            path.write_text(f'<network volume="{self.volume}"/>')

    monkeypatch.setattr(calibrate, "place_counters", lambda session, sites: {
        (1, 2): SimpleNamespace(AttValue=lambda name: session.volume)})
    com = importlib.import_module("realtwin.rt_vissim.com")
    monkeypatch.setattr(com, "VissimSession", Session)
    return Session


def test_evaluator_resets_each_candidate(fake_session, tmp_path):
    session = fake_session()
    problem = calibrate.Problem([calibrate.InflowKnob(1, 1, intervals=1)],
                               counted={(1, 2): 100}, max_inflow=400)
    history = tmp_path / "new" / "history.csv"
    evaluator = calibrate.Evaluator(session, problem, 900, {(1, 2): [2]},
                                    seed=7, resolution=4, start_time=28800,
                                    target_geh=3, history=history)
    assert evaluator([0.25]) == 0
    assert evaluator([0.75]) > 0
    assert session.events == ["stop", "apply", "run", "stop", "apply", "run"]
    assert session.vissim.Simulation.attributes["StartTm"] == 28800
    assert session.vissim.Simulation.attributes["RandSeed"] == 7
    assert session.vissim.Evaluation.attributes["DataCollInterval"] == 900
    assert calibrate.read_history(history)["target_geh"] == 3
    assert evaluator.best[1] == [0.25]


@pytest.fixture
def cli_inputs(tmp_path, monkeypatch):
    paths = {}
    for name in ("inpx", "matchup", "links", "movements"):
        path = tmp_path / name
        path.write_text("<network/>" if name == "inpx" else "")
        paths[name] = path
    paths["traffic-dir"] = tmp_path / "Traffic"
    paths["traffic-dir"].mkdir()
    counts = pd.DataFrame({"IntervalStart": [28800], "IntervalEnd": [29700],
                          "FromLinkNo_Vissim": [1], "ToLinkNo_Vissim": [2],
                          "Count": [100], "Turn": ["NBT"], "IntersectionName": ["camera"]})
    monkeypatch.setattr(stage, "MatchupTable", lambda path: SimpleNamespace(df=pd.DataFrame({
        "JunctionID_OpenDrive": ["J_A"], "Need calibration?": ["Y"]})))
    monkeypatch.setattr(stage, "build_turn_counts", lambda table, directory: counts)
    monkeypatch.setattr(stage, "read_links_csv", lambda path: {})
    monkeypatch.setattr(stage.pd, "read_csv", lambda path: pd.DataFrame())
    problem = calibrate.Problem([calibrate.InflowKnob(1, 1, "J_A", 1)],
                               counted={(1, 2): 100}, max_inflow=400)
    monkeypatch.setattr(stage, "build_problem", lambda *args: (problem, []))
    monkeypatch.setattr(stage, "counting_sites", lambda *args: ({(1, 2): [2]}, []))
    monkeypatch.setattr(stage, "draw_charts", lambda *args: [])
    arguments = [part for name, path in paths.items() for part in (f"--{name}", str(path))]
    config = tmp_path / "effective.json"
    config.write_text(json.dumps({"Calibration": {
        "scenario_config": {"sim_start_time": 28800, "sim_end_time": 29700,
                            "calibration_time_step": 0.25,
                            "calibration_target": {"GEH": 3, "GEHPercent": 0.9}},
        "turn_inflow": {"max_inflow": 400, "seed": 7, "optimizer_seed": 123},
        "sa_config": {"epoch": 1, "pop_size": 2, "temp_init": 10, "step_size": 0.05},
    }}))
    return [*arguments, "--config", str(config), "--algo", "sa"], paths


@pytest.mark.parametrize("algorithm", ["ga", "sa", "ts", "bo"])
def test_cli_runs_real_optimizer_with_updated_config(cli_inputs, fake_session, tmp_path, algorithm):
    arguments, paths = cli_inputs
    arguments[arguments.index("--algo") + 1] = algorithm
    config_path = Path(arguments[arguments.index("--config") + 1])
    config = json.loads(config_path.read_text())
    config["Calibration"].update({
        "ga_config": {"epoch": 1, "pop_size": 10},
        "ts_config": {"epoch": 1, "pop_size": 2, "neighbour_size": 2, "tabu_size": 2},
        "bo_config": {"max_evaluations": 3, "random_points": 8,
                      "total_run": 1, "seed": 321, "target": 0, "tolerance": 0},
    })
    config_path.write_text(json.dumps(config))
    output = tmp_path / "calibrated" / "result.inpx"
    assert stage.main([*arguments, "--out", str(output)]) == 0
    summary = json.loads(output.with_name("result_summary.json").read_text())
    if algorithm == "bo":
        assert summary["settings"]["max_evaluations"] == 3
        exports = output.with_name("result_bo_result")
        assert (exports / "vissim_bayesopt_1runs_points.csv").exists()
        assert (exports / "best_run_fitness_convergence.png").exists()
    else:
        assert summary["settings"]["epoch"] == 1
    if algorithm == "sa":
        assert summary["settings"]["step_size"] == 0.05
    assert summary["algorithm"] == algorithm
    assert summary["period"] == [28800, 29700]
    assert summary["optimiser_seed"] == (321 if algorithm == "bo" else 123)
    assert summary["vissim_seed"] == 7
    assert summary["resolution"] == 4
    assert summary["max_inflow"] == 400
    assert summary["calibration_target"] == {"GEH": 3, "GEHPercent": 0.9}
    assert summary["evaluations"] >= 3
    assert paths["inpx"].read_text() == "<network/>"
    assert fake_session.instances[0].events[-1] == "stop"
    assert fake_session.instances[0].volume == pytest.approx(summary["solution"][0])


def test_cli_defaults_to_ga(cli_inputs, fake_session, tmp_path):
    arguments, _ = cli_inputs
    del arguments[-2:]
    output = tmp_path / "default.inpx"
    assert stage.main([*arguments, "--epoch", "1", "--out", str(output)]) == 0
    summary = json.loads(output.with_name("default_summary.json").read_text())
    assert summary["algorithm"] == "ga"


def test_turn_inflow_step_overrides_shared_scenario_step(cli_inputs, fake_session, tmp_path):
    arguments, _ = cli_inputs
    config_path = Path(arguments[arguments.index("--config") + 1])
    config = json.loads(config_path.read_text())
    config["Calibration"]["scenario_config"]["calibration_time_step"] = 1.0
    config["Calibration"]["turn_inflow"]["calibration_time_step"] = 0.1
    config_path.write_text(json.dumps(config))
    output = tmp_path / "result.inpx"

    assert stage.main([*arguments, "--out", str(output)]) == 0

    summary = json.loads(output.with_name("result_summary.json").read_text())
    assert summary["resolution"] == 10


def test_invalid_bo_settings_fail_before_com(cli_inputs, fake_session):
    arguments, _ = cli_inputs
    arguments[arguments.index("--algo") + 1] = "bo"
    path = Path(arguments[arguments.index("--config") + 1])
    config = json.loads(path.read_text())
    config["Calibration"]["bo_config"] = {"random_points": 2, "max_evaluations": 3}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="random_points"):
        stage.main(arguments)
    assert not fake_session.instances


@pytest.mark.parametrize("budget, successful", [(8, True), (1, False)])
def test_bo_budget_counts_independent_runs(cli_inputs, fake_session, budget, successful):
    arguments, _ = cli_inputs
    arguments[arguments.index("--algo") + 1] = "bo"
    path = Path(arguments[arguments.index("--config") + 1])
    config = json.loads(path.read_text())
    config["Calibration"]["bo_config"] = {"random_points": 8, "total_run": 2}
    path.write_text(json.dumps(config))
    options = [*arguments, "--budget", str(budget), "--dry-run"]
    if successful:
        assert stage.main(options) == 0
    else:
        with pytest.raises(SystemExit):
            stage.main(options)
    assert not fake_session.instances


def test_cli_dry_run_needs_no_simulator(cli_inputs, fake_session):
    arguments, _ = cli_inputs
    assert stage.main([*arguments, "--dry-run"]) == 0
    assert not fake_session.instances


def test_cli_rejects_overwriting_input(cli_inputs, fake_session):
    arguments, paths = cli_inputs
    with pytest.raises(SystemExit):
        stage.main([*arguments, "--out", str(paths["inpx"])])
    assert not fake_session.instances


def test_cli_rejects_empty_counts(cli_inputs, fake_session):
    arguments, _ = cli_inputs
    assert stage.main([*arguments, "--start", "09:00", "--end", "10:00"]) == 1
    assert not fake_session.instances


def test_cli_rejects_partial_count_window(cli_inputs, fake_session):
    arguments, _ = cli_inputs
    assert stage.main([*arguments, "--end", "08:20"]) == 1
    assert not fake_session.instances


def test_cli_rejects_gap_in_count_intervals(cli_inputs, fake_session, monkeypatch):
    arguments, _ = cli_inputs
    counts = pd.DataFrame({"IntervalStart": [28800, 30600], "IntervalEnd": [29700, 31500],
                           "FromLinkNo_Vissim": [1, 1], "ToLinkNo_Vissim": [2, 2],
                           "Count": [100, 100]})
    monkeypatch.setattr(stage, "build_turn_counts", lambda table, directory: counts)
    assert stage.main([*arguments, "--end", "08:45"]) == 1
    assert not fake_session.instances


@pytest.fixture
def behavior_inputs(tmp_path):
    network = tmp_path / "behavior.inpx"
    network.write_text('''<network><drivingBehaviors>
    <drivingBehavior no="1" carFollowModType="WIEDEMANN74"/>
    </drivingBehaviors><links><link no="1"/><link no="2"/></links></network>''')
    calibration = {
        "scenario_config": {"sim_start_time": 28800, "sim_end_time": 29700,
                            "calibration_time_step": 0.25},
        "behavior": {"is_calibration": True, "params_ranges": {"W74ax": [1, 3]},
                     "driving_behavior_ids": [1], "seed": 7, "optimizer_seed": 123,
                     "sel_behavior_routes": [["route", 1, 2, 35]]},
        "ga_config": {"epoch": 1, "pop_size": 10},
        "sa_config": {"epoch": 1, "pop_size": 2, "temp_init": 10, "step_size": 0.05},
        "ts_config": {"epoch": 1, "pop_size": 2, "neighbour_size": 2, "tabu_size": 2},
        "bo_config": {"max_evaluations": 3, "random_points": 8, "total_run": 1,
                      "seed": 321, "target": 0, "tolerance": 0},
    }
    config = tmp_path / "behavior.json"
    config.write_text(json.dumps({"Calibration": calibration}))
    output = tmp_path / "saved.inpx"
    return network, calibration, config, output


@pytest.fixture
def behavior_session(fake_session, monkeypatch):
    original_init = fake_session.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.behavior_values = {"W74ax": 2}
        self.missing_routes = False

        def set_parameter(name, value):
            assert not self.running
            self.behavior_values[name] = value
            self.events.append("behavior")

        def add_measure(no, first, start_position, last, end_position):
            assert first.no == 1 and last.no == 2
            self.markers = (start_position, end_position)

            def read(name):
                if name == "Dist":
                    return 200
                if name == "Vehs(Current,Last,All)":
                    return 0 if self.missing_routes else 10
                assert name == "TravTm(Current,Last,All)"
                return None if self.missing_routes else 20 + self.behavior_values["W74ax"] * 10

            return SimpleNamespace(SetAttValue=lambda name, value: None, AttValue=read)

        self.net.DrivingBehaviors = SimpleNamespace(ItemByKey=lambda no: SimpleNamespace(
            SetAttValue=set_parameter))
        self.net.Links = SimpleNamespace(ItemByKey=lambda no: SimpleNamespace(
            no=no, AttValue=lambda name: 100))
        self.net.VehicleTravelTimeMeasurements = SimpleNamespace(
            GetAll=lambda: [], AddVehicleTravelTimeMeasurement=add_measure)

    monkeypatch.setattr(fake_session, "__init__", initialize)
    monkeypatch.setattr(fake_session, "save_net_as", lambda self, path: path.write_text(
        json.dumps(self.behavior_values)))
    return fake_session


@pytest.mark.parametrize("algorithm", ["ga", "sa", "ts", "bo"])
def test_behavior_uses_shared_optimizer_and_native_parameters(behavior_inputs, behavior_session, algorithm):
    network, calibration, config, output = behavior_inputs
    source = network.read_bytes()
    assert stage.main(["--stage", "behavior", "--inpx", str(network), "--config", str(config),
                       "--out", str(output), "--algo", algorithm]) == 0
    summary = json.loads(output.with_name("saved_summary.json").read_text())
    assert summary["algorithm"] == algorithm
    assert summary["period"] == [28800, 29700]
    assert summary["resolution"] == 4
    assert summary["vissim_seed"] == 7
    assert summary["optimiser_seed"] == (321 if algorithm == "bo" else 123)
    assert summary["settings"] == {**stage.config_defaults({"Calibration": calibration})[algorithm]}
    assert summary["after"]["missing_routes"] == 0
    assert summary["after"]["mae_seconds"] <= summary["before"]["mae_seconds"]
    assert json.loads(output.read_text()) == summary["values"]
    assert network.read_bytes() == source
    assert behavior_session.instances[0].events[-1] == "stop"
    assert behavior_session.instances[0].vissim.Evaluation.attributes["VehTravTmsInterval"] == 900
    assert summary["route_positions"] == {"route": {"start": 0, "end": 100}}
    assert output.with_name("saved_routes.csv").exists()
    if algorithm == "bo":
        assert (output.with_name("saved_bo_result") / "vissim_bayesopt_1runs_points.csv").exists()


def test_behavior_default_ga_and_dry_run(behavior_inputs, behavior_session):
    network, _, config, output = behavior_inputs
    args = ["--stage", "behavior", "--inpx", str(network), "--config", str(config),
            "--out", str(output)]
    assert stage.main([*args, "--dry-run"]) == 0
    assert not behavior_session.instances
    assert stage.main(args) == 0
    assert json.loads(output.with_name("saved_summary.json").read_text())["algorithm"] == "ga"


def test_behavior_cli_reads_native_settings_and_reference_provenance(behavior_inputs, behavior_session):
    network, calibration, config, output = behavior_inputs
    native = calibration["behavior"].copy()
    calibration["scenario_config"]["calibration_time_step"] = 1.0
    native["calibration_time_step"] = 0.1
    native["reference_source"] = {"kind": "simulated_baseline", "seed": 42}
    calibration["behavior"] = {"is_calibration": True,
                               "params_ranges": {"min_gap": [1, 3]}, "vissim": native}
    config.write_text(json.dumps({"Calibration": calibration}))

    assert stage.main(["--stage", "behavior", "--inpx", str(network), "--config", str(config),
                       "--out", str(output), "--algo", "bo"]) == 0

    summary = json.loads(output.with_name("saved_summary.json").read_text())
    assert summary["params_ranges"] == {"W74ax": [1, 3]}
    assert summary["reference_source"] == {"kind": "simulated_baseline", "seed": 42}
    assert summary["resolution"] == 10
    assert summary["after"]["missing_routes"] == 0


@pytest.mark.parametrize("updates, message", [
    ({"params_ranges": {"min_gap": [1, 3]}}, "Unsupported Vissim"),
    ({"params_ranges": {"W99cc1Distr": [1, 3]}}, "Unsupported Vissim"),
    ({"params_ranges": {"W99cc0": [1, 3]}}, "requires WIEDEMANN99"),
    ({"params_ranges": {"W74ax": [-1, 3]}}, "Invalid Vissim bounds"),
    ({"params_ranges": {"W74ax": [3, 1]}}, "Invalid Vissim bounds"),
    ({"driving_behavior_ids": [2]}, "absent"),
    ({"driving_behavior_ids": []}, "unique positive"),
    ({"sel_behavior_routes": []}, "requires sel_behavior_routes"),
    ({"sel_behavior_routes": [["route", 1, 3, 30]]}, "existing integer"),
    ({"sel_behavior_routes": [["route", 1, 2, 0]]}, "positive seconds"),
    ({"missing_route_penalty": float("nan")}, "finite positive"),
    ({"route_positions": {"route": {"end": -1}}}, "nonnegative metres"),
    ({"route_positions": {"unknown": {"end": 1}}}, "configured route names"),
])
def test_behavior_invalid_inputs_fail_before_com(behavior_inputs, behavior_session, updates, message):
    network, calibration, config, _ = behavior_inputs
    calibration["behavior"].update(updates)
    config.write_text(json.dumps({"Calibration": calibration}))
    with pytest.raises(ValueError, match=message):
        stage.main(["--stage", "behavior", "--inpx", str(network), "--config", str(config)])
    assert not behavior_session.instances


def test_behavior_uses_explicit_measurement_positions(behavior_inputs, behavior_session):
    network, calibration, config, output = behavior_inputs
    calibration["behavior"]["route_positions"] = {"route": {"start": 5, "end": 0.5}}
    config.write_text(json.dumps({"Calibration": calibration}))
    assert stage.main(["--stage", "behavior", "--inpx", str(network), "--config", str(config),
                       "--out", str(output), "--algo", "sa"]) == 0
    summary = json.loads(output.with_name("saved_summary.json").read_text())
    assert summary["route_positions"] == {"route": {"start": 5, "end": 0.5}}
    assert behavior_session.instances[0].markers == (5, 0.5)


def test_behavior_rejects_markers_outside_link(behavior_inputs, behavior_session):
    network, calibration, config, output = behavior_inputs
    calibration["behavior"]["route_positions"] = {"route": {"end": 101}}
    config.write_text(json.dumps({"Calibration": calibration}))
    with pytest.raises(ValueError, match="lie within their links"):
        stage.main(["--stage", "behavior", "--inpx", str(network), "--config", str(config),
                    "--out", str(output)])
    assert not output.exists()


def test_missing_route_has_penalty_and_does_not_save(behavior_inputs, behavior_session, monkeypatch):
    network, calibration, config, output = behavior_inputs
    info = cali_behavior.export_behavior_info({"VISSIM": {"behavior": calibration}}, network)
    session = behavior_session()
    session.missing_routes = True
    evaluator = cali_behavior.BehaviorEvaluator(session, info, start=28800, end=29700)
    score = evaluator.run([0.5])
    assert score["mae_seconds"] == 10035
    assert score["routes"][0]["modelled_seconds"] is None
    assert score["missing_routes"] == 1
    original_run = cali_behavior.BehaviorEvaluator.run

    def missing_run(self, solution=None):
        self.session.missing_routes = True
        return original_run(self, solution)

    monkeypatch.setattr(cali_behavior.BehaviorEvaluator, "run", missing_run)
    with pytest.raises(RuntimeError, match="without completed vehicles"):
        stage.main(["--stage", "behavior", "--inpx", str(network), "--config", str(config),
                    "--out", str(output), "--algo", "sa"])
    assert not output.exists()
