"""Optional Aimsun control data regressions without a native Aimsun licence."""

import builtins
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from realtwin.rt_aimsun import rt_aimsun as aimsun


@pytest.fixture
def twin(monkeypatch, tmp_path):
    """Keep the model and configured input folder distinct to check data lookup."""
    model_path = tmp_path / "models" / "network.ang"
    model_path.parent.mkdir()
    model_path.touch()
    configuration = {
        "input_dir": str(tmp_path / "inputs"),
        "AIMSUN": {
            "model_fname": str(model_path),
            "exe_path": "aconsole.exe",
            "site_packages": sys.path[0],
        },
    }
    monkeypatch.setattr(aimsun, "load_input_configs", lambda path: configuration)
    model = aimsun.RealTwinAimsun("config.yaml", input_confirm=False)
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(communicate=lambda: (
            "Matchup table saved to: MatchupTable.xlsx\n", None))

    monkeypatch.setattr(aimsun.subprocess, "Popen", launch)
    return model, commands


@pytest.mark.parametrize("section", ["missing", None, {}, {"is_control_data": False}])
def test_aimsun_initializes_optional_control_configuration(monkeypatch, section):
    expected_flag = not (isinstance(section, dict) and section.get("is_control_data") is False)
    configuration = {} if section == "missing" else {"AIMSUN": section}
    monkeypatch.setattr(aimsun, "load_input_configs", lambda path: configuration)

    model = aimsun.RealTwinAimsun("config.yaml", input_confirm=False)

    assert model.input_config["AIMSUN"]["is_control_data"] is expected_flag
    assert model.input_config["AIMSUN"]["aimsun_file"] == model.aimsun_file


@pytest.mark.parametrize(
    "contents, flag, enabled",
    [("missing", None, False), ("empty", None, False), ("notes", None, False),
     ("directory", None, False), ("signal", None, True), ("signal", False, False)],
)
def test_generate_inputs_detects_control_beside_model(twin, contents, flag, enabled):
    model, commands = twin
    # A CSV in input_dir must not enable control read from the model directory.
    input_control = Path(model.input_config["input_dir"]) / "Control"
    input_control.mkdir(parents=True)
    (input_control / "signals.csv").write_text("signal data", encoding="utf-8")
    control = Path(model.input_config["AIMSUN"]["model_fname"]).parent / "Control"
    if contents != "missing":
        control.mkdir()
    if contents == "signal":
        (control / "signals.CSV").write_text("signal data", encoding="utf-8")
    elif contents == "notes":
        (control / "README.txt").write_text("notes", encoding="utf-8")
    elif contents == "directory":
        (control / "signals.csv").mkdir()
    if flag is not None:
        model.input_config["AIMSUN"]["is_control_data"] = flag

    model.generate_inputs()

    assert control.is_dir()
    assert model.input_config["AIMSUN"]["is_control_data"] is enabled
    assert [command[2] for command in commands] == [
        model.aimsun_file["step0.1"], model.aimsun_file["step1"]]
    assert json.loads(commands[0][-1])["AIMSUN"]["is_control_data"] is enabled


def test_control_loading_can_be_reenabled_after_adding_data(twin):
    model, commands = twin
    model.generate_inputs()
    control = Path(model.input_config["AIMSUN"]["model_fname"]).parent / "Control"
    (control / "signals.csv").write_text("signal data", encoding="utf-8")

    model.generate_abstract_scenario()
    assert json.loads(commands[-1][-1])["AIMSUN"]["is_control_data"] is False
    model.input_config["AIMSUN"]["is_control_data"] = True
    model.generate_abstract_scenario()

    assert json.loads(commands[-1][-1])["AIMSUN"]["is_control_data"] is True


@pytest.mark.parametrize("method", ["generate_abstract_scenario", "generate_concrete_scenario"])
@pytest.mark.parametrize("flag", [None, True, False])
@pytest.mark.parametrize("has_signal_data", [False, True])
def test_scenario_generation_honors_optional_control(twin, method, flag, has_signal_data):
    model, commands = twin
    if flag is None:
        model.input_config["AIMSUN"].pop("is_control_data")
    else:
        model.input_config["AIMSUN"]["is_control_data"] = flag
    if has_signal_data:
        control = Path(model.input_config["AIMSUN"]["model_fname"]).parent / "Control"
        control.mkdir()
        (control / "signals.csv").write_text("signal data", encoding="utf-8")
    enabled = has_signal_data and flag is not False

    getattr(model, method)()

    stages = ["step2"] if method == "generate_abstract_scenario" else ["step3"]
    if method == "generate_concrete_scenario" and enabled:
        stages.extend(["step4.1", "step4.2", "step4.3"])
    assert [command[2] for command in commands] == [model.aimsun_file[stage] for stage in stages]
    for command in commands:
        assert json.loads(command[-1])["AIMSUN"]["is_control_data"] is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_prepare_simulation_passes_control_flag_to_scenario_stage(twin, enabled):
    model, commands = twin
    model.input_config["AIMSUN"]["is_control_data"] = enabled

    model.prepare_simulation()

    assert [command[2] for command in commands] == [
        model.aimsun_file["step5.1"], model.aimsun_file["step5.2"]]
    assert json.loads(commands[0][-1])["AIMSUN"]["is_control_data"] is enabled
    assert all(command[3] == model.input_config["AIMSUN"]["model_fname"] for command in commands)


def test_calibration_keeps_disabled_control_and_model(twin, monkeypatch):
    model, commands = twin
    model.input_config["AIMSUN"]["is_control_data"] = False
    model.input_config["Calibration"] = {
        "turn_inflow": {"is_calibration": True},
        "behavior": {"is_calibration": False},
    }
    calibrate = Mock(return_value=True)
    monkeypatch.setattr(aimsun, "cali_aimsun", calibrate)

    assert model.calibrate() is True

    configuration = calibrate.call_args.kwargs["input_config"]
    assert configuration["AIMSUN"]["is_control_data"] is False
    assert configuration["AIMSUN"]["model_fname"] == model.input_config["AIMSUN"]["model_fname"]
    assert commands == []


@pytest.fixture
def load_stage(monkeypatch):
    """Load native script code with empty PyANG modules; tests supply used APIs."""
    for name in ("PyANGBasic", "PyANGKernel", "PyANGConsole"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    def load(filename):
        path = Path(aimsun.__file__).parent / filename
        specification = importlib.util.spec_from_file_location("aimsun_control_stage", path)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module

    return load


@pytest.mark.parametrize(
    "flag, has_signal_data", [(False, False), (False, True), (True, True), (None, True)]
)
def test_matchup_stage_skips_optional_control_and_preserves_demand(
    load_stage, monkeypatch, tmp_path, flag, has_signal_data
):
    stage = load_stage("Step2_MatchupTableDataInput.py")
    model_path = tmp_path / "network.ang"
    native_model = SimpleNamespace(getDocumentFileName=lambda: str(model_path))
    native_console = SimpleNamespace(open=lambda path: True, getModel=lambda: native_model)
    monkeypatch.setattr(stage, "ANGConsole", lambda: native_console, raising=False)
    traffic = tmp_path / "Traffic"
    traffic.mkdir()
    pd.DataFrame([
        ["Intersection", "Intersection"], ["Date", "2026-10-12"],
        [None, "Through"], ["Northbound", "10"],
    ]).to_excel(traffic / "counts.xlsx", index=False, header=False)
    matchup_path = tmp_path / "MatchupTable.xlsx"
    pd.DataFrame([{
        "JunctionID_Aimsun": "1", "Bearing": "0", "Numbering": "NB",
        "FromRoadID_Aimsun": "10", "ToRoadID_Aimsun": "20", "Turn": "thru",
        "File_GridSmart": "counts.xlsx", "Date_GridSmart": None,
        "IntersectionName_GridSmart": None, "Turn_GridSmart": None,
        "File_Synchro": "signals.csv", "IntersectionID_Synchro": "7",
        "Turn_Synchro": "NBL", "Need calibration?": "Y",
    }]).to_excel(matchup_path, startrow=1, index=False)
    original_table = matchup_path.read_bytes()
    if has_signal_data:
        control = tmp_path / "Control"
        control.mkdir()
        (control / "signals.csv").write_text(
            "[Lanes]\nskipped\nRECORDNAME,INTID,NBT\nLanes,7,1\n", encoding="utf-8")
    configuration = {"AIMSUN": {"site_packages": sys.path[0]}}
    if flag is not None:
        configuration["AIMSUN"]["is_control_data"] = flag
    signal_reads = []
    original_open = builtins.open

    def open_file(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file).parent.name == "Control":
            signal_reads.append(file)
            if flag is False:
                pytest.fail("Disabled Synchro data was read")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_file)

    assert stage.main(["stage.py", str(model_path), json.dumps(configuration)]) == 0

    frame = pd.read_excel(matchup_path, skiprows=1, dtype=str)
    assert frame.loc[0, "Turn_GridSmart"] == "NBT"
    assert frame.loc[0, "IntersectionName_GridSmart"] == "Intersection"
    assert frame.loc[0, "Need calibration?"] == "N"
    assert frame.loc[0, "File_Synchro"] == "signals.csv"
    assert frame.loc[0, "Turn_Synchro"] == ("NBL" if flag is False else "NBT")
    assert bool(signal_reads) is (flag is not False)
    assert Path(str(matchup_path) + ".bak").read_bytes() == original_table


@pytest.mark.parametrize("flag", [False, True, None, "legacy"])
@pytest.mark.parametrize("has_master", [False, True])
def test_scenario_stage_makes_master_plan_optional(
    load_stage, monkeypatch, tmp_path, flag, has_master, capsys
):
    """Run scenario generation, including its native object-creation path."""
    stage = load_stage("Step5.1_ScenarioGeneration.py")
    demand = SimpleNamespace(getName=lambda: "Traffic Demand", getId=lambda: 1)
    master = SimpleNamespace(getName=lambda: "RealTwin Master Control", getId=lambda: 2)
    catalog = {"GKTrafficDemand": {1: demand},
               "GKMasterControlPlan": {2: master} if has_master else {}}
    folder = Mock()
    commander = Mock()
    native_model = SimpleNamespace(
        getType=lambda name: name,
        getCatalog=lambda: SimpleNamespace(getObjectsByType=lambda name: catalog.get(name, {})),
        getCreateRootFolder=lambda: SimpleNamespace(findFolder=lambda name: folder),
        getCommander=lambda: commander,
    )
    native_console = Mock()
    native_console.open.return_value = True
    native_console.getModel.return_value = native_model
    monkeypatch.setattr(stage, "ANGConsole", lambda: native_console, raising=False)
    objects = {name: Mock() for name in ("GKScenario", "GKExperiment", "GKReplication")}
    system = SimpleNamespace(newObject=Mock(side_effect=lambda name, model: objects[name]))
    monkeypatch.setattr(stage, "GKSystem", SimpleNamespace(getSystem=lambda: system), raising=False)
    monkeypatch.setattr(stage, "GKExperiment", SimpleNamespace(eMicro=1, eOneShot=2), raising=False)
    monkeypatch.setattr(stage, "GKObject", SimpleNamespace(eModified=1), raising=False)
    model_path = str(tmp_path / "network.ang")
    arguments = ["stage.py", model_path]
    if flag != "legacy":
        configuration = {"AIMSUN": {}}
        if flag is not None:
            configuration["AIMSUN"]["is_control_data"] = flag
        arguments.append(json.dumps(configuration))

    stage.main(arguments)

    if flag is not False and not has_master:
        assert "master control plan" in capsys.readouterr().out
        system.newObject.assert_not_called()
        native_console.save.assert_not_called()
        return
    assert [call.args[0] for call in system.newObject.call_args_list] == [
        "GKScenario", "GKExperiment", "GKReplication"]
    objects["GKScenario"].setDemand.assert_called_once_with(demand)
    if flag is False:
        objects["GKScenario"].setMasterControlPlan.assert_not_called()
    else:
        objects["GKScenario"].setMasterControlPlan.assert_called_once_with(master)
    native_console.save.assert_called_once_with(model_path)
    native_console.close.assert_called_once()
    assert catalog["GKMasterControlPlan"] == ({2: master} if has_master else {})


@pytest.mark.parametrize("missing_input", ["traffic", "matchup"])
def test_optional_control_still_requires_demand_inputs(
    load_stage, monkeypatch, tmp_path, missing_input, capsys
):
    stage = load_stage("Step2_MatchupTableDataInput.py")
    model_path = tmp_path / "network.ang"
    native_model = SimpleNamespace(getDocumentFileName=lambda: str(model_path))
    native_console = SimpleNamespace(open=lambda path: True, getModel=lambda: native_model)
    monkeypatch.setattr(stage, "ANGConsole", lambda: native_console, raising=False)
    if missing_input != "traffic":
        (tmp_path / "Traffic").mkdir()
    if missing_input != "matchup":
        (tmp_path / "MatchupTable.xlsx").touch()
    configuration = {"AIMSUN": {"is_control_data": False, "site_packages": sys.path[0]}}

    assert stage.main(["stage.py", str(model_path), json.dumps(configuration)]) == 1

    message = capsys.readouterr().out
    assert ("traffic dir not found" if missing_input == "traffic" else "matchup table not found") in message
    assert not list(tmp_path.glob("*.bak*"))
