"""Regression coverage for disabling external SUMO signal control data."""

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from realtwin.func_lib._c_abstract_scenario import _abstractScenario as abstract
from realtwin.func_lib._c_abstract_scenario import rt_demand_generation as demand
from realtwin.func_lib._c_abstract_scenario.rt_matchup_table_generation import (
    generate_matchup_table,
)
from realtwin.func_lib._e_simulation._generate_simulation import SimPrep
from realtwin.func_lib._e_simulation._sumo import SUMOPrep
from realtwin.rt_sumo import rt_sumo
from realtwin.util_lib.check_abstract_scenario_inputs import check_abstract_inputs


@pytest.fixture
def control_inputs(tmp_path: Path) -> dict:
    """Create demand inputs with stale references to a missing signal file."""
    traffic_dir = tmp_path / "Traffic"
    traffic_dir.mkdir()
    (traffic_dir / "demand.csv").write_text("Time,NBT\n08:00,10\n", encoding="utf-8")
    matchup = pd.DataFrame(
        [
            {
                "JunctionID_OpenDrive": "junction",
                "Bearing": "0",
                "Numbering": "0",
                "FromRoadID_OpenDrive": "1",
                "ToRoadID_OpenDrive": "2",
                "Turn": "thru",
                "File_GridSmart": None,
                "Date_GridSmart": None,
                "IntersectionName_GridSmart": None,
                "Turn_GridSmart": None,
                "File_Synchro": "missing.csv",
                "IntersectionID_Synchro": "001",
                "Turn_Synchro": "NBT",
                "Need calibration?": "N",
            }
        ]
    )
    generate_matchup_table(matchup, str(tmp_path / "MatchupTable.xlsx"))
    return {
        "input_dir": str(tmp_path),
        "output_dir": str(tmp_path / "output"),
        "demo_data": False,
        "Network": {"NetworkName": "test"},
        "SUMO": {"is_control_data": False},
        "Control": {"Signal": "Control/missing.csv"},
    }


@pytest.mark.parametrize("control_as_string", [False, True])
@pytest.mark.parametrize("provided_signal", [None, {"Phases": "ignored"}])
def test_abstract_scenario_skips_disabled_control(
    control_inputs, monkeypatch, control_as_string, provided_signal
):
    if control_as_string:
        control_inputs["Control"] = "missing.csv"
    scenario = abstract.AbstractScenario(control_inputs)
    scenario.Control.Signal = {"Phases": "previously loaded"}
    signal_loader = Mock(side_effect=AssertionError("Disabled signals were loaded"))
    monkeypatch.setattr(abstract, "load_control_signal", signal_loader)
    volume = pd.DataFrame(
        {"IntersectionName": ["junction"], "Time": ["08:00"], "NBT": [10]}
    )

    scenario.update_AbstractScenario_from_input(
        df_volume=volume, signal_dict=provided_signal
    )

    signal_loader.assert_not_called()
    assert scenario.Control.Signal == {}
    assert scenario.Traffic.Volume["Count"].tolist() == [10]


@pytest.mark.parametrize("empty_control_dir", [False, True])
def test_disabled_control_inputs_are_optional(control_inputs, empty_control_dir):
    if empty_control_dir:
        (Path(control_inputs["input_dir"]) / "Control").mkdir()

    assert check_abstract_inputs(control_inputs["input_dir"], is_control_data=False)
    with pytest.raises(Exception, match="Control folder"):
        check_abstract_inputs(control_inputs["input_dir"])


@pytest.mark.parametrize("missing_input", ["demand", "matchup"])
def test_disabled_control_still_requires_demand_inputs(control_inputs, missing_input):
    input_dir = Path(control_inputs["input_dir"])
    if missing_input == "demand":
        (input_dir / "Traffic/demand.csv").unlink()
        expected_error = "Traffic folder is empty"
    else:
        (input_dir / "MatchupTable.xlsx").unlink()
        expected_error = "Matchup table does not exist"

    with pytest.raises(Exception, match=expected_error):
        check_abstract_inputs(str(input_dir), is_control_data=False)


def test_matchup_update_skips_disabled_synchro(control_inputs, monkeypatch):
    signal_loader = Mock(side_effect=AssertionError("Synchro data was read"))
    monkeypatch.setattr(demand, "process_signal_from_utdf", signal_loader)
    input_dir = Path(control_inputs["input_dir"])

    updated = demand.update_matchup_table(
        str(input_dir / "MatchupTable.xlsx"),
        control_dir=str(input_dir / "Control"),
        traffic_dir=str(input_dir / "Traffic"),
        is_control_data=False,
    )

    signal_loader.assert_not_called()
    assert updated["JunctionID_OpenDrive"].tolist() == ["junction"]
    assert updated["Turn_Synchro"].tolist() == ["NBT"]
    assert updated["Need calibration?"].tolist() == ["Y"]
    saved = pd.read_excel(input_dir / "MatchupTable.xlsx", skiprows=1, dtype=str)
    assert saved["FromRoadID_OpenDrive"].tolist() == ["1"]


@pytest.mark.parametrize("sumo_config", [{}, {"is_control_data": True}])
def test_matchup_update_keeps_enabled_signal_loading(
    control_inputs, monkeypatch, sumo_config
):
    control_inputs["SUMO"] = sumo_config
    signal_loader = Mock(side_effect=FileNotFoundError("Missing enabled signal"))
    monkeypatch.setattr(demand, "process_signal_from_utdf", signal_loader)
    input_dir = Path(control_inputs["input_dir"])

    with pytest.raises(FileNotFoundError, match="Missing enabled signal"):
        demand.update_matchup_table(
            str(input_dir / "MatchupTable.xlsx"),
            control_dir=str(input_dir / "Control"),
            is_control_data=sumo_config.get("is_control_data", True),
        )

    signal_loader.assert_called_once()


@pytest.mark.parametrize("sumo_config", [None, {}, {"is_control_data": True}])
def test_sim_prep_keeps_signal_loading_enabled_by_default(sumo_config):
    input_config = {} if sumo_config is None else {"SUMO": sumo_config}
    scenario = SimpleNamespace(input_config=input_config)
    preparation = SimPrep()
    preparation.SUMOSim = Mock()

    preparation.create_sumo_sim(scenario, seed=101)

    preparation.SUMOSim.importSignal.assert_called_once_with(scenario)


def test_direct_signal_import_skips_disabled_control_without_files():
    scenario = SimpleNamespace(input_config={"SUMO": {"is_control_data": False}})

    assert SUMOPrep().importSignal(scenario) is None


@pytest.mark.parametrize("existing_signals", [False, True])
@pytest.mark.parametrize("control_folder", ["missing", "empty", "populated"])
def test_disabled_control_flows_through_scenarios_to_network(
    control_inputs, monkeypatch, existing_signals, control_folder
):
    """Exercise the scenario handoff and real network/config writers."""
    control_dir = Path(control_inputs["input_dir"]) / "Control"
    if control_folder != "missing":
        control_dir.mkdir()
    if control_folder == "populated":
        (control_dir / "missing.csv").write_text("invalid signal data", encoding="utf-8")
    volume = pd.DataFrame(
        {"IntersectionName": ["junction"], "Time": ["08:00"], "NBT": [10]}
    )
    lookup = pd.DataFrame(
        {
            "IntersectionName": ["junction"],
            "Turn": ["NBT"],
            "OpenDriveFromID": ["1"],
            "OpenDriveToID": ["2"],
        }
    )
    generate_demand = Mock(return_value=(volume, lookup))
    signal_ids = Mock(side_effect=AssertionError("Signal IDs were reassigned"))
    signal_loader = Mock(side_effect=AssertionError("Signals were imported"))
    monkeypatch.setattr(rt_sumo, "load_input_configs", Mock(return_value=control_inputs))
    monkeypatch.setattr(rt_sumo, "generate_turn_demand", generate_demand)
    monkeypatch.setattr(rt_sumo, "parse_SUMO_TLS_ID", signal_ids)
    monkeypatch.setattr(SUMOPrep, "importSignal", signal_loader)
    monkeypatch.setattr(SUMOPrep, "importDemand", Mock())
    twin = rt_sumo.RealTwinSUMO("config.yaml", input_confirm=False)
    twin.abstract_scenario = abstract.AbstractScenario(control_inputs)
    source_net = Path(control_inputs["output_dir"]) / "OpenDrive/test.net.xml"
    signal_logic = (
        '<tlLogic id="junction" programID="original" type="static" offset="0">'
        '<phase duration="60" state="G"/></tlLogic>'
        if existing_signals
        else ""
    )
    source_net.write_text(
        f'<net><edge id="1"/><edge id="2"/>{signal_logic}</net>', encoding="utf-8"
    )
    original_network = source_net.read_bytes()
    twin.abstract_scenario.Network.OpenDriveNetwork.OpenDrive_network = [
        str(source_net),
        "",
    ]
    twin.sel_sim = ["sumo"]

    twin.generate_abstract_scenario()
    twin.generate_concrete_scenario()
    assert twin.prepare_simulation(seed=101)

    generate_demand.assert_called_once()
    signal_ids.assert_not_called()
    signal_loader.assert_not_called()
    assert twin.abstract_scenario.Control.Signal == {}
    assert twin.concrete_scenario.input_config is twin.input_config
    assert twin.concrete_scenario.TrafficControl.Signal == {}
    assert source_net.read_bytes() == original_network
    output_dir = Path(control_inputs["output_dir"]) / "SUMO"
    controllers = ET.parse(output_dir / "test.net.xml").getroot().findall("tlLogic")
    assert len(controllers) == int(existing_signals)
    if existing_signals:
        assert controllers[0].get("programID") == "original"
        assert controllers[0].find("phase").get("state") == "G"
    config = ET.parse(output_dir / "test.sumocfg").getroot()
    assert config.find("input/net-file").get("value") == "test.net.xml"
    assert config.find("random/seed").get("value") == "101"


@pytest.mark.parametrize("empty_control_dir", [False, True])
def test_generate_inputs_disables_missing_control_without_sumo_section(
    control_inputs, monkeypatch, empty_control_dir
):
    control_inputs.pop("SUMO")
    input_dir = Path(control_inputs["input_dir"])
    if empty_control_dir:
        (input_dir / "Control").mkdir()
    source_net = input_dir / "source.net.xml"
    source_net.write_text("<net/>", encoding="utf-8")
    monkeypatch.setattr(rt_sumo, "load_input_configs", Mock(return_value=control_inputs))
    monkeypatch.setattr(abstract.AbstractScenario, "create_OpenDrive_network", Mock())
    monkeypatch.setattr(
        rt_sumo, "format_junction_bearing", Mock(return_value=pd.DataFrame())
    )
    monkeypatch.setattr(rt_sumo, "generate_matchup_table", Mock())
    twin = rt_sumo.RealTwinSUMO("config.yaml", input_confirm=False)

    twin.generate_inputs(incl_sumo_net=str(source_net))

    assert twin.input_config["SUMO"]["is_control_data"] is False
    assert (Path(control_inputs["output_dir"]) / "OpenDrive/test.net.xml").exists()
