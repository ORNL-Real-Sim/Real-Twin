"""Regression coverage for SUMO signal import with pandas string dtypes."""

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from realtwin.func_lib._e_simulation import _sumo as sumo

pytestmark = pytest.mark.filterwarnings("error::pandas.errors.ChainedAssignmentError")


@pytest.fixture
def signal_scenario(tmp_path: Path) -> SimpleNamespace:
    """Write a three-movement junction with two active signal phases."""
    control_dir = tmp_path / "Control"
    control_dir.mkdir()
    network_dir = tmp_path / "output" / "SUMO"
    network_dir.mkdir(parents=True)
    (network_dir / "test.net.xml").write_text(
        '<net><edge id="-1"/><edge id="-2"/><edge id="-3"/><edge id="-4"/>'
        '<tlLogic id="junction" programID="old" type="static" offset="0">'
        '<phase duration="60" state="rrr"/></tlLogic>'
        '<connection from="-1" to="-2" tl="junction" linkIndex="0" dir="r"/>'
        '<connection from="-1" to="-3" tl="junction" linkIndex="1" dir="s"/>'
        '<connection from="-1" to="-4" tl="junction" linkIndex="2" dir="l"/>'
        "</net>",
        encoding="utf-8",
    )
    records = []
    for turn, to_id in zip(("NBR", "NBT", "NBL"), ("2", "3", "4")):
        records.append(
            {
                "JunctionID_OpenDrive": "junction",
                "FromRoadID_OpenDrive": "1",
                "ToRoadID_OpenDrive": to_id,
                "Turn_Synchro": turn,
                "File_GridSmart": None,
                "Date_GridSmart": None,
                "IntersectionName_GridSmart": None,
                "File_Synchro": "signals.csv",
                "IntersectionID_Synchro": "001",
                "Need calibration?": "N",
            }
        )
    pd.DataFrame(records).to_excel(
        tmp_path / "MatchupTable.xlsx", startrow=1, index=False
    )
    (control_dir / "signals.csv").write_text(
        "[Lanes]\nLane Group Data\n"
        "RECORDNAME,INTID,NBL,NBT,NBR\n"
        "Lanes,001,1,1,1\n"
        "Shared,001,0,0,0\n"
        "Phase1,001,2,1,2\n"
        "PermPhase1,001,,,\n"
        "Allow RTOR,001,0,0,1\n"
        "DetectSize1,001,20,50,50\n"
        "Description,001,left,through,right\n"
        "[Timeplans]\nTime Plan Data\n"
        "RECORDNAME,INTID,DATA\n"
        "Offset,001,0\n"
        "Cycle Length,001,60\n"
        "Control Type,001,0\n"
        "Referenced To,001,1\n"
        "[Phases]\nPhase Data\n"
        "RECORDNAME,INTID,D1,D2,D3\n"
        "BRP,001,111,211,212\n"
        "MinGreen,001,5,6,0\n"
        "MaxGreen,001,20,30,0\n"
        "InhibitMax,001,0,0,0\n"
        "Recall,001,1,3,0\n"
        "VehExt,001,2,2,0\n"
        "Yellow,001,3,3,0\n"
        "AllRed,001,1,1,0\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        input_config={
            "input_dir": str(tmp_path),
            "output_dir": str(tmp_path / "output"),
        },
        Supply=SimpleNamespace(NetworkName="test"),
    )


def test_signal_parser_preserves_text_ids_and_missing_values(signal_scenario):
    signal_path = (
        Path(signal_scenario.input_config["input_dir"]) / "Control/signals.csv"
    )
    tables = sumo.process_signal_data(str(signal_path))
    lanes = tables["Lanes"]

    assert set(tables) == {"Lanes", "Timeplans", "Phases"}
    assert lanes["INTID"].eq("001").all()
    assert lanes.loc[lanes["RECORDNAME"] == "Description", "NBT"].iloc[0] == "through"
    assert lanes.loc[lanes["RECORDNAME"] == "PermPhase1", "NBT"].isna().all()


@pytest.mark.parametrize("fixed_time", [False, True])
@pytest.mark.parametrize(
    ("shared", "expected_state", "expected_rtor"),
    [
        ("0", "sGr", "0"),
        ("1", "sGG", "0"),
        ("2", "GGr", ""),
        ("3", "GGG", ""),
        ("", "sGr", "0"),
    ],
)
def test_signal_import_shared_movements(
    signal_scenario, monkeypatch, fixed_time, shared, expected_state, expected_rtor
):
    signal_path = (
        Path(signal_scenario.input_config["input_dir"]) / "Control/signals.csv"
    )
    source = signal_path.read_text(encoding="utf-8")
    signal_path.write_text(
        source.replace("Shared,001,0,0,0", f"Shared,001,0,{shared},0"),
        encoding="utf-8",
    )
    state_inputs = []
    original_state_generator = sumo.generate_complete_state_string

    def record_state(protected, permitted, rtor, movement_count):
        state_inputs.append((protected, permitted, rtor, movement_count))
        return original_state_generator(protected, permitted, rtor, movement_count)

    monkeypatch.setattr(sumo, "generate_complete_state_string", record_state)
    sumo.SUMOPrep(FixedTime=fixed_time).importSignal(signal_scenario)

    network = Path(signal_scenario.input_config["output_dir"]) / "SUMO/test.net.xml"
    controller = ET.parse(network).getroot().find("tlLogic")
    assert controller.get("programID") == "NEMA"
    phases = controller.findall("phase")
    assert [phase.get("name") for phase in phases] == ["1", "2"]
    assert phases[0].get("state") == expected_state
    assert phases[1].get("state") == "GrG"
    assert float(phases[0].get("maxDur")) == 20.0
    assert float(phases[1].get("minDur")) == 6.0
    assert state_inputs[0][2] == expected_rtor
    params = {
        param.get("key"): param.get("value") for param in controller.findall("param")
    }
    assert params["minRecall"] == ("" if fixed_time else "1")
    assert params["maxRecall"] == ("1,2" if fixed_time else "2")
