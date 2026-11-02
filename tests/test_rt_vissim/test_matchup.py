"""Regression tests for Vissim network identities and SUMO seed translation."""

import importlib

import pandas as pd
import pytest

from realtwin.rt_vissim import matchup
from realtwin.rt_vissim.matchup import MatchupTable, generate_matchup_table


@pytest.fixture
def movements():
    """Two native junctions with IDs different from their SUMO source."""
    return pd.DataFrame(
        {
            "JunctionID_OpenDrive": [1, 4],
            "Bearing": [0.0, 90.0],
            "Numbering": ["NB", "EB"],
            "FromLinkNo_Vissim": [100, 200],
            "ToLinkNo_Vissim": [101, 201],
            "Turn": ["thru", "right"],
            "InternalLinks_Vissim": ["300", "400"],
        }
    )


def test_blank_demand_table_keeps_native_link_numbers(tmp_path, movements):
    """A newly generated workbook needs no observations to be structurally valid."""
    path_table = generate_matchup_table(movements, tmp_path / "MatchupTable.xlsx")

    table = MatchupTable(path_table)

    assert table.link_numbers() == {100, 101, 200, 201}
    assert table.gridsmart_files() == {}


@pytest.mark.parametrize("control_enabled", [False, True])
def test_matchup_stage_can_skip_synchro_reads(
    monkeypatch, tmp_path, movements, control_enabled
):
    """Disabling control preserves signal references while still deriving demand."""
    stage = importlib.import_module("realtwin.rt_vissim.02_matchup_table")
    signal_module = importlib.import_module(
        "realtwin.func_lib._c_abstract_scenario.rt_demand_generation")
    path_movements = tmp_path / "network_movements.csv"
    movements.to_csv(path_movements, index=False)
    frame = movements.reindex(columns=matchup.ALL_COLUMNS).astype(object)
    frame.loc[0, ["File_GridSmart", "File_Synchro", "IntersectionID_Synchro", "Turn_Synchro"]] = [
        "counts.xlsx", "signals.csv", "7", "NBL"]
    path_table = tmp_path / "MatchupTable.xlsx"
    matchup.write_matchup_table(frame, path_table)
    (tmp_path / "counts.xlsx").touch()
    monkeypatch.setattr(matchup, "_gridsmart_info", lambda path: (
        "Intersection", "2026-10-12", {"NBT"}))
    monkeypatch.setattr(stage, "build_turn_counts", lambda *args: pd.DataFrame())
    monkeypatch.setattr(stage, "validate", lambda *args: [])
    signal_reads = []

    def read_signal(path):
        signal_reads.append(path)
        return {}

    monkeypatch.setattr(signal_module, "process_signal_from_utdf", read_signal)
    arguments = ["--movements", str(path_movements), "--outdir", str(tmp_path),
                 "--traffic-dir", str(tmp_path), "--control-dir", str(tmp_path)]
    if not control_enabled:
        arguments.append("--no-control")

    assert stage.main(arguments) == 0

    assert len(signal_reads) == int(control_enabled)
    table = MatchupTable(path_table)
    assert table.df.loc[0, "Turn_GridSmart"] == "NBT"
    assert table.df.loc[0, "IntersectionName_GridSmart"] == "Intersection"
    assert table.synchro_file() == "signals.csv"
    assert table.df.loc[0, "IntersectionID_Synchro"] == "7"
    assert table.df.loc[0, "Turn_Synchro"] == "NBL"


@pytest.mark.parametrize("missing_column", ["FromLinkNo_Vissim", "ToLinkNo_Vissim"])
def test_matchup_rejects_missing_native_link_numbers(tmp_path, movements, missing_column):
    """Relabeling SUMO workbook headers cannot supply native Vissim identities."""
    movements.loc[0, missing_column] = None
    path_table = generate_matchup_table(movements, tmp_path / "MatchupTable.xlsx")

    with pytest.raises(ValueError, match="without FromLinkNo_Vissim or ToLinkNo_Vissim"):
        MatchupTable(path_table)


@pytest.mark.parametrize("missing_links", [False, True])
def test_matchup_stage_rejects_stale_table_without_changing_it(
    tmp_path, movements, missing_links, capsys
):
    """A network replacement must not overwrite a workbook from the old network."""
    stage = importlib.import_module("realtwin.rt_vissim.02_matchup_table")
    path_movements = tmp_path / "network_movements.csv"
    movements.to_csv(path_movements, index=False)
    stale_movements = movements.copy()
    stale_movements.loc[0, "FromLinkNo_Vissim"] = None if missing_links else 999
    path_table = generate_matchup_table(stale_movements, tmp_path / "MatchupTable.xlsx")
    original_bytes = path_table.read_bytes()

    result = stage.main(["--movements", str(path_movements), "--outdir", str(tmp_path)])

    assert result == 1
    assert "--regenerate" in capsys.readouterr().out
    assert path_table.read_bytes() == original_bytes


def test_seed_from_sumo_translates_junctions_and_preserves_native_links(tmp_path, movements):
    """SUMO junction 2's observations belong to native junction 4, not native 2."""
    stage = importlib.import_module("realtwin.rt_vissim.02_matchup_table")
    path_table = generate_matchup_table(movements, tmp_path / "MatchupTable.xlsx")
    path_sumo = tmp_path / "sumo.xlsx"
    pd.DataFrame(
        {
            "JunctionID_OpenDrive": [2, 6],
            "File_GridSmart": ["Gunbarrel.xls", None],
            "IntersectionID_Synchro": [16, None],
            "File_Synchro": ["signals.csv", None],
        }
    ).to_excel(path_sumo, startrow=1, index=False)
    path_junctions = tmp_path / "network_junctions.csv"
    pd.DataFrame(
        {"JunctionID_OpenDrive": [1, 4], "Name_OpenDrive": [6, 2]}
    ).to_csv(path_junctions, index=False)

    assert stage.seed_from_sumo(path_table, path_sumo, path_junctions) == 1
    table = MatchupTable(path_table)

    assert table.gridsmart_files() == {"4": "Gunbarrel.xls"}
    gunbarrel = table.df[table.df["JunctionID_OpenDrive"] == "4"]
    assert gunbarrel["IntersectionID_Synchro"].iloc[0] == "16"
    assert table.link_numbers() == {100, 101, 200, 201}
