"""Pandas compatibility regressions for traffic and calibration data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from realtwin.func_lib._c_abstract_scenario._abstractScenario import load_traffic_volume
from realtwin.func_lib._f_calibration.algo_sumo import (
    util_cali_behavior,
    util_cali_turn_inflow,
)


@pytest.mark.parametrize("as_csv", [False, True])
def test_traffic_counts_preserve_missing_identifiers(tmp_path: Path, as_csv: bool):
    traffic = pd.DataFrame(
        {
            "IntersectionName": ["junction", None],
            "Time": ["08:00", "08:15"],
            "NBT": ["10", None],
        }
    )
    if as_csv:
        path = tmp_path / "traffic.csv"
        traffic.to_csv(path, index=False)
        traffic = str(path)

    volume = load_traffic_volume(traffic)

    assert volume["Count"].tolist() == [10, 0]
    assert volume["IntervalStart"].tolist() == [28800, 29700]
    assert volume["IntervalEnd"].tolist() == [29700, 30600]
    assert volume["IntersectionName"].isna().tolist() == [False, True]


def test_traffic_counts_reject_invalid_numbers():
    traffic = pd.DataFrame(
        {
            "IntersectionName": ["junction"],
            "Time": ["08:00"],
            "NBT": ["invalid"],
        }
    )

    with pytest.raises(ValueError, match="invalid"):
        load_traffic_volume(traffic)


@pytest.mark.parametrize("numeric_dtype", [int, str])
def test_calibration_preserves_fractional_ratios_and_counts(numeric_dtype):
    turns = pd.DataFrame(
        {
            "OpenDriveFromID": ["1", "1"],
            "OpenDriveToID": ["2", "3"],
            "TurnRatio": pd.Series([0, 1], dtype=numeric_dtype),
        }
    )
    calibration = pd.DataFrame(
        {
            "JunctionID_OpenDrive": ["junction", "junction"],
            "Numbering": [0, 0],
            "Calibration variable?": [1, 0],
            "OpenDriveFromID": ["1", "1"],
            "OpenDriveToID": ["2", "3"],
        }
    )
    inflows = pd.DataFrame(
        {
            "OpenDriveFromID": ["1"],
            "Count": pd.Series([10], dtype=numeric_dtype),
        }
    )

    updated_turns, updated_inflows = (
        util_cali_turn_inflow.update_turn_inflow_from_solution(
            np.array([0.25, 101]), turns, calibration, inflows, ["1"], 3600, 900
        )
    )

    assert updated_turns is turns
    assert updated_inflows is inflows
    assert turns["TurnRatio"].tolist() == pytest.approx([0.25, 0.75])
    assert inflows["Count"].tolist() == pytest.approx([25.25])


def test_behavior_calibration_with_integer_excel_columns(tmp_path: Path):
    turn_path = tmp_path / "turn.xlsx"
    inflow_path = tmp_path / "inflow.xlsx"
    pd.DataFrame(
        {
            "OpenDriveFromID": [290, 290],
            "OpenDriveToID": [298, 299],
            "TurnRatio": [0, 1],
        }
    ).to_excel(turn_path, index=False)
    pd.DataFrame({"OpenDriveFromID": [331], "Count": [10]}).to_excel(
        inflow_path, index=False
    )

    turns, inflows = util_cali_behavior.update_turn_inflow_from_solution(
        str(turn_path), str(inflow_path), np.array([0.25] * 12 + [101] * 4), 3600, 900
    )

    assert turns["TurnRatio"].tolist() == pytest.approx([0.25, 0.75])
    assert inflows["Count"].tolist() == pytest.approx([25.25])
