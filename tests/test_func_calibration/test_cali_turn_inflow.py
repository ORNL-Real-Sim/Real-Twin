##############################################################################
# Copyright (c) 2024-, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################

"""Tests for SUMO turn and inflow calibration."""

import socket
from unittest.mock import patch

import numpy as np
import pytest

# pyufunc resolves the local hostname during import. Some isolated test
# environments do not publish that hostname in DNS, so keep the import
# deterministic without changing application behavior.
with patch.object(socket, "gethostbyname", return_value="127.0.0.1"):
    from realtwin.func_lib._f_calibration.algo_sumo.cali_turn_inflow import fitness_func_turn_flow


@pytest.mark.parametrize(
    ("geh_percent", "expected_objective"),
    [
        (0.0, 1.0),
        (0.6842, 0.3158),
        (0.75, 0.25),
        (1.0, 0.0),
    ],
)
def test_fitness_prioritizes_geh_percentage(tmp_path, geh_percent, expected_objective):
    """A higher accepted-link percentage must always produce a better fitness."""
    scenario_config = {
        "TurnDf_Calibration": object(),
        "TurnToCalibrate": object(),
        "InflowDf_Calibration": object(),
        "InflowEdgeToCalibrate": object(),
        "RealSummary_Calibration": object(),
        "network_name": "test-network",
        "sim_start_time": 0,
        "sim_end_time": 3600,
        "path_net": "test-network.net.xml",
        "dir_turn_inflow": str(tmp_path),
        "sim_name": "test-network.sumocfg",
        "calibration_target": {"GEH": 5, "GEHPercent": 0.85},
    }

    with (
        patch(
            "realtwin.func_lib._f_calibration.algo_sumo.cali_turn_inflow.update_turn_flow_from_solution",
            return_value=(object(), object()),
        ),
        patch(
            "realtwin.func_lib._f_calibration.algo_sumo.cali_turn_inflow.run_jtrrouter_to_create_rou_xml"
        ),
        patch(
            "realtwin.func_lib._f_calibration.algo_sumo.cali_turn_inflow.run_SUMO_create_EdgeData"
        ),
        patch(
            "realtwin.func_lib._f_calibration.algo_sumo.cali_turn_inflow.result_analysis_on_EdgeData",
            return_value=(0, 4.0, geh_percent),
        ),
    ):
        objective = fitness_func_turn_flow(np.array([0.5]), scenario_config)

    assert objective == pytest.approx(expected_objective)
