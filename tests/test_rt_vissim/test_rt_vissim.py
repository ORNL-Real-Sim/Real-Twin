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

"""Regression tests for the Vissim workflow."""

import subprocess
import sys

import pytest

from realtwin.rt_vissim import rt_vissim as vissim_module
from realtwin.rt_vissim.network import VissimLink
from realtwin.rt_vissim.routes import decision_position, route_end_position
from realtwin.rt_vissim.routes import plan_decision_positions


@pytest.mark.parametrize("use_supplied_network", [False, True])
def test_generate_inputs_runs_stages_as_package_modules(
    monkeypatch, tmp_path, use_supplied_network
):
    """Stage scripts should not shadow standard-library modules."""
    input_config = {
        "VISSIM": {"sumo_xml_file": "network.net.xml"},
        "Network": {"NetworkName": "network"},
        "demo_data": None,
        "input_dir": str(tmp_path / "input"),
        "output_dir": str(tmp_path / "output"),
    }
    commands = []

    class StubProcess:
        """Record subprocess commands without launching Vissim."""

        def __init__(self, command, **_kwargs):
            commands.append(command)
            self.returncode = 0
            self.stdout = iter(())

        def wait(self):
            return self.returncode

    monkeypatch.setattr(vissim_module, "load_input_configs", lambda _path: input_config)
    monkeypatch.setattr(vissim_module.subprocess, "Popen", StubProcess)

    twin = vissim_module.RealTwinVISSIM("config.yaml", input_confirm=False)
    supplied_network = tmp_path / "supplied.net.xml"
    supplied_network.write_text("<net/>", encoding="utf-8")
    twin.generate_inputs(
        incl_sumo_net=str(supplied_network) if use_supplied_network else None)

    assert commands[0][:3] == [
        sys.executable,
        "-m",
        "realtwin.rt_vissim.01_import_opendrive",
    ]
    assert commands[1][:3] == [
        sys.executable,
        "-m",
        "realtwin.rt_vissim.02_matchup_table",
    ]
    expected_network = str(supplied_network) if use_supplied_network else "network.net.xml"
    assert commands[0][commands[0].index("--net") + 1] == expected_network
    assert commands[1][commands[1].index("--traffic-dir") + 1] == (
        tmp_path / "input" / "Traffic"
    ).as_posix()
    assert commands[1][commands[1].index("--control-dir") + 1] == (
        tmp_path / "input" / "Control"
    ).as_posix()
    assert "--no-control" in commands[1]
    assert twin.input_config["VISSIM"]["is_control_data"] is False


@pytest.mark.parametrize(
    "control_contents, config_flag, expected_enabled",
    [
        ("missing", None, False),
        ("empty", None, False),
        ("notes", None, False),
        ("directory", None, False),
        ("signal", None, True),
        ("signal", False, False),
    ],
)
def test_generate_inputs_detects_optional_control(
    monkeypatch, tmp_path, control_contents, config_flag, expected_enabled
):
    """Only signal CSV files enable control; an explicit opt-out takes precedence."""
    commands = []
    monkeypatch.setattr(vissim_module, "_run_vissim_stage", commands.append)
    control = tmp_path / "inputs" / "Control"
    if control_contents != "missing":
        control.mkdir(parents=True)
    if control_contents == "signal":
        (control / "signals.CSV").write_text("signal data", encoding="utf-8")
    elif control_contents == "notes":
        (control / "README.txt").write_text("notes", encoding="utf-8")
    elif control_contents == "directory":
        (control / "signals.csv").mkdir()
    network = tmp_path / "network.net.xml"
    network.write_text("<net/>", encoding="utf-8")
    twin = vissim_module.RealTwinVISSIM.__new__(vissim_module.RealTwinVISSIM)
    twin.input_config = {
        "VISSIM": None if config_flag is None else {"is_control_data": config_flag},
        "Network": {"NetworkName": "network"},
        "demo_data": None,
        "input_dir": str(tmp_path / "inputs"),
        "output_dir": str(tmp_path / "outputs"),
    }
    twin.vissim_module = {"step1": "import_stage", "step2": "matchup_stage"}
    twin._input_confirm = False

    twin.generate_inputs(incl_sumo_net=str(network))

    assert control.is_dir()
    assert twin.input_config["VISSIM"]["is_control_data"] is expected_enabled
    assert ("--no-control" not in commands[1]) is expected_enabled


@pytest.mark.parametrize("failed_stage", [1, 2])
def test_generate_inputs_propagates_stage_failures(monkeypatch, tmp_path, failed_stage):
    """Import or workbook errors must stop input generation."""
    commands = []

    def run_stage(command):
        commands.append(command)
        if len(commands) == failed_stage:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(vissim_module, "_run_vissim_stage", run_stage)
    twin = vissim_module.RealTwinVISSIM.__new__(vissim_module.RealTwinVISSIM)
    twin.input_config = {
        "VISSIM": {"sumo_xml_file": "network.net.xml"},
        "Network": {"NetworkName": "network"},
        "demo_data": None,
        "input_dir": str(tmp_path / "input"),
        "output_dir": str(tmp_path / "output"),
    }
    twin.vissim_module = {"step1": "import_stage", "step2": "matchup_stage"}
    twin._input_confirm = False

    with pytest.raises(subprocess.CalledProcessError):
        twin.generate_inputs()

    assert len(commands) == failed_stage
    assert "matchup_table_path" not in twin.input_config["VISSIM"]


def test_generate_abstract_scenario_uses_configured_data_folders(monkeypatch, tmp_path):
    """Abstract preparation must use the same dataset as concrete preparation."""
    commands = []
    monkeypatch.setattr(vissim_module, "_run_vissim_stage", commands.append)
    twin = vissim_module.RealTwinVISSIM.__new__(vissim_module.RealTwinVISSIM)
    twin.input_config = {
        "Network": {"NetworkName": "network"},
        "input_dir": str(tmp_path / "custom_input"),
        "output_dir": str(tmp_path / "custom_output"),
    }
    twin.vissim_module = {"step2": "realtwin.rt_vissim.02_matchup_table"}

    twin.generate_abstract_scenario()

    assert commands[0][commands[0].index("--traffic-dir") + 1] == (
        tmp_path / "custom_input" / "Traffic"
    ).as_posix()
    assert commands[0][commands[0].index("--control-dir") + 1] == (
        tmp_path / "custom_input" / "Control"
    ).as_posix()
    assert "--no-control" in commands[0]
    assert twin.input_config["VISSIM"]["is_control_data"] is False


def test_short_link_route_markers_stay_ordered_and_in_bounds():
    """Short OpenDRIVE links should receive valid Vissim marker positions."""
    link_length = 0.71
    gap = 3.33

    route_end = route_end_position(link_length, gap)
    decision, warnings = decision_position(link_length, gap)

    assert 0.0 <= route_end < decision <= link_length
    assert warnings


def test_normal_link_route_markers_keep_standard_positions():
    """Links with enough room should retain the existing placement contract."""
    gap = 3.33

    assert route_end_position(20.0, gap) == 2.0
    assert decision_position(20.0, gap) == (5.33, [])


def test_decision_skips_short_upstream_destination_link():
    """A short prior route destination should not host the next decision."""
    links = {
        4: VissimLink(no=4, length=0.71),
        6: VissimLink(no=6, length=45.95),
        7: VissimLink(no=7, length=25.37),
        100: VissimLink(
            no=100, is_connector=True, from_link=4, to_link=6
        ),
        101: VissimLink(
            no=101, is_connector=True, from_link=6, to_link=7
        ),
    }

    placements, warnings = plan_decision_positions(
        [(1, 7)], links, destinations={4}, gap=3.33
    )

    assert placements[(1, 7)] == (6, 3.33)
    assert any("destination link 4" in warning for warning in warnings)


@pytest.mark.parametrize("control_flag", [None, True, False])
@pytest.mark.parametrize("has_signal_data", [False, True])
def test_generate_concrete_scenario_honors_optional_control(
    monkeypatch, tmp_path, control_flag, has_signal_data
):
    """Demand is always generated; signals require enabled, available control data."""
    commands = []
    monkeypatch.setattr(vissim_module, "_run_vissim_stage", commands.append)

    twin = vissim_module.RealTwinVISSIM.__new__(vissim_module.RealTwinVISSIM)
    twin.input_config = {
        "Calibration": {
            "scenario_config": {
                "sim_start_time": 28800,
                "sim_end_time": 32400,
            }
        },
        "Network": {"NetworkName": "network"},
        "input_dir": str(tmp_path / "custom_input"),
        "output_dir": str(tmp_path),
    }
    twin.vissim_module = {
        "step3": "realtwin.rt_vissim.03_write_demand",
        "step4": "realtwin.rt_vissim.04_write_signals",
    }
    if control_flag is not None:
        twin.input_config["VISSIM"] = {"is_control_data": control_flag}
    if has_signal_data:
        control = tmp_path / "custom_input" / "Control"
        control.mkdir(parents=True)
        (control / "signals.csv").write_text("signal data", encoding="utf-8")

    twin.generate_concrete_scenario()

    expected_stages = ["realtwin.rt_vissim.03_write_demand"]
    if has_signal_data and control_flag is not False:
        expected_stages.append("realtwin.rt_vissim.04_write_signals")
    assert [command[2] for command in commands] == expected_stages
    assert commands[0][-4:] == ["--start", "08:00", "--end", "09:00"]
    assert commands[0][commands[0].index("--traffic-dir") + 1] == (
        tmp_path / "custom_input" / "Traffic"
    ).as_posix()
    if len(commands) == 1:
        assert twin.input_config["VISSIM"]["is_control_data"] is False
        return
    assert commands[1][commands[1].index("--control-dir") + 1] == (
        tmp_path / "custom_input" / "Control"
    ).as_posix()
    assert commands[1][commands[1].index("--inpx") + 1].endswith(
        "network_demand.inpx"
    )


@pytest.mark.parametrize("control_flag", [None, True, False])
def test_prepare_simulation_runs_selected_concrete_network(monkeypatch, tmp_path, control_flag):
    """A disabled signal stage must not hand simulation a stale signal network."""
    commands = []
    monkeypatch.setattr(vissim_module, "_run_vissim_stage", commands.append)

    twin = vissim_module.RealTwinVISSIM.__new__(vissim_module.RealTwinVISSIM)
    twin.input_config = {
        "Network": {"NetworkName": "network"},
        "output_dir": str(tmp_path),
    }
    twin.vissim_module = {"step5": "realtwin.rt_vissim.05_check_run"}
    if control_flag is not None:
        twin.input_config["VISSIM"] = {"is_control_data": control_flag}
    output = tmp_path / "VISSIM"
    output.mkdir()
    for suffix in ("demand", "demand_signals"):
        (output / f"network_{suffix}.inpx").write_text("<network/>", encoding="utf-8")

    twin.prepare_simulation()

    assert commands[0][2] == "realtwin.rt_vissim.05_check_run"
    expected_suffix = "demand" if control_flag is False else "demand_signals"
    assert commands[0][commands[0].index("--inpx") + 1].endswith(
        f"network_{expected_suffix}.inpx")
    assert commands[0][-2:] == ["--run", "900"]
