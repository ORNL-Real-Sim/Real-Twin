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

"""The real-twin developed by ORNL Applied Research and Mobility System (ARMS) group"""

import re
import json as json_lib
import subprocess
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
import time
import pyufunc as pf
from rich.console import Console
from rich import json
from rich import print as rprint

# environment setup
from realtwin.util_lib.create_venv import venv_create, venv_delete
from realtwin.func_lib._a_install_simulator.inst_vissim import install_vissim
from realtwin.util_lib.find_executable_path import find_executable_on_win

# input data loading
from realtwin.func_lib._b_load_inputs.loader_config import load_input_configs
from realtwin.func_lib._f_calibration._calibration_workflow import (
    apply_calibration_overrides,
    select_calibration_algorithms,
)
from realtwin.rt_vissim.cali_behavior import (
    export_behavior_info, resolve_behavior_config)

console = Console()
# info: dim cyan, warning: magenta, danger: bold red


def _run_vissim_stage(cmd: list[str]) -> None:
    """Run a Vissim stage while preserving its merged diagnostic output."""
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    if process.stdout is None:
        raise RuntimeError("Could not capture Vissim stage output.")
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


class RealTwinVISSIM:
    """
    The real-twin developed by ORNL Applied Research and Mobility System (ARMS) group that
    enables the simulation of twin-structured cities.
    """

    def __init__(self, input_config_file: str = "", **kwargs):
        """
        Initialize the RealTwinVISSIM class.

        Args:
            input_config_file (str): Path to the input configuration file.
            **kwargs: Additional keyword arguments for configuration.
        """

        # initialize the input directory
        if not input_config_file:
            raise Exception(
                "\n  :Input configuration file is not provided."
                "\n  :RealTwin requires a configuration file to be provided.")

        self.input_config = load_input_configs(input_config_file)

        self.vissim_module = {
            "step1": "realtwin.rt_vissim.01_import_opendrive",
            "step2": "realtwin.rt_vissim.02_matchup_table",
            "step3": "realtwin.rt_vissim.03_write_demand",
            "step4": "realtwin.rt_vissim.04_write_signals",
            "step5": "realtwin.rt_vissim.05_check_run",
            "step6": "realtwin.rt_vissim.06_calibrate_turn_inflow",
        }

        # add venv_create and delete as object methods
        self.venv_create = venv_create
        self.venv_delete = venv_delete
        self._venv_name = "venv_rt"
        self._proj_dir = os.getcwd()  # get current working directory

        # extract data from kwargs
        self.verbose = kwargs.get("verbose", False)

        # whether to stop the program to let user confirm input
        self._input_confirm = kwargs.get("input_confirm", True)

    def env_setup(self,
                  *,
                  sel_sim: list | None = None,
                  sel_dir: list | None = None,
                  strict_sumo_version: str | None = None,
                  strict_vissim_version: str | None = None,
                  strict_aimsun_version: str | None = None,
                  **kwargs) -> str:
        """Check and set up the environment for the simulation

        Args:
            sel_sim (list): select simulator to be set up. Default is None.
                Currently available options are ["VISSIM"].
            sel_dir (list): A list of directories to search for the executables. Defaults to None.
            strict_sumo_version (str): Whether to strictly check the version is installed.
                if specified, will check and install the version. Default is None.
            strict_vissim_version (str): Whether to strictly check the version is installed.
                if specified, will check and install the version. Default is False.
            strict_aimsun_version (str): Whether to strictly check the version is installed.
                if specified, will check and install the version. Default is False.
            kwargs: Additional keyword arguments.

        Examples:
            >>> import realtwin as rt
            >>> twin = rt.RealTwinVISSIM(input_config_file="config.yaml", verbose=True)

            check simulator is installed or not, default to VISSIM
            >>> twin.env_setup(sel_sim=["VISSIM"])

            add additional directories to search for the executables
            >>> additional_dir = [r"path-to-your-local-installed-sumo-bin"]
            >>> twin.env_setup(sel_sim=["SUMO"], sel_dir=additional_dir)

            strict version check: will install the required version if not found
            >>> twin.env_setup(sel_sim=["SUMO"], sumo_version="1.21.0", strict_sumo_version=True)

            or with additional directories
            >>> twin.env_setup(sel_sim=["SUMO"], sel_dir=additional_dir,
            >>>                sumo_version="1.21.0", strict_sumo_version=True)

        Returns:
            str: The selected simulator installation status.
        """

        # 0. Check if the sim_env is selected,
        #    default to VISSIM, case insensitive and add self.sel_sim as a class attribute
        sel_sim = [sim.lower() for sim in sel_sim] if sel_sim else ["vissim"]

        # 1. Check simulator installation - mapping function
        simulator_installation = {
            "sumo": None,
            "vissim": install_vissim,
            "aimsun": None,
        }

        # 2. check if the simulator is installed, if not, install it
        console.print("\n[bold green]Check / Install the selected simulators:")

        kwargs['sel_dir'] = sel_dir
        kwargs['strict_sumo_version'] = strict_sumo_version
        kwargs['strict_vissim_version'] = strict_vissim_version
        kwargs['strict_aimsun_version'] = strict_aimsun_version
        kwargs['verbose'] = self.verbose

        invalid_sim = []
        for simulator in sel_sim:
            try:
                sim_status = simulator_installation.get(simulator)(**kwargs)
                if not sim_status:
                    invalid_sim.append(simulator)
            except Exception:
                invalid_sim.append(simulator)
                rprint(
                    f"  :[bold magenta]Could not install {simulator} (strict version) on your operation system", end="")

        sel_sim_ = list(set(sel_sim) - set(invalid_sim))

        if not sel_sim_:
            raise Exception(
                "  :Error: No simulator is available (strict version). Please select available version(s).")
        # if "VISSIM" exist, update the dictionary with the executable path
        if isinstance(self.input_config.get("VISSIM", {}), dict):
            self.input_config["VISSIM"].update({"exe_path": sim_status[0]})
        self.sel_sim = sel_sim_

        return f"  :Info: Selected simulators: {sel_sim_}  installed successfully."

    def _control_data_enabled(self) -> bool:
        """Honor the control flag and disable loading when Control has no CSVs."""
        if self.input_config.get("VISSIM") is None:
            self.input_config["VISSIM"] = {}
        vissim_config = self.input_config["VISSIM"]
        if not vissim_config.setdefault("is_control_data", True):
            return False

        path_control = Path(self.input_config["input_dir"]) / "Control"
        if not path_control.is_dir() or not any(
            path.is_file() and path.suffix.lower() == ".csv"
            for path in path_control.iterdir()
        ):
            vissim_config["is_control_data"] = False
            console.print(
                f"  [dim cyan]:No signal CSVs in {path_control}; "
                "VISSIM.is_control_data=False. External signal loading is skipped.")
            return False
        return True

    def generate_inputs(self, *, incl_sumo_net: str | None = None, **kwargs):
        """Generate the inputs for the simulation

        Args:
            incl_sumo_net (str): The path to the updated SUMO network file (.net.xml) provided by the user.
                If provided, the OpenDrive network will be generated based on this SUMO network.
                If not provided, the OpenDrive network will be generated based on the vertices from the config file.
            kwargs: Additional keyword arguments.
        """
        is_control_data = self._control_data_enabled()
        # Select the supplied network before stage 1 imports it.
        if demo_data := self.input_config["demo_data"]:
            incl_sumo_net = pf.path2linux(
                Path(self.input_config["input_dir"]) / f"updated_net/{demo_data}.net.xml")

        if incl_sumo_net:
            if incl_sumo_net.endswith(".net.xml") and os.path.exists(incl_sumo_net):
                self.input_config["incl_sumo_net"] = incl_sumo_net
                self.input_config["VISSIM"]["sumo_xml_file"] = incl_sumo_net
            else:
                console.print(
                    "  [magenta]:NOTE: incl_sumo_net is not exist or not with .net.xml extension.\n"
                    "  :Please provide a valid SUMO file with .net.xml extension or leave it empty.",
                    soft_wrap=True, no_wrap=False)
        else:
            rprint("  [dim cyan]:INFO: You can use your own SUMO network by providing the path "
                   "to the incl_sumo_net parameter. The path should be a .net.xml file. \n",
                   end="")

        with console.status("[bold cyan]Generating inputs...", spinner="dots"):

            # Convert the selected SUMO network and import it into Vissim.
            console.print("\n[bold green]Load the input network")

            cmd = [
                sys.executable,
                "-m",
                self.vissim_module["step1"],
                "--net",
                self.input_config["VISSIM"]["sumo_xml_file"],
                "--outdir",
                pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM"),
                "--name",
                self.input_config["Network"]["NetworkName"]
            ]
            _run_vissim_stage(cmd)

            path_input = pf.path2linux(Path(self.input_config.get("input_dir")))

            # check if Control folder exists in the input directory
            path_control = pf.path2linux(Path(path_input) / "Control")
            if not os.path.exists(path_control):
                os.makedirs(path_control)

            if is_control_data:
                console.print(f"  [dim cyan]:Control folder exists: {path_control}.[/dim cyan]\n"
                              "  :NOTICE: [bold red]Please include Synchro UTDF file (signal) inside Control folder\n")

            # check if Traffic folder exists in the input directory
            path_traffic = pf.path2linux(Path(path_input) / "Traffic")
            if not os.path.exists(path_traffic):
                os.makedirs(path_traffic)

            # check if the Traffic folder is empty
            elif not os.listdir(path_traffic):
                console.print(f"  [magenta]:Traffic folder is empty: {path_traffic}.")

            console.print(f"  [dim cyan]:Traffic folder exists: {path_traffic}.[/dim cyan]\n"
                          "  :NOTICE: [bold red]Please include turn movement file for each intersection "
                          "inside Traffic folder and add the file names to the MatchupTable.xlsx "
                          "(The generated MatchupTable.xlsx is inside <output_dir>/VISSIM)."
                          " For how to fill the MatchupTable.xlsx, please refer to official documentation\n",
                          soft_wrap=True, no_wrap=False)

            # Generate the native matchup table in the VISSIM output directory.
            console.print("\n[bold green]Generate the matchup table")
            cmd = [
                sys.executable,
                "-m",
                self.vissim_module["step2"],
                "--movements",
                pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM" / f"{self.input_config['Network']['NetworkName']}_movements.csv"),
                "--outdir",
                pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM"),
                "--traffic-dir",
                path_traffic,
                "--control-dir",
                path_control,
            ]
            if not is_control_data:
                cmd.append("--no-control")

            _run_vissim_stage(cmd)

            self.input_config["VISSIM"]["matchup_table_path"] = pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM" / "MatchupTable.xlsx")

            if self.input_config["VISSIM"]["matchup_table_path"] is None:
                raise Exception("  :Error: Could not generate the matchup table. Please check the input network and try again.")

            console.print(
                f"  [dim cyan]:NOTE: Matchup table is generated and saved to {self.input_config['VISSIM']['matchup_table_path']}.[/dim cyan]\n"
                "  :NOTICE: [bold red]Please update the generated VISSIM Matchup table"
                " and then run generate_abstract_scenario()."
                " For details please refer to official documentation: \n", soft_wrap=True, no_wrap=False)

            # whether to stop the program to let user confirm input
            if self._input_confirm:
                required_data = "Control and Traffic" if is_control_data else "Traffic"
                required_columns = ("File_GridSmart, File_Synchro and IntersectionID_Synchro"
                                    if is_control_data else "File_GridSmart")
                console.rule(f"[bold green]Program stopped. Please prepare the {required_data} data and "
                            "fill in the Matchup Table before proceeding.\n"
                            "[bold red] Use the exported <network>_junctions.csv to identify each surveyed intersection. "
                            f"In the generated VISSIM/MatchupTable.xlsx, fill {required_columns} "
                            "for that junction. Keep the generated native Vissim link columns, "
                            "then save and close the workbook. \n")
                time.sleep(2)  # wait for 2 seconds before exiting
                usr_input = False
                while not usr_input:
                    usr_input = console.input(":warning: [bold magenta]Please update the generated VISSIM Matchup table "
                                                "before pressing Enter or type 'y' / 'yes' to continue")
                    if usr_input in {"", "y", "Y", "yes", "Yes"}:
                        console.print("  [dim cyan]:INFO: User confirmed to continue (Matchup Table Updated).")
                        usr_input = True

    def generate_abstract_scenario(self, **kwargs):
        """Generate the abstract scenario for the simulation

        Args:
            kwargs: Additional keyword arguments.
        """

        console.print("\n[bold green]Generate the abstract scenario")
        is_control_data = self._control_data_enabled()

        # check if the matchup table has been updated by user
        cmd = [
            sys.executable,
            "-m",
            self.vissim_module["step2"],
            "--movements",
            pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM" / f"{self.input_config['Network']['NetworkName']}_movements.csv"),
            "--outdir",
            pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM"),
            "--traffic-dir",
            pf.path2linux(Path(self.input_config["input_dir"]) / "Traffic"),
            "--control-dir",
            pf.path2linux(Path(self.input_config["input_dir"]) / "Control"),
        ]
        if not is_control_data:
            cmd.append("--no-control")

        _run_vissim_stage(cmd)

        console.print("\n[bold green]Abstract Scenario successfully generated.")

    def generate_concrete_scenario(self, **kwargs) -> str:
        is_control_data = self._control_data_enabled()
        # Import Demand
        console.print("[bold green]  :Importing Demand Data into VISSIM")

        start_time = self.input_config["Calibration"]["scenario_config"]["sim_start_time"]
        end_time = self.input_config["Calibration"]["scenario_config"]["sim_end_time"]
        # convert start time and end time to HH:MM format
        start_time = f"{start_time // 3600:02d}:{(start_time % 3600) // 60:02d}"
        end_time = f"{end_time // 3600:02d}:{(end_time % 3600) // 60:02d}"

        cmd = [
            sys.executable,
            "-m",
            self.vissim_module["step3"],
            "--inpx",
            pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM" / f"{self.input_config['Network']['NetworkName']}.inpx"),
            "--matchup",
            pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM" / "MatchupTable.xlsx"),
            "--links",
            pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM" / f"{self.input_config['Network']['NetworkName']}_links.csv"),
            "--movements",
            pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM" / f"{self.input_config['Network']['NetworkName']}_movements.csv"),
            "--traffic-dir",
            pf.path2linux(Path(self.input_config["input_dir"]) / "Traffic"),
            "--start",
            str(start_time),
            "--end",
            str(end_time)
        ]
        _run_vissim_stage(cmd)

        if not is_control_data:
            console.print("  [dim cyan]:Skipping external signal loading into VISSIM.")
            console.print("\n[bold green]Concrete Scenario successfully generated.")
            return

        # Import Signal Part 1: generate detector
        console.print("[bold green]  :Importing Signal Data into VISSIM")
        cmd = [
            sys.executable,
            "-m",
            self.vissim_module["step4"],
            "--inpx",
            pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM" / f"{self.input_config['Network']['NetworkName']}_demand.inpx"),
            "--matchup",
            pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM" / "MatchupTable.xlsx"),
            "--links",
            pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM" / f"{self.input_config['Network']['NetworkName']}_links.csv"),
            "--movements",
            pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM" / f"{self.input_config['Network']['NetworkName']}_movements.csv"),
            "--control-dir",
            pf.path2linux(Path(self.input_config["input_dir"]) / "Control"),
            "--strict",
        ]

        _run_vissim_stage(cmd)

        console.print("\n[bold green]Concrete Scenario successfully generated.")

    def prepare_simulation(self, **kwargs) -> str:

        # Generate Scenario
        console.print("[bold green]  :Generating Scenario in VISSIM")
        is_control_data = (self.input_config.get("VISSIM") or {}).get("is_control_data", True)
        network_suffix = "demand_signals" if is_control_data else "demand"
        cmd = [
            sys.executable,
            "-m",
            self.vissim_module["step5"],
            "--inpx",
            pf.path2linux(Path(self.input_config["output_dir"]) / "VISSIM" / f"{self.input_config['Network']['NetworkName']}_{network_suffix}.inpx"),
            "--run",
            "900"
        ]
        _run_vissim_stage(cmd)

        console.print("\n[bold green]VISSIM simulation successfully prepared.")

    def calibrate(
        self,
        sel_behavior_routes: list | None = None,
        *,
        sel_algo: dict | None = None,
        update_turn_inflow_algo: dict | None = None,
        update_behavior_algo: dict | None = None,
    ) -> bool:
        """Run enabled demand and behavior calibration stages with GA, SA, TS or BO.

        Args:
            sel_algo (dict | None): The selected algorithm for calibration.
            sel_behavior_routes (list | None): Routes as [name, start Vissim link,
                end Vissim link, observed travel time in seconds]. Takes precedence
                over Calibration.behavior.sel_behavior_routes.
            update_turn_inflow_algo (dict | None): The updated turn inflow algorithm settings.
            update_behavior_algo (dict | None): The updated behavior algorithm settings.

        Returns:
            bool: True if calibration was successful, False otherwise.
        """
        calibration = self.input_config.get("Calibration", {})
        if not self.input_config.get("VISSIM"):
            self.input_config["VISSIM"] = {}
        vissim_config = self.input_config["VISSIM"]
        vissim_config["turn_inflow"] = deepcopy(calibration)
        vissim_config["behavior"] = resolve_behavior_config(calibration)

        apply_calibration_overrides(
            vissim_config,
            turn_inflow=update_turn_inflow_algo,
            behavior=update_behavior_algo,
        )

        enabled = {
            "turn_inflow": vissim_config["turn_inflow"].get("turn_inflow", {}),
            "behavior": vissim_config["behavior"].get("behavior", {}),
        }
        algorithms = select_calibration_algorithms(enabled, sel_algo, console=console)
        if algorithms is None:
            return False
        if sel_behavior_routes is not None:
            vissim_config["behavior"]["sel_behavior_routes"] = deepcopy(sel_behavior_routes)
        output_dir = Path(self.input_config["output_dir"]) / "VISSIM"
        network_name = self.input_config["Network"]["NetworkName"]
        network_suffix = "demand_signals" if vissim_config.get("is_control_data", True) else "demand"
        net_path = output_dir / f"{network_name}_{network_suffix}.inpx"
        # Check behavior inputs before spending time on either calibration stage.
        # A bad configuration should identify its missing field in the notebook.
        if enabled["behavior"].get("is_calibration", False):
            export_behavior_info(self.input_config, net_path)
        # The child process must receive the live settings, including per-call
        # overrides, rather than reloading an unrelated YAML file from disk.
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="calibration_", dir=output_dir) as directory:
            config_path = Path(directory) / "config.json"
            for stage in ("turn_inflow", "behavior"):
                if not enabled[stage].get("is_calibration", False):
                    continue
                effective = vissim_config[stage]
                algorithm = algorithms[stage]
                options = effective.get(stage, {})
                suffix = "calibrated" if stage == "turn_inflow" else "behavior"
                out_path = (Path(options["out"]).resolve() if options.get("out")
                            else net_path.with_name(f"{net_path.stem}_{suffix}_{algorithm}.inpx"))
                # The explicit output avoids accidentally handing an older run's
                # file to behavior when the demand stage has no variables.
                result_path = Path(directory) / f"{stage}_result.json"
                cmd = [sys.executable, "-m", self.vissim_module["step6"],
                       "--stage", stage, "--inpx", str(net_path),
                       "--algo", algorithm, "--out", str(out_path),
                       "--result", str(result_path), "--config", str(config_path)]
                if stage == "turn_inflow":
                    cmd.extend(["--matchup", str(output_dir / "MatchupTable.xlsx"),
                                "--links", str(output_dir / f"{network_name}_links.csv"),
                                "--movements", str(output_dir / f"{network_name}_movements.csv"),
                                "--traffic-dir", str(Path(self.input_config["input_dir"]) / "Traffic")])
                snapshot = {"Calibration": effective,
                            "VISSIM": {"progid": vissim_config.get("progid")}}
                config_path.write_text(json_lib.dumps(snapshot), encoding="utf-8")
                _run_vissim_stage(cmd)
                if result_path.exists():
                    saved = json_lib.loads(result_path.read_text(encoding="utf-8"))
                    net_path = Path(saved["network"])
                    vissim_config["calibrated_model_fname"] = str(net_path)
        return True

