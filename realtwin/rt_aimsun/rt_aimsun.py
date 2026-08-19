# -*- coding:utf-8 -*-
##############################################################
# Created Date: Friday, August 7th 2026
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################

import cmd
from concurrent.futures import process
import subprocess
import os
import re
import shutil
from pathlib import Path
import time
import pyufunc as pf
from rich.console import Console
from rich import json, print as rprint
# environment setup
from realtwin.util_lib.create_venv import venv_create, venv_delete
from realtwin.util_lib.find_executable_path import find_executable_on_win
from realtwin.func_lib._a_install_simulator.inst_aimsun import install_aimsun

# input data loading
from realtwin.func_lib._b_load_inputs.loader_config import load_input_configs
from realtwin.util_lib.mapping_SUMO_OpenDrive_ID import parse_SUMO_ID, parse_SUMO_to_OpenDrive, parse_SUMO_TLS_ID

# scenario generation
# from realtwin.util_lib.download_elevation_tif import download_elevation_tif_by_bbox
from realtwin.util_lib.check_abstract_scenario_inputs import check_abstract_inputs
from realtwin.func_lib._c_abstract_scenario._abstractScenario import AbstractScenario
from realtwin.func_lib._c_abstract_scenario.rt_matchup_table_generation import generate_matchup_table
from realtwin.func_lib._c_abstract_scenario.rt_matchup_table_generation import format_junction_bearing
from realtwin.func_lib._c_abstract_scenario.rt_demand_generation import generate_turn_demand, update_matchup_table

from realtwin.func_lib._d_concrete_scenario._concreteScenario import ConcreteScenario

# simulation
from realtwin.func_lib._e_simulation._generate_simulation import SimPrep

# calibration
from realtwin.func_lib._f_calibration.calibration_sumo import cali_sumo
from realtwin.data_lib.config_data_lib import sel_behavior_routes as sel_behavior_routes_demo

console = Console()
# info: dim cyan, warning: magenta, danger: bold red


class RealTwinInputConfigError(Exception):
    """Raised when a required RealTwin input configuration file is missing."""


class RealTwinAimsun:
    """The RealTwin Aimsun integration developed by ORNL Applied Research and Mobility System (ARMS) group that
    enables the simulation of twin-structured cities.
    """
    def __init__(self, input_config_file: str = "", **kwargs):
        """Initialize the Aimsun Object

        Args:
            input_config_file (str): The directory containing the input files.
            kwargs: Additional keyword arguments. Will be used in the future.

        Raises:
            RealTwinInputConfigError: Raised when no input configuration file is supplied.
        """

        # initialize the input directory
        if not input_config_file:
            raise RealTwinInputConfigError(
                "\n  :Input configuration file is not provided."
                "\n  :RealTwin requires a configuration file to be provided.")

        self.input_config = load_input_configs(input_config_file)

        # add venv_create and delete as object methods
        self.venv_create = venv_create
        self.venv_delete = venv_delete
        self._venv_name = "venv_rt"
        self._proj_dir = os.getcwd()  # get current working directory

        # extract data from kwargs
        self.verbose = kwargs.get("verbose", False)

        # whether to stop the program to let user confirm input
        self._input_confirm = kwargs.get("input_confirm", True)

        # check and print out the Aimsun version and Aimsun's python version


        self.aimsun_file_list = [
            pf.path2linux(Path(__file__).parent / "Step0_aimsum_py_version.py"),
            pf.path2linux(Path(__file__).parent / "Step0_NetworkImport.py"),
            pf.path2linux(Path(__file__).parent / "Step1_MatchupTableGeneration.py"),
            pf.path2linux(Path(__file__).parent / "Step2_MatchupTableDataInput.py"),
            pf.path2linux(Path(__file__).parent / "Step3_DemandImport.py"),
            pf.path2linux(Path(__file__).parent / "Step4.1_DetectorGeneration.py"),
            pf.path2linux(Path(__file__).parent / "Step4.2_SignalImport.py"),
            pf.path2linux(Path(__file__).parent / "Step4.3_ControlPlanConfiguration.py"),
            pf.path2linux(Path(__file__).parent / "Step5.1_ScenarioGeneration.py"),
            pf.path2linux(Path(__file__).parent / "Step5.2_ScenarioOutputConfiguration.py")
        ]
    def env_setup(self,
                  *,
                  sel_sim: list | None = None,
                  sel_dir: list | None = None,
                  **kwargs) -> str:
        """Check and set up the environment for the simulation

        Args:
            sel_sim (list): select simulator to be set up. Default is None.
                Currently available options are ["AIMSUN"].
            sel_dir (list): A list of directories to search for the executables. Defaults to None.
            kwargs: Additional keyword arguments.

        Examples:
            >>> import realtwin as rt
            >>> twin = rt.REALTWIN(input_config_file="config.yaml", verbose=True)

            check simulator is installed or not, default to SUMO, optional: VISSIM, AIMSUN
            >>> twin.env_setup(sel_sim=["AIMSUN"])

            add additional directories to search for the executables
            >>> additional_dir = [r"path-to-your-local-installed-aimsun-bin"]
            >>> twin.env_setup(sel_sim=["AIMSUN"], sel_dir=additional_dir)

        Returns:
            str: The selected simulator installation status.
        """

        # 0. Check if the sim_env is selected,
        #    default to AIMSUN, case insensitive and add self.sel_sim as a class attribute
        sel_sim = [sim.lower() for sim in sel_sim] if sel_sim else ["aimsun"]

        # 1. Check simulator installation - mapping function
        simulator_installation = {
            "sumo": None,
            "vissim": None,
            "aimsun": install_aimsun,
        }

        # 2. check if the simulator is installed, if not, install it
        console.print("\n[bold green]Check / Install the selected simulators:")

        kwargs['sel_dir'] = sel_dir
        kwargs['verbose'] = self.verbose

        invalid_sim = []
        for simulator in sel_sim:
            try:
                sim_status = simulator_installation.get(simulator)(**kwargs)
                if not sim_status:
                    invalid_sim.append(simulator)
            except Exception:
                invalid_sim.append(simulator)
                rprint(f"  :[bold magenta]Could not install {simulator} on your operation system", end="")

        sel_sim_ = list(set(sel_sim) - set(invalid_sim))

        if not sel_sim_:
            raise Exception("  :Error: No simulator is available. Please select available version(s).")  # noqa: TRY002
        self.sel_sim = sel_sim_

        # check Aimsun version installation
        try:
            print("  :[bold green]Check Aimsun version installation:")
            path_to_aimsun = pf.path2linux(find_executable_on_win("aconsole.exe", sel_dir=sel_dir, verbose=False)[0])
            # print(f"  :[bold green]Aimsun executable path: {path_to_aimsun}", end="")
            model_path = self.input_config["AIMSUN"]["model_fname"]
            cmd = [
                rf"{path_to_aimsun}",
                "-script",
                rf"{self.aimsun_file_list[0]}",
                rf"{model_path}"
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            out, _ = process.communicate()

            # split the output by lines
            version_flag = False
            for line in out.splitlines():
                if "Aimsun Python Version" in line and "Not Found" not in line:
                    version_flag = True
                    self.input_config["AIMSUN"]["python_version"] = line.split(":")[-1].strip()
                    self.input_config["AIMSUN"]["exe_path"] = path_to_aimsun
                    rprint(f"  :[bold green]{line}", end="")
                    rprint(
                        "  :[bold green]Please ensure that your Python site-packages are compatible with the Aimsun Python version you are using.", end="")
                    break
            if not version_flag:
                rprint("  :[bold magenta]Could not check AIMSUN version, no dongle found or not installed properly.", end="")

        except Exception as e:  # noqa: BLE001
            rprint(f"  :[bold magenta]Could not check Aimsun version: {e}, no dongle found", end="")
            return f"  :Info: Selected simulators: {sel_sim_} are installed successfully."

        return f"  :Info: Selected simulators: {sel_sim_} are installed successfully."

    def generate_inputs(self, **kwargs) -> str:
        """Generate the required input files for the simulation.

        Args:
            kwargs: Additional keyword arguments.
        """
        with console.status("[bold cyan]Generating inputs...", spinner="dots"):

            # Load the input xodr file and import it into Aimsun
            console.print("\n[bold green]Load the input network")

            cmd = [
                self.input_config["AIMSUN"]["exe_path"],
                "-script",
                self.aimsun_file_list[1],
                self.input_config["AIMSUN"]["model_fname"],
                json.dumps(self.input_config)
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
            out, _ = process.communicate()
            if "Cannot load the network" in out:
                raise Exception("  :Error: Cannot load the network. Please check the Aimsun model file path and ensure it is correct.")  # noqa: TRY002

            # check demand and control data
            # check if Control folder exists in the input directory
            path_model = Path(self.input_config["AIMSUN"]["model_fname"]).parent
            path_control = pf.path2linux(path_model / "Control")
            if not os.path.exists(path_control):
                os.makedirs(path_control)

            # check if the Control folder is empty
            elif not os.listdir(path_control):
                console.print(f"[dim cyan]Control folder is empty: {path_control}.")

            console.print(f"  [dim cyan]:Control folder exists: {path_control}.[/dim cyan]\n"
                            "  :NOTICE: [bold red]Please include Synchro UTDF file (signal) inside Control folder\n")

            # check if Traffic folder exists in the input directory
            path_traffic = pf.path2linux(path_model / "Traffic")
            if not os.path.exists(path_traffic):
                os.makedirs(path_traffic)

            # check if the Traffic folder is empty
            elif not os.listdir(path_traffic):
                console.print(f"  [magenta]:Traffic folder is empty: {path_traffic}.")

            console.print(f"  [dim cyan]:Traffic folder exists: {path_traffic}.[/dim cyan]\n"
                            "  :NOTICE: [bold red]Please include turn movement file for each intersection "
                            "inside Traffic folder and add the file names to the MatchupTable.xlsx "
                            "(You will notice the generated MatchupTable.xlsx inside your input folder)."
                            " For how to fill the MatchupTable.xlsx, please refer to official documentation\n",
                            soft_wrap=True, no_wrap=False)

            # generate the matchup table and save it to the input directory
            console.print("\n[bold green]Generate the matchup table")
            cmd = [
                self.input_config["AIMSUN"]["exe_path"],
                "-script",
                self.aimsun_file_list[2],
                self.input_config["AIMSUN"]["model_fname"],
                # json.dumps(self.input_config)
            ]

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            out, _ = process.communicate()
            for line in out.splitlines():
                if "Matchup table saved to" in line:
                    self.input_config["AIMSUN"]["matchup_table_path"] = line.split(":")[-1].strip()
                    break
            if self.input_config["AIMSUN"]["matchup_table_path"] is None:
                raise Exception("  :Error: Could not generate the matchup table. Please check the Aimsun model file and ensure it is correct.")  # noqa: TRY002

    def generate_abstract_scenario(self, **kwargs) -> str:
            console.print(
                f"  [dim cyan]:NOTE: Matchup table is generated and saved to {self.input_config['AIMSUN']['matchup_table_path']}.[/dim cyan]\n"
                "  :NOTICE: [bold red]Please update the Matchup table from input folder"
                " and then run generate_abstract_scenario()."
                " For details please refer to official documentation: \n", soft_wrap=True, no_wrap=False)

            # whether to stop the program to let user confirm input
            if self._input_confirm:
                console.rule("[bold green]Program stopped. Please prepare the Control and Traffic data and "
                            "fill in the Matchup Table before proceeding.\n"
                            "[bold red] As an example, open the aimsun network, double click on the left-most intersection (Shallowford & Amin) ,get its node id, close aimsun! In the MatchupTable.xlsx, find the row with this node id, fill in column G of this row with Shallowford+Amin_07112023.xls fill in column K of this row with Synchro_signal.csv, fill in column L of this row with 4, then save and close. \n")
                time.sleep(2)  # wait for 2 seconds before exiting
                usr_input = False
                while not usr_input:
                    usr_input = console.input(":warning: [bold magenta]Please update the generated Matchup table from "
                                                "input folder before pressing Enter or type 'y' / 'yes' to continue")
                    if usr_input in {"", "y", "Y", "yes", "Yes"}:
                        console.print("  [dim cyan]:INFO: User confirmed to continue (Matchup Table Updated).")
                        usr_input = True

            # Grab Data From Traffic and Control Folder into Matchup Table
            print("[bold green]  :Importing Traffic and Control Data into Matchup Table")
            cmd = [
                self.input_config["AIMSUN"]["exe_path"],
                "-script",
                self.aimsun_file_list[3],
                self.input_config["AIMSUN"]["model_fname"],
                json.dumps(self.input_config)
            ]

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            out, _ = process.communicate()
            # print(out)


            # Import Demand
            print("[bold green]  :Importing Demand Data into Matchup Table")
            cmd = [
                self.input_config["AIMSUN"]["exe_path"],
                "-script",
                self.aimsun_file_list[4],
                self.input_config["AIMSUN"]["model_fname"],
                json.dumps(self.input_config)
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            out, _ = process.communicate()
            # print(out)

            # Import Signal Part 1: generate detector
            print("[bold green]  :Importing Signal Data into Matchup Table: Part 1 - generate detector")
            cmd = [
                self.input_config["AIMSUN"]["exe_path"],
                "-script",
                self.aimsun_file_list[5],
                self.input_config["AIMSUN"]["model_fname"],
                json.dumps(self.input_config)
            ]

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            out, _ = process.communicate()
            # print(out)

            # Import Signal Part 2: import signal
            print("[bold green]  :Importing Signal Data into Matchup Table: Part 2 - import signal")
            cmd = [
                self.input_config["AIMSUN"]["exe_path"],
                "-script",
                self.aimsun_file_list[6],
                self.input_config["AIMSUN"]["model_fname"],
                json.dumps(self.input_config)
            ]

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            out, _ = process.communicate()
            # print(out)

            # Import Signal Part 3: configure Aimsun control plan
            print("[bold green]  :Importing Signal Data into Matchup Table: Part 3 - configure Aimsun control plan")
            cmd = [
                self.input_config["AIMSUN"]["exe_path"],
                "-script",
                self.aimsun_file_list[7],
                self.input_config["AIMSUN"]["model_fname"],
                json.dumps(self.input_config)
            ]

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            out, _ = process.communicate()
            # print(out)

            console.print("\n[bold green]Abstract Scenario successfully generated.")

    def generate_concrete_scenario(self, **kwargs) -> str:

        # Generate Scenario
        print("[bold green]  :Generating Abstract Scenario")
        cmd = [
            self.input_config["AIMSUN"]["exe_path"],
            "-script",
            self.aimsun_file_list[7],
            self.input_config["AIMSUN"]["model_fname"],
            # json.dumps(self.input_config)
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        out, _ = process.communicate()
        # print(out)

        # Configure Simulation Output Path in Scenario
        print("[bold green]  :Configuring Simulation Output Path in Scenario")
        cmd = [
            self.input_config["AIMSUN"]["exe_path"],
            "-script",
            self.aimsun_file_list[7],
            self.input_config["AIMSUN"]["model_fname"],
            # json.dumps(self.input_config)
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        out, _ = process.communicate()
        # print(out)
        console.print("\n[bold green]Concrete Scenario successfully generated.")

