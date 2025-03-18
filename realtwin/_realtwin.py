##############################################################################
# Copyright (c) 2024-, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a MIT               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################

"""The real-twin developed by ORNL Applied Research and Mobility System (ARMS) group"""
import os
from pathlib import Path
import pyufunc as pf

# environment setup
from realtwin.utils_lib.create_venv import venv_create, venv_delete
from realtwin.func_lib._a_install_simulator.inst_sumo import install_sumo

# input data loading
from realtwin.func_lib._b_load_inputs.loader_config import load_input_config

# scenario generation
from realtwin.utils_lib.download_elevation_tif import download_elevation_tif_by
from realtwin.func_lib._c_abstract_scenario._abstractScenario import AbstractScenario
from realtwin.func_lib._d_concrete_scenario._concreteScenario import ConcreteScenario

# simulation
from realtwin.func_lib._e_simulation._generate_simulation import SimPrep

# calibration
from realtwin.func_lib._f_calibration.calibration_sumo import cali_sumo
from realtwin.func_lib._f_calibration.calibration_sumo_ import cali_sumo as cali_sumo_


class RealTwin:
    """The real-twin developed by ORNL Applied Research and Mobility System (ARMS) group that
    enables the simulation of twin-structured cities.
    """

    def __init__(self, input_config_file: str = "", **kwargs):
        """Initialize the REALTWIN object.

        Args:
            input_config_file (str): The directory containing the input files.
            kwargs: Additional keyword arguments. Will be used in the future.
        """

        # initialize the input directory
        if not input_config_file:
            raise Exception(
                "\n  :Input configuration file is not provided."
                "\n  :RealTwin will use default network for demonstration in the future.")

        self.input_config = load_input_config(input_config_file)

        # add venv_create and delete as object methods
        self.venv_create = venv_create
        self.venv_delete = venv_delete
        self._venv_name = "venv_rt"
        self._proj_dir = os.getcwd()  # get current working directory

        # extract data from kwargs
        self.verbose = kwargs["verbose"] if "verbose" in kwargs else False

    def env_setup(self,
                  *,
                  sel_sim: list = None,
                  sel_dir: list = None,
                  strict_sumo_version: str = None,
                  strict_vissim_version: str = None,
                  strict_aimsun_version: str = None,
                  create_venv: bool = False,
                  **kwargs) -> bool:
        """Check and set up the environment for the simulation

        Args:
            sel_sim (list): select simulator to be set up. Default is None.
                Currently available options are ["SUMO", "VISSIM", "AIMSUN"].
            sel_dir (list): A list of directories to search for the executables. Defaults to None.
            strict_sumo_version (str): Whether to strictly check the version is installed.
                if specified, will check and install the version. Default is None.
            strict_vissim_version (str): Whether to strictly check the version is installed.
                if specified, will check and install the version. Default is False.
            strict_aimsun_version (str): Whether to strictly check the version is installed.
                if specified, will check and install the version. Default is False.
            create_venv (bool): Whether to create a virtual environment. Default is False.
            kwargs: Additional keyword arguments.

        Examples:
            >>> import realtwin as rt
            >>> twin = rt.REALTWIN(input_config_file="config.yaml", verbose=True)

            check simulator is installed or not, default to SUMO, optional: VISSIM, AIMSUN
            >>> twin.env_setup(sel_sim=["SUMO"])

            add additional directories to search for the executables
            >>> additional_dir = [r"path-to-your-local-installed-sumo-bin"]
            >>> twin.env_setup(sel_sim=["SUMO"], sel_dir=additional_dir)

            strict version check: will install the required version if not found
            >>> twin.env_setup(sel_sim=["SUMO"], sumo_version="1.21.0", strict_sumo_version=True)

            or with additional directories
            >>> twin.env_setup(sel_sim=["SUMO"], sel_dir=additional_dir,
            >>>                sumo_version="1.21.0", strict_sumo_version=True)

        Returns:
            bool: True if the environment is set up successfully, False otherwise.
        """

        # 0 create a virtual environment
        if create_venv:
            print(f"Creating a virtual environment: {self._venv_name}")
            self.venv_create(venv_name=self._venv_name,
                             venv_dir=self.input_config["output_dir"],
                             verbose=True)

        # 0. Check if the sim_env is selected,
        #    default to SUMO, case insensitive and add self.sel_sim as a class attribute
        sel_sim = ["sumo"] if not sel_sim else [sim.lower() for sim in sel_sim]

        # 1. Check simulator installation - mapping function
        simulator_installation = {
            "sumo": install_sumo,
            "vissim": None,
            "aimsun": None,
        }

        # 2. check if the simulator is installed, if not, install it
        print("\nCheck / install the selected simulators:")
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
                print(f"\n  :Could not install {simulator} (strict version) on your operation system")

        sel_sim_ = list(set(sel_sim) - set(invalid_sim))

        if not sel_sim_:
            raise Exception("  :Error: No simulator is available (strict version). Please select available version(s).")
        self.sel_sim = sel_sim_

        return True

    def generate_abstract_scenario(self, *, incl_elevation_tif: bool = True):
        """Generate the abstract scenario: create OpenDrive files
        """
        # check whether the elevation tif data is provided
        path_elev = pf.path2linux(
            Path(self.input_config.get("input_dir")) / self.input_config.get("Network").get("ElevationMap"))
        if not os.path.exists(path_elev):
            print("  :Elevation map is not provided. we will download from network BBOX.")
            if incl_elevation_tif:
                print("  :Downloading elevation map from network BBOX.")
                # download elevation map from network bbox
                bbox = self.input_config.get("Network").get("Net_BBox")
                output_file = pf.path2linux(Path(self.input_config.get("input_dir")) / "elevation_map.tif")
                download_elevation_tif_by(bbox, output_file)

                # update tif file in the input configuration
                self.input_config.get("Network")["ElevationMap"] = "elevation_map.tif"

        # 1. Generate the abstract scenario based on the input data
        self.abstract_scenario = AbstractScenario(self.input_config)
        self.abstract_scenario.update_AbstractScenario_from_input()
        print("\nAbstract Scenario successfully generated.")

    def generate_concrete_scenario(self):
        """Generate the concrete scenario: generate unified scenario from abstract scenario
        """

        # 1. Generate the concrete scenario based on abstract scenario
        # 2. Save the concrete scenario to the output directory

        if not hasattr(self, 'abstract_scenario'):
            print("  :Warning: Abstract Scenario is not generated yet. Please (generate_abstract_scenario).")
            return

        self.concrete_scenario = ConcreteScenario()
        self.concrete_scenario.get_unified_scenario(self.abstract_scenario)
        print("\nConcrete Scenario successfully generated.")

    def prepare_simulation(self,
                           start_time: float = 3600 * 8,
                           end_time: float = 3600 * 10,
                           seed: list | int = 812,
                           step_length: float = 0.1) -> bool:
        """Simulate the concrete scenario: generate simulation files for the selected simulator

        Args:
            start_time (float): The start time of the simulation. Default is 3600 * 8.
            end_time (float): The end time of the simulation. Default is 3600 * 10.
            seed (list or int): The seed for the simulation. Default is [101].
            step_length (float): The simulation step size. Default is 0.1.

        Examples:
            import realtwin package
            >>> import realtwin as rt

            load the input configuration file
            >>> twin = rt.REALTWIN(input_config_file="config.yaml", verbose=True)

            check simulator is installed or not, default to SUMO
            >>> twin.env_setup(sel_sim=["SUMO"])

            generate abstract scenario and concrete scenario
            >>> twin.generate_abstract_scenario()
            >>> twin.generate_concrete_scenario()

            prepare simulation with start time, end time, seed, and step size
            >>> twin.prepare_simulation(start_time=3600 * 8, end_time=3600 * 10, seed=[101], step_length=0.1)

        Returns:
            bool: True if the simulation is prepared successfully, False otherwise.
        """

        # 1. prepare Simulate docs from the concrete scenario
        # 2. Save results to the output directory

        sim_prep = {
            "sumo": SimPrep().create_sumo_sim,
            "vissim": SimPrep().create_vissim_sim,
            "aimsun": SimPrep().create_aimsun_sim,
        }

        # TODO according sel_sim to run different simulators
        self.sim = SimPrep()
        for simulator in self.sel_sim:
            sim_prep.get(simulator)(self.concrete_scenario,
                                    start_time=start_time,
                                    end_time=end_time,
                                    seed=seed,
                                    step_length=step_length)
            print(f"\n{simulator.upper()} simulation successfully Prepared.")
        return True

    def calibrate(self, *, sel_algo: dict = None) -> bool:
        """Calibrate the turn and inflow, and behavioral parameters using the selected algorithms.

        Args:
            sel_algo (dict): The dictionary of algorithms to be used for calibration.
                Default is None, will use genetic algorithm. e.g. {"turn_inflow": "ga", "behavior": "ga"}.

        """
        # TDD
        print()
        if sel_algo is None:  # default to genetic algorithm
            sel_algo = {"turn_inflow": "ga",
                        "behavior": "ga"}

        if not isinstance(sel_algo, dict):
            print("  :Error:parameter sel_algo must be a dict with"
                  " keys of 'turn_inflow' and 'behavior', using"
                  " genetic algorithm as default values.")
            sel_algo = {"turn_inflow": "ga", "behavior": "ga"}

        # check if the selected algorithm is supported within the package
        # convert the algorithm to lower case
        sel_algo = {key: value.lower() for key, value in sel_algo.items()}
        if (algo := sel_algo["turn_inflow"]) not in ["ga", "sa", "ts"]:
            print(f"  :Selected algorithms are {sel_algo}")
            print(f"  :{algo} for turn and inflow calibration is not supported. Must be one of ['ga', 'sa', 'ts']")
            return False

        if (algo := sel_algo["behavior"]) not in ["ga", "sa", "ts"]:
            print(f"  :Selected algorithms are {sel_algo}")
            print(f"  :{algo} for behavior calibration is not supported. Must be one of ['ga', 'sa', 'ts']")
            return False

        # run calibration based on the selected algorithm
        # cali_sumo(sel_algo=sel_algo, input_config=self.input_config, verbose=self.verbose)
        cali_sumo_(sel_algo=sel_algo, input_config=self.input_config, verbose=self.verbose)

        print("  :Calibration successfully completed.")

        return True

    def post_process(self):
        """Post-process the simulation results.
        """

        # 1. Post-process the simulation results
        # 2. Save the post-processed results to the output directory

    def visualize(self):
        """Visualize the simulation results.
        """

        # 1. Visualize the simulation results
        # 2. Save the visualization results to the output directory
