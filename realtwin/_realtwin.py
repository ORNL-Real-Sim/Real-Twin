##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a __TBD__           #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors (Add you name below to acknowledge your contribution):        #
# Xiangyong Roy Luo                                                          #
##############################################################################

"""The real-twin developed by ORNL Applied Research and Mobility System (ARMS) group"""

from realtwin.utils_lib.create_venv import venv_create, venv_delete
from realtwin.func_lib._a_install_simulator.inst_sumo import install_sumo
from realtwin.func_lib._b_load_inputs.loader_config import load_input_config

from realtwin.func_lib._c_abstract_scenario._abstractScenario import AbstractScenario
from realtwin.func_lib._d_concrete_scenario._concreteScenario import ConcreteScenario
from realtwin.func_lib._e_simulation._generate_simulation import SimPrep


class REALTWIN:
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

        # extract data from kwargs
        self.verbose = kwargs["verbose"] if "verbose" in kwargs else False

    def env_setup(self,
                  *,
                  sel_sim: list = None,
                  sel_dir: list = None,
                  sumo_version: str = "1.21.0",
                  vissim_version: str = "",
                  aimsun_version: str = "",
                  strict_sumo_version: bool = False,
                  strict_vissim_version: bool = False,
                  strict_aimsun_version: bool = False,
                  create_venv: bool = False,
                  **kwargs):
        """Check and set up the environment for the simulation

        Args:
            sel_sim (list): select simulator to be set up. Default is None.
                Currently available options are ["SUMO", "VISSIM", "AIMSUN"].
            sel_dir (list): A list of directories to search for the executables. Defaults to None.
            sumo_version (str): The version of SUMO to be installed. Default is "1.20.0".
            vissim_version (str): The version of VISSIM to be installed. Default is "".
            aimsun_version (str): The version of Aimsun to be installed. Default
            strict_sumo_version (bool): Whether to strictly check the version is installed.
                Default is False.
            strict_vissim_version (bool): Whether to strictly check the version is installed.
                Default is False.
            strict_aimsun_version (bool): Whether to strictly check the version is installed.
                Default is False.
            create_venv (bool): Whether to create a virtual environment. Default is False.
            kwargs: Additional keyword arguments.

        Examples:
            >>> import realtwin as rt
            >>> twin = rt.REALTWIN(input_config_file="config.yaml", verbose=True)

            # check simulator is installed or not, default to SUMO, optional: VISSIM, AIMSUN
            >>> twin.env_setup(sel_sim=["SUMO"])

            # add additional directories to search for the executables
            >>> additional_dir = [r"path-to-your-local-installed-sumo-bin"]
            >>> twin.env_setup(sel_sim=["SUMO"], sel_dir=additional_dir)

            # strict version check: will install the required version if not found
            >>> twin.env_setup(sel_sim=["SUMO"], sumo_version="1.21.0", strict_sumo_version=True)

            # or with additional directories
            >>> twin.env_setup(sel_sim=["SUMO"], sel_dir=additional_dir,
            >>>                sumo_version="1.21.0", strict_sumo_version=True)

        Returns:
            None
        """

        # 0 create a virtual environment
        if create_venv:
            print(f"Creating a virtual environment: {self._venv_name}")
            self.venv_create(venv_name=self._venv_name,
                             venv_dir=self.input_config["output_dir"],
                             verbose=True)

        # 0. Check if the sim_env is selected,
        #    default to SUMO, case insensitive
        sel_sim = ["sumo"] if not sel_sim else [sim.lower() for sim in sel_sim]

        # 1. Check simulator installation - mapping function
        sim_install = {
            "sumo": install_sumo,
            "vissim": None,
            "aimsun": None,
        }

        # 2. check if the simulator is installed, if not, install it
        print("\nChecking and installing the selected simulators:")
        kwargs['sel_dir'] = sel_dir

        kwargs['sumo_version'] = sumo_version
        kwargs['strict_sumo_version'] = strict_sumo_version

        kwargs['vissim_version'] = vissim_version
        kwargs['strict_vissim_version'] = strict_vissim_version

        kwargs['aimsun_version'] = aimsun_version
        kwargs['strict_aimsun_version'] = strict_aimsun_version

        kwargs['verbose'] = self.verbose

        for sim in sel_sim:
            try:
                sim_install.get(sim)(**kwargs)
                print()
            except Exception as e:
                print()
                print(f"  :Could not install {sim} due to error: {e}")

    def generate_abstract_scenario(self):
        """Generate the abstract scenario.
        """
        # 1. Generate the abstract scenario based on the input data
        self.abstract_scenario = AbstractScenario(self.input_config)
        self.abstract_scenario.update_AbstractScenario_from_input()
        print("  :Abstract Scenario successfully generated.")

    def generate_concrete_scenario(self):
        """Generate the concrete scenario.
        """

        # 1. Generate the concrete scenario based on abstract scenario
        # 2. Save the concrete scenario to the output directory

        if not hasattr(self, 'abstract_scenario'):
            print("  :Warning: Abstract Scenario is not generated yet. Please (generate_abstract_scenario).")
            return

        self.concrete_scenario = ConcreteScenario()
        self.concrete_scenario.get_unified_scenario(self.abstract_scenario)
        print("  :Concrete Scenario successfully generated.")

    def prepare_simulation(self,
                           start_time: float = 3600 * 8,
                           end_time: float = 3600 * 10,
                           seed: list | int = [101],
                           step_length: float = 0.1):
        """Simulate the concrete scenario.

        Args:
            start_time (float): The start time of the simulation. Default is 3600 * 8.
            end_time (float): The end time of the simulation. Default is 3600 * 10.
            seed (list or int): The seed for the simulation. Default is [101].
            step_length (float): The simulation step size. Default is 0.1.

        Examples:
            # import realtwin package
            >>> import realtwin as rt

            # load the input configuration file
            >>> twin = rt.REALTWIN(input_config_file="config.yaml", verbose=True)

            # check simulator is installed or not, default to SUMO
            >>> twin.env_setup(sel_sim=["SUMO"])

            # generate abstract scenario and concrete scenario
            >>> twin.generate_abstract_scenario()
            >>> twin.generate_concrete_scenario()

            # prepare simulation with start time, end time, seed, and step size
            >>> twin.prepare_simulation(start_time=3600 * 8, end_time=3600 * 10, seed=[101], step_length=0.1)

        Returns:
            None
        """

        # 1. prepare Simulate docs from the concrete scenario
        # 2. Save results to the output directory
        self.sim = SimPrep()
        self.sim.create_sumo_sim(self.concrete_scenario, start_time, end_time, seed, step_length)

    def calibrate(self, **kwargs):
        """Calibrate the simulation results.
        """

        if 'calibration_data' in kwargs:
            calibration_data = kwargs['calibration_data']
            print(calibration_data)

        # 1. Calibrate the simulation results
        # 2. Save the calibrated results to the output directory

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
