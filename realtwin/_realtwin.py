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

import os
import pyufunc as pf

from realtwin.utils_lib.create_venv import venv_create, venv_delete
from realtwin.func_lib.install_simulator.inst_sumo import install_sumo


class REALTWIN:
    """The real-twin developed by ORNL Applied Research and Mobility System (ARMS) group that
    enables the simulation of twin-structured cities.
    """

    def __init__(self, input_dir: str = "", **kwargs):
        """Initialize the REALTWIN object.

        Args:
            input_dir (str): The directory containing the input files.
            kwargs: Additional keyword arguments.
                output_dir (str): The directory to save the output files. Default is None.
        """
        self._input_dir = input_dir

        # check if the input_dir is empty
        if not self._input_dir:
            self._input_dir = pf.path2linux(os.getcwd())

        # check if the input directory exists
        if not os.path.exists(self._input_dir):
            raise FileNotFoundError(f"Input directory '{self._input_dir}' does not exist.")

        # check if output_dir in kwargs, if not set, the output_dir should be the same as input_dir
        if 'output_dir' in kwargs:
            self._output_dir = kwargs['output_dir']
            if not os.path.exists(self._output_dir):
                self._output_dir = pf.path2linux(os.path.join(self._input_dir, 'output'))
        else:
            self._output_dir = pf.path2linux(os.path.join(self._input_dir, 'output'))

        # add venv_create and delete as object methods
        self.venv_create = venv_create
        self.venv_delete = venv_delete
        self._venv_name = "rt_venv"

        # extract data from kwargs
        self.verbose = kwargs["verbose"] if "verbose" in kwargs else False

    def env_setup(self, *, sel_sim: list = None, create_venv: bool = False) -> None:
        """Check and set up the environment for the simulation.

        Args:
            sel_sim (list): select simulator to be set up. Default is None.
                Currently available options are ["SUMO", "VISSIM", "AIMSUN"].
            create_env (bool): Whether to create a virtual environment. Default is False.
        """

        # 0 create a virtual environment
        if create_venv:
            print(f"Creating a virtual environment: {self._venv_name}")
            self.venv_create(venv_name=self._venv_name,
                             venv_dir=self._output_dir,
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
        for sim in sel_sim:
            try:
                sim_install.get(sim)()
            except Exception as e:
                print(f"  :Could not install {sim} due to error: {e}")

        return None

    def load_inputs(self) -> None:
        """Load the input files.
        """

        #  load and verify the input files
        #  Print out / log the processing information
        #  such as: # of nodes, # of edges, # signalized intersections, # general control information, etc.

        #  1. load the network data

        #  2. load the traffic data

        #  3. load the controller data

        #  4. load the applications

        return None

    def generate_concrete_scenario(self) -> None:
        """Generate the concrete scenario.
        """

        # 1. Generate the concrete scenario based on the input data
        # 2. Save the concrete scenario to the output directory

        return None

    def simulate(self) -> None:
        """Simulate the concrete scenario.
        """

        # 1. Simulate the concrete scenario
        # 2. Save the simulation results to the output directory

        return None

    def calibration(self, **kwargs) -> None:
        """Calibrate the simulation results.
        """

        if 'calibration_data' in kwargs:
            calibration_data = kwargs['calibration_data']
            print(calibration_data)

        # 1. Calibrate the simulation results
        # 2. Save the calibrated results to the output directory

        return None

    def post_process(self) -> None:
        """Post-process the simulation results.
        """

        # 1. Post-process the simulation results
        # 2. Save the post-processed results to the output directory

        return None

    def visualization(self) -> None:
        """Visualize the simulation results.
        """

        # 1. Visualize the simulation results
        # 2. Save the visualization results to the output directory

        return None
