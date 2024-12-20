import os
from pathlib import Path
import pandas as pd


class REALTWIN(object):
    """The real-twin developed by ORNL Applied Research and Mobility System (ARMS) group that
    enables the simulation of twin-structured cities.
    """

    def __init__(self, input_dir: str = "", **kwargs) -> None:
        """Initialize the REALTWIN object.

        Args:
            input_dir (str): The directory containing the input files.
            output_dir (str): The directory to save the output files. Default is None.

        Returns:
            None
        """
        self._input_dir = input_dir

        # TDD
        if not isinstance(input_dir, str):
            raise Exception("input_dir must be a string")

        # check if the input_dir is empty
        if not self._input_dir:
            self._input_dir = os.getcwd()

        # check if the input directory exists
        if not os.path.exists(self._input_dir):
            raise FileNotFoundError(f"Input directory '{self._input_dir}' does not exist.")

        # check if output_dir in kwargs, if not set, the output_dir should be the same as input_dir
        if 'output_dir' in kwargs:
            self._output_dir = kwargs['output_dir']
            if not os.path.exists(self._output_dir):
                self._output_dir = os.path.join(self._input_dir, 'output')
        else:
            self._output_dir = os.path.join(self._input_dir, 'output')

    def env_setup(self, env_ctrl: list = ["SUMO"]) -> None:
        """Set up the environment for the simulation.

        Args:
            env_ctrl (list): The list of simulation environments to be set up. Default is ["SUMO"].
        """

        # 1. Check if SUMO is installed onto the system,
        # if not, run setup_sumo to install SUMO

        # 2. Check if VISSIM is installed onto the system,
        # if not, warn the user to install VISUM, or some functionalities will not be available

        # 3. Check if AIMSUN is installed onto the system,
        # if not, warn the user to install AIMSUN, or some functionalities will not be available

        # 4. Future extension: Additional simulation env setup

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