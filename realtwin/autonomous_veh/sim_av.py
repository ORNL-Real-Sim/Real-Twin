##############################################################################
# Copyright (c) 2024--, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################

"""Autonomous Vehicle Simulation in SUMO."""

from pathlib import Path
import shutil
import pyufunc as pf

from ._load_config import load_av_configs
from ._sim_av_util import (create_veh_type_attributes,
                           create_sumo_rou_xml,
                           update_sumo_flow_xml,
                           add_veh_types_to_rou,
                           create_sumo_config,
                           run_sumo_simulation)
from ._generate_loop_detector import generate_sumo_loop_detector_add_xml


def prepare_av_configs(dest_dir: str = "") -> bool:
    """Generate configuration files for autonomous vehicles simulation.

    Args:
        dest_dir (str): Directory where the configuration files will be saved.
        If the directory does not exist, use the current working directory.

    Example:
        >>> from realtwin import prepare_av_configs
        >>> prepare_av_configs(dest_dir="path/to/directory")
        >>> "Autonomous vehicle configuration file created at: path/to/directory/AVConfig.yml"
        >>> True

    Returns:
        bool: True if the configuration files were successfully created, False otherwise.
    """

    # check if output directory exists
    if not Path(dest_dir).exists():
        dest_dir = Path.cwd()

    # copy the default configuration file to the output directory
    default_config = Path(__file__).parent.parent / 'data_lib/config_av.yaml'
    fname = Path(dest_dir) / 'config_av.yaml'
    shutil.copy(default_config, fname)
    print(f"  :Autonomous vehicle configuration file created at: {pf.path2linux(fname)}")
    return True


class SimAV:
    """Autonomous Vehicle Simulation

    See Also:
        "config_file": you can run `prepare_av_configs` to generate a default configuration file.
        for details, refer to the `prepare_av_configs` function in the `realtwin` module.
    """

    def __init__(self, path_config: str | Path = "", verbose: bool = True):
        """Initialize the SimAV class.

        Args:
            path_config (str | Path): Path to the configuration file for autonomous vehicle simulation.
            verbose (bool): If True, print detailed information during the simulation.

        See Also:
            "config_file": you can run `prepare_av_configs` to generate a default configuration file.
        """

        # TDD
        if isinstance(path_config, (str, Path)):
            self.path_config = Path(path_config)
        else:
            raise ValueError("  :Error: path_config must be a string or Path object.")

        if not self.path_config.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.path_config}")

        # self.av_config = load_av_configs(self.path_config)
        self.verbose = verbose

    def prepare_av_configs(self, dest_dir: str = "") -> bool:
        """Generate configuration files for autonomous vehicles simulation.

        Args:
            dest_dir (str): Directory where the configuration files will be saved.
            If the directory does not exist, use the current working directory.

        Example:
            >>> from realtwin import prepare_av_configs
            >>> prepare_av_configs(dest_dir="path/to/directory")
            >>> "Autonomous vehicle configuration file created at: path/to/directory/AVConfig.yml"
            >>> True

        Returns:
            bool: True if the configuration files were successfully created, False otherwise.
        """

        return prepare_av_configs(dest_dir)

    def load_config(self, path_config: str) -> bool:
        """Load the configuration file for autonomous vehicles simulation.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            dict: Loaded configuration data.
        """
        try:
            self.av_config = load_av_configs(path_config)
            return True
        except Exception as e:
            print(f"  :Error loading configuration data: {e}")
        return False

    def run_simulation(self, av_config: str | Path | dict = None) -> bool:
        """Run the autonomous vehicle simulation."""

        # check if av_config is provided
        if isinstance(av_config, (str, Path)):
            # load configuration from file
            self.av_config = load_av_configs(av_config)
        elif isinstance(av_config, dict):
            # use the provided configuration dictionary
            self.av_config = av_config

        if not hasattr(self, 'av_config'):
            try:
                self.av_config = load_av_configs(self.path_config)
            except Exception as e:
                raise Exception(
                    "  :Error: No AV configuration provided or loaded."
                    " Please run `load_config` or provide a valid config file or dictionary."
                ) from e

        # check if required inputs are valid
        if not check_inputs_from_config(self.av_config):
            raise Exception("  :Error: Invalid inputs in the configuration file,"
                            " please check the input directory and files.")

        # get paths from the configuration
        input_dir = self.av_config['input']['input_dir']
        path_net = Path(input_dir) / self.av_config['input']['net_file']
        path_rou = Path(f"{name_without_suffixes(path_net)}.rou.xml")
        path_flow = Path(input_dir) / self.av_config['input']['flow_file']
        path_turn = Path(input_dir) / self.av_config['input']['turn_file']

        # create/update output directory
        output_dir = Path(input_dir) / 'output_AV'
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # copy input files to output directory
        path_net_output = output_dir / path_net.name
        path_rou_output = output_dir / path_rou.name
        path_flow_output = output_dir / path_flow.name
        path_turn_output = output_dir / path_turn.name
        shutil.copy(path_net, path_net_output)
        # shutil.copy(path_rou, path_rou_output)
        shutil.copy(path_flow, path_flow_output)
        shutil.copy(path_turn, path_turn_output)

        # generate loop detector add XML file
        path_add = output_dir / 'detector.add.xml'
        if not path_add.exists():
            generate_sumo_loop_detector_add_xml(path_net_output,
                                                detector_type="E1",
                                                add_fname=path_add.name,
                                                dest_dir=output_dir)

        # create vehicle type attributes
        veh_types = self.av_config['veh_types']
        veh_type_attributes = create_veh_type_attributes(self.av_config)

        # update SUMO flow XML with new vehicle types
        update_sumo_flow_xml(path_flow_output, veh_types, veh_type_attributes)

        # create SUMO route XML
        create_sumo_rou_xml(path_net_output, path_flow_output, path_turn_output, path_rou_output)

        # add vehicle types to the route file
        add_veh_types_to_rou(path_rou_output, veh_types, veh_type_attributes)

        # create SUMO configuration file
        path_cfg = output_dir / 'SimAV_config.sumocfg'
        create_sumo_config(path_cfg,
                           sim_name=self.av_config.get('sim_name', 'chatt'),
                           sim_time=self.av_config.get('sim_time', 3600))

        # run simulation
        run_sumo_simulation(path_cfg, sim_time=self.av_config.get('sim_time', 3600))
        print(f"  :Simulation completed successfully. Output files are saved in: {output_dir}")
        return True


def check_inputs_from_config(av_config: dict) -> bool:
    """Check if the required inputs are present in the configuration"""

    input_dict = av_config.get('input', {})

    if not input_dict:
        print("  :Warning: No inputs found in the configuration.")
        return False

    input_dir = input_dict.get('input_dir', '')
    if not input_dir:
        print("  :Warning: Input directory not specified in the configuration.")
        return False
    if not Path(input_dir).exists():
        print(f"  :Warning: Input directory does not exist: {input_dir}")
        return False
    if not Path(input_dir).is_dir():
        print(f"  :Warning: Input path is not a directory: {input_dir}")
        return False
    print(f"  :Input directory is valid: {input_dir}")

    # check net, flow, rou and turn files
    net_file = input_dict.get('net_file', '')
    if not net_file:
        print("  :Warning: Network file not specified in the configuration.")
        return False
    path_net = Path(input_dir) / net_file
    if not path_net.exists():
        print(f"  :Warning: Network file does not exist: {path_net}")
        return False

    # rou_file = input_dict.get('rou_file', '')
    # if not rou_file:
    #     print("  :Warning: Route file not specified in the configuration.")
    #     return False
    # path_rou = Path(input_dir) / rou_file
    # if not path_rou.exists():
    #     print(f"  :Warning: Route file does not exist: {path_rou}")
    #     return False

    flow_file = input_dict.get('flow_file', '')
    if not flow_file:
        print("  :Warning: Flow file not specified in the configuration.")
        return False
    path_flow = Path(input_dir) / flow_file
    if not path_flow.exists():
        print(f"  :Warning: Flow file does not exist: {path_flow}")
        return False

    turn_file = input_dict.get('turn_file', '')
    if not turn_file:
        print("  :Warning: Turn file not specified in the configuration.")
        return False
    path_turn = Path(input_dir) / turn_file
    if not path_turn.exists():
        print(f"  :Warning: Turn file does not exist: {path_turn}")
        return False
    print("  :All input files are valid.")
    return True


def name_without_suffixes(p: Path) -> str:
    # peel off each suffix until none remain
    while p.suffix:
        p = p.with_suffix('')
    return p.name
