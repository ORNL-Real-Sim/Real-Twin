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

"""Control of module imports from func_lib."""

# _a_install_simulator: Functions to install and check simulation environments
from realtwin.func_lib._a_install_simulator.check_sim_env import (is_sumo_installed,
                                                                  is_vissim_installed,
                                                                  is_aimsun_installed)
from realtwin.func_lib._a_install_simulator.inst_sumo import (install_sumo,
                                                              install_sumo_windows,
                                                              install_sumo_linux,
                                                              install_sumo_macos)

# _b_load_inputs: Functions to load input configurations and data
from realtwin.func_lib._b_load_inputs.loader_config import load_input_config, get_bounding_box_from


# _c_abstract_scenario: Abstract classes and functions for scenarios
from realtwin.func_lib._c_abstract_scenario._abstractScenario import (AbstractScenario,
                                                                      load_traffic_volume,
                                                                      load_traffic_turning_ratio,
                                                                      load_control_signal)

from realtwin.func_lib._c_abstract_scenario._network import Network, OpenDriveNetwork, OSMRoad
from realtwin.func_lib._c_abstract_scenario._traffic import Traffic
from realtwin.func_lib._c_abstract_scenario._control import Control
from realtwin.func_lib._c_abstract_scenario._application import Application

# _d_concrete_scenario: Concrete implementations of scenarios
from realtwin.func_lib._d_concrete_scenario._concreteScenario import ConcreteScenario
from realtwin.func_lib._d_concrete_scenario._supply import Supply
from realtwin.func_lib._d_concrete_scenario._demand import Demand
from realtwin.func_lib._d_concrete_scenario._behavior import Behavior
from realtwin.func_lib._d_concrete_scenario._route import Route
from realtwin.func_lib._d_concrete_scenario._trafficControl import TrafficControl

# _e_simulation: Functions to prepare and run simulations
from realtwin.func_lib._e_simulation._generate_simulation import SimPrep
from realtwin.func_lib._e_simulation._sumo import SUMOPrep
from realtwin.func_lib._e_simulation._aimsun import AimsunPrep
from realtwin.func_lib._e_simulation._vissim import VissimPrep

# _f_calibration: Calibration functions and algorithms for different simulators
from realtwin.func_lib._f_calibration.calibration_sumo import cali_sumo
from realtwin.func_lib._f_calibration.calibration_aimsun import cali_aimsun
from realtwin.func_lib._f_calibration.calibration_vissim import cali_vissim

from realtwin.func_lib._f_calibration.algo_sumo.cali_behavior import BehaviorCali
from realtwin.func_lib._f_calibration.algo_sumo.cali_turn_inflow import TurnInflowCali


__all__ = [
    # _a_install_simulator
    "is_sumo_installed",
    "is_vissim_installed",
    "is_aimsun_installed",
    "install_sumo",
    "install_sumo_windows",
    "install_sumo_linux",
    "install_sumo_macos",

    # _b_load_inputs
    "load_input_config",
    "get_bounding_box_from",

    # _c_abstract_scenario
    "AbstractScenario",
    "load_traffic_volume",  # Load traffic volume data
    "load_traffic_turning_ratio",  # Load traffic turning ratio data
    "load_control_signal",  # Load control signal data
    # "Network",  # Network class for road networks
    "OpenDriveNetwork",  # OpenDrive network class for OpenDrive format
    "OSMRoad",  # OSMRoad class for OpenStreetMap road representation
    # "Traffic",
    # "Control",
    # "Application",

    # _d_concrete_scenario
    "ConcreteScenario",
    # "Supply",  # Supply class for concrete supply scenarios
    # "Demand",  # Demand class for concrete demand scenarios
    # "Behavior",  # Behavior class for concrete behavior scenarios
    # "Route",  # Route class for concrete route scenarios
    # "TrafficControl",  # TrafficControl class for concrete traffic control scenarios

    # _e_simulation
    "SimPrep",  # Simulation preparation class
    "SUMOPrep",  # SUMO simulation preparation class
    "AimsunPrep",  # Aimsun simulation preparation class
    "VissimPrep",  # Vissim simulation preparation class

    # _f_calibration
    "cali_sumo",  # Calibration function for SUMO simulator
    "cali_aimsun",  # Calibration function for Aimsun simulator
    "cali_vissim",  # Calibration function for Vissim simulator
    "BehaviorCali",  # Behavior calibration algorithm for SUMO simulator
    "TurnInflowCali"  # Turn inflow calibration algorithm for SUMO simulator

    # _g_analyzer

    # _h_visualization
]
