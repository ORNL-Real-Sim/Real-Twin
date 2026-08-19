
##############################################################################
# Copyright (c) 2024-, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Email: realtwin@ornl.gov                                                   #
##############################################################################

"""Control of module imports for the RealTwin package."""

import sys
from realtwin._realtwin import RealTwin
from realtwin.func_lib._a_install_simulator.check_sim_env import (is_sumo_installed,
                                                                  is_vissim_installed,
                                                                  is_aimsun_installed)
from realtwin.func_lib import (install_sumo,
                               SimPrep,
                               BehaviorCali,
                               TurnInflowCali,
                               cali_sumo,
                               cali_aimsun,
                               cali_vissim,
                               load_input_configs)

from realtwin.util_lib.create_venv import venv_create, venv_delete
from realtwin.util_lib.create_config import prepare_realtwin_configs
from realtwin.util_lib import get_bbox_from_vertices

# Autonomous vehicle simulation
from realtwin.autonomous_veh import SimAV, prepare_av_configs, load_av_configs

from realtwin.rt_aimsun import RealTwinAimsun


__version__ = '0.1.2'

"The minimum required Python version for RealTwin is 3.10"

__all__ = [
    'RealTwin',
    'prepare_realtwin_configs',
    "load_input_configs",

    # util_lib
    'venv_create', 'venv_delete',
    'is_sumo_installed', 'is_vissim_installed', 'is_aimsun_installed',
    "get_bbox_from_vertices",

    # func_lib
    'install_sumo', 'SimPrep', 'BehaviorCali', 'TurnInflowCali', 'cali_sumo', 'cali_aimsun', 'cali_vissim',

    # autonomous_veh
    'SimAV',
    'prepare_av_configs',
    'load_av_configs',

    # rt_aimsun
    'RealTwinAimsun'
]


def check_python_version(min_version: str = "3.10") -> tuple:
    """ Check if the current Python version meets the minimum requirement."""
    # Split the version string and convert to tuple of integers
    # version_tuple = tuple(map(int, sys.version.split()[0].split('.')))
    version_tuple = tuple(int(val) for val in sys.version.split()[0].split('.'))

    # Check if the version is greater than or equal to the minimum version required
    major, minor = min_version.split(".")
    try:
        if version_tuple < (int(major), int(minor)):
            raise EnvironmentError(f"Python version {min_version} or higher is required.")
    except Exception:
        print(f"pyufunc supports Python {min_version} or higher.")
    return version_tuple


check_python_version(min_version="3.10")
