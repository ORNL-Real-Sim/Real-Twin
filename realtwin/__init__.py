
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
"""Control of module imports for the RealTwin package."""

from realtwin._realtwin import RealTwin
from realtwin.utils_lib.create_venv import venv_create, venv_delete
from realtwin.func_lib._a_install_simulator.check_sim_env import (is_sumo_installed,
                                                                  is_vissim_installed,
                                                                  is_aimsun_installed)

__version__ = '0.1.0'

"The minimum required Python version for RealTwin is 3.10"

__all__ = [
    'RealTwin',

    # utils_lib.check_env
    'is_sumo_installed', 'is_vissim_installed', 'is_aimsun_installed',

    # utils_lib.create_venv
    'venv_create', 'venv_delete',

    # func_lib
]
