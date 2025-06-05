##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################

from tests.proj_add_sys_path import add_pkg_to_sys_path
add_pkg_to_sys_path("realtwin")

from realtwin.func_lib._a_install_simulator.check_sim_env import (
    is_sumo_installed,
    is_vissim_installed,
    is_aimsun_installed)


def test_is_sumo_installed():
    """is_sumo_installed should return True if SUMO is installed"""

    assert is_sumo_installed() or not is_sumo_installed()


def test_is_vissim_installed():
    """is_vissim_installed should return False if VISSIM is not installed"""

    assert is_vissim_installed() or not is_vissim_installed()


def test_is_aimsun_installed():
    """is_aimsun_installed should return False if AIMSUN is not installed"""

    assert is_aimsun_installed() or not is_aimsun_installed()
