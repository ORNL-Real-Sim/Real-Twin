##############################################################################
# Copyright (c) 2024-, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################
import os
import zipfile
import pyufunc as pf

from realtwin.func_lib._a_install_simulator.check_sim_env import is_aimsun_installed


def install_aimsun(sel_dir: list | None = None,
                #    strict_aimsun_version: str = "Next 23",
                   verbose: bool = True,
                   **kwargs) -> bool:
    """Install the Aimsun simulator.

    Args:
        sel_dir (list): A list of directories to search for the Aimsun executable. Defaults to None.
        #    strict_aimsun_version (bool): If True, check and install the exact version of Aimsun. Default is Next 23
        verbose (bool): If True, print the installation process. Default is True.
        kwargs: Additional keyword arguments.

    Returns:
        bool: True if the Aimsun is installed successfully, False otherwise
    """

    # check sel_dir is a list
    if not isinstance(sel_dir, (list, type(None))):
        raise ValueError("sel_dir should be a list.")  # noqa: TRY004

    # Check if Aimsun is already installed
    version_lst = is_aimsun_installed(sel_dir=sel_dir, verbose=verbose)
    if version_lst:
        # Check if the exact version of Aimsun is installed
        # if strict_aimsun_version is None or strict_aimsun_version in version_lst:
            print(
                f"  :Aimsun is already installed, available versions: {version_lst}")
            return True

    print("  :Error: Unsupported operating system.")
    return False
