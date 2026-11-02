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

from realtwin.func_lib._a_install_simulator.check_sim_env import is_vissim_installed


def install_vissim(sel_dir: list | None = None,
                   verbose: bool = True,
                   **kwargs) -> bool:
    """Install the Vissim simulator.

    Args:
        sel_dir (list): A list of directories to search for the Vissim executable. Defaults to None.
        verbose (bool): If True, print the installation process. Default is True.
        kwargs: Additional keyword arguments.

    Returns:
        bool: True if the Vissim is installed successfully, False otherwise
    """

    # check sel_dir is a list
    if not isinstance(sel_dir, (list, type(None))):
        raise ValueError("sel_dir should be a list.")  # noqa: TRY004


    # Check if Vissim is already installed
    version_lst = is_vissim_installed(sel_dir=sel_dir, verbose=verbose)
    if version_lst:
        # Check if the exact version of Vissim is installed
        # if strict_aimsun_version is None or strict_aimsun_version in version_lst:
        print(
            f"  :Vissim is already installed, available versions: {version_lst}")
        return version_lst

    print("  :Error: Unsupported operating system.")
    return False
