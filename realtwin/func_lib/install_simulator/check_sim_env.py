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
'''
##############################################################
# Created Date: Wednesday, December 18th 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
import pyufunc as pf
import subprocess
import re

from realtwin.utils_lib.find_exe_from_PATH import find_executable_from_PATH_on_win


# Check required simulation environments


def is_sumo_installed(*, ext: str = "exe", sel_dir: list = None, verbose: bool = True) -> bool | list:
    """Check if SUMO is installed onto the system.

    Args:
        ext (str): The extension of the executable. Defaults to "exe" for executable files.
        sel_dir (list): A list of directories to search for the SUMO executable. Defaults to None.
        verbose (bool): Whether to print the process info. Defaults to True.

    Raises:
        Exception: Unsupported OS, could not find SUMO executable

    Returns:
        bool or list: a list of installed versions, False otherwise.
    """

    # check the operation system
    if pf.is_windows():
        print("  :Checking SUMO installation on Windows.")
        sumo_executable = "sumo.exe"  # For Windows

    elif pf.is_linux():
        print("  :Checking SUMO installation on Linux.")
        sumo_executable = None  # TODO: Check name of the executable

    elif pf.is_mac():
        print("  :Checking SUMO installation on MacOS.")
        sumo_executable = None  # TODO: Check name of the executable

    else:
        raise Exception("  :Unsupported OS, could not find SUMO executable.")

    # Check if 'sumo' executable is in PATH: return None if not found
    sumo_exe_lst = find_executable_from_PATH_on_win(sumo_executable, ext=ext, sel_dir=sel_dir, verbose=False)

    if sumo_exe_lst:
        # remove duplicates
        sumo_exe_lst = list(set(sumo_exe_lst))

        # print out the version of SUMO if more than one path is found
        if len(sumo_exe_lst) > 1:
            print("  :Multiple SUMO executables found in the system PATH:")

        # run SUMO to check the version
        version_lst = []
        for exe_path in sumo_exe_lst:
            try:
                version_check = subprocess.run([exe_path],
                                               capture_output=True,
                                               text=True)

                if version_check.returncode == 0:
                    # find the version number from the output

                    # Define the pattern to match the version number
                    pattern = r'Version (\d+\.\d+\.\d+)'

                    # Search for the pattern in the text
                    match = re.search(pattern, version_check.stdout)

                    # Extract and print the version number if a match is found
                    if match:
                        version = match.group(1)
                        version_lst.append(str(version))
                        print(f"  :SUMO version: {version} found at: {exe_path}")
                    else:
                        pass

            except Exception as e:
                if verbose:
                    print(f"  :Error running SUMO: {e}")
        if version_lst:
            return list(set(version_lst))

    print("  :SUMO not found in the system PATH.")
    return False


def is_vissim_installed() -> bool:
    """Check if VISSIM is installed onto the system.

    Returns:
        bool: True if VISSIM is installed, False otherwise.
    """
    return False


def is_aimsun_installed() -> bool:
    """Check if AIMSUN is installed onto the system.

    Returns:
        bool: True if AIMSUN is installed, False otherwise.
    """
    return False
