'''
##############################################################
# Created Date: Wednesday, December 18th 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
import os
import shutil
import pyufunc as pf

# Check required simulation environments


def is_sumo_installed() -> bool:
    """Check if SUMO is installed onto the system.

    Returns:
        bool: True if SUMO is installed, False otherwise.
    """

    # check the operation system
    if pf.is_windows():
        sumo_executable = "sumo.exe"  # For Windows

    elif pf.is_linux():
        sumo_executable = "sumo.exe"  # TODO: Check the executable name

    elif pf.is_mac():
        sumo_executable = "sumo.app"  # TODO: Check the executable name

    else:
        raise Exception("  :Unsupported OS, could not find SUMO executable.")

    # Check if 'sumo' executable is in PATH
    sumo_path = shutil.which(sumo_executable)  # will return None if not found

    if sumo_path:
        print(f"SUMO is installed. Found at: {sumo_path}")
        return True

    print("SUMO is not installed or not in the system PATH.")
    return False


def is_vissim_installed() -> bool:
    """Check if VISSIM is installed onto the system.

    Returns:
        bool: True if VISSIM is installed, False otherwise.
    """
    return True


def is_aimsun_installed() -> bool:
    """Check if AIMSUN is installed onto the system.

    Returns:
        bool: True if AIMSUN is installed, False otherwise.
    """
    return True
