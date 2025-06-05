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
"""Control of module imports for the RealTwin function library."""

import os
import zipfile
import pyufunc as pf

from realtwin.func_lib._a_install_simulator.check_sim_env import is_sumo_installed
from realtwin.util_lib.download_file_from_web import download_single_file_from_web


def install_sumo(sel_dir: list = None,
                 strict_sumo_version: str = "1.21.0",
                 verbose: bool = True,
                 **kwargs) -> bool:
    """Install the SUMO simulator.

    Args:
        sel_dir (list): A list of directories to search for the SUMO executable. Defaults to None.
        strict_sumo_version (bool): If True, check and install the exact version of SUMO. Default is 1.21.0
        verbose (bool): If True, print the installation process. Default is True.
        kwargs: Additional keyword arguments.

    Returns:
        bool: True if the SUMO is installed successfully, False otherwise
    """

    # check sel_dir is a list
    if not isinstance(sel_dir, (list, type(None))):
        raise ValueError("sel_dir should be a list.")

    # Check if SUMO is already installed
    version_lst = is_sumo_installed(sel_dir=sel_dir, verbose=verbose)
    if version_lst:
        # Check if the exact version of SUMO is installed
        if strict_sumo_version is None or strict_sumo_version in version_lst:
            print(f"  :SUMO is already installed, available versions: {version_lst}")
            return True

        print(f"\n  :Installing strict_sumo_version SUMO version {strict_sumo_version} "
              f"(Available versions: {version_lst})...")

    # If SUMO not installed,
    # Or strict_sumo_version is True and the version is not installed
    # Install the SUMO
    if pf.is_windows():
        return install_sumo_windows(strict_sumo_version, verbose=verbose)

    if pf.is_linux():
        print("  :Error: Linux is not supported yet.")
        return False

    if pf.is_mac():
        print("  :Error: MacOS is not supported yet.")
        return False

    print("  :Error: Unsupported operating system.")
    return False


def install_sumo_windows(sumo_version: str = "1.21.0", verbose: bool = True) -> bool:
    """Install SUMO onto the windows system.

    Args:
        sumo_version (str): The version of SUMO to be installed. Default is "1.21.0".
        verbose (bool): If True, print the installation process. Default is True.

    Returns:
        bool: True if the SUMO is installed successfully, False otherwise
    """

    print(f"  :Installing SUMO {sumo_version} for Windows...")

    # Download SUMO from the official website
    sumo_release_url = "https://sumo.dlr.de/releases/"
    sumo_version_win = f"sumo-win64-{sumo_version}.zip"

    download_path = pf.path2linux(os.path.join(os.getcwd(), "sumo.zip"))
    extract_path = pf.path2linux(os.path.join(os.getcwd(), "SUMO"))

    # Download the SUMO zip file from the official website
    sumo_zip_url = f"{sumo_release_url}{sumo_version}/{sumo_version_win}"
    id_download = download_single_file_from_web(sumo_zip_url, download_path)

    if not id_download:
        return False

    # Extract the SUMO zip file
    if verbose:
        print(f"  :Extracting SUMO {sumo_version} for Windows at: {extract_path}...")

    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Clean up the downloaded zip file
    os.remove(download_path)

    # check if SUMO bin folder exists
    sumo_bin_path = os.path.join(extract_path, f"sumo-{sumo_version}", "bin")
    sumo_bin_path = pf.path2linux(sumo_bin_path)
    if not os.path.exists(sumo_bin_path):
        print(f"  :Error: bin folder not found in extracted SUMO directory: {sumo_bin_path}")
        return False

    # Add the SUMO bin folder to the system PATH
    if sumo_bin_path not in os.environ['PATH']:

        os.environ["PATH"] += os.pathsep + sumo_bin_path[0].upper() + sumo_bin_path[1:]
        os.environ["PATH"] += os.pathsep + sumo_bin_path[0].lower() + sumo_bin_path[1:]

        # add_path = subprocess.run(["setx", "PATH", f"%PATH%;{sumo_bin_path}"], shell=True, check=True)
        # if add_path.returncode == 0:
        #     print("  :SUMO is installed successfully.")
        # else:
        #     print("  :Error: Failed to add SUMO bin folder to system PATH.")

    return True


def install_sumo_linux(sumo_version: str = "1.21.0", verbose: bool = True) -> bool:
    """Install SUMO onto the linux system.

    Args:
        sumo_version (str): The version of SUMO to be installed. Default is "1.21.0".
        verbose (bool): If True, print the installation process. Default is True.

    Note:
        The installation of SUMO on Linux is not supported yet.
        upcoming feature.
    Returns:
        bool: True if the SUMO is installed successfully on Linux, False otherwise
    """
    print(f"  :Installing SUMO {sumo_version} for Linux system...")
    print("verbose", verbose)
    print("  :Warning: Install SUMO on Linux is not supported yet.")

    return False


def install_sumo_macos(sumo_version: str = "1.21.0", verbose: bool = True) -> bool:
    """Install SUMO onto the mac system.

    Args:
        sumo_version (str): The version of SUMO to be installed. Default is "1.21.0".
        verbose (bool): If True, print the installation process. Default is True.

    Note:
        The installation of SUMO on Mac is not supported yet.
        upcoming feature.

    Returns:
        bool: True if the SUMO is installed successfully on Mac, False otherwise
    """
    print(f"  :Installing SUMO {sumo_version} for Mac system...")
    print("verbose", verbose)
    print("  :Warning: Install SUMO on Mac is not supported yet.")

    return False
