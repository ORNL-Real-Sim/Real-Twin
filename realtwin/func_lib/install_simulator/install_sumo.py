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
# Created Date: Monday, December 23rd 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
import os
import subprocess
from urllib import request
import zipfile


def install_sumo_windows(sumo_version: str = "1.20.0", verbose: bool = True) -> bool:
    """Install SUMO onto the windows system.

    Returns:
        bool: True if the SUMO is installed successfully, False otherwise
    """
    # Download SUMO from the official website
    sumo_release_url = "https://sumo.dlr.de/releases/"
    sumo_version_win = f"sumo-win64-{sumo_version}.zip"

    download_path = os.path.join(os.getcwd(), "sumo.zip")
    extract_path = os.path.join(os.getcwd(), "SUMO")

    # Download the SUMO zip file from the official website
    if verbose:
        print(f"  :Downloading SUMO {sumo_version} for Windows...")

    request.urlretrieve(sumo_release_url + sumo_version_win, download_path)

    # Extract the SUMO zip file
    if verbose:
        print(f"  :Extracting SUMO {sumo_version} for Windows at: {extract_path}...")

    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Clean up the downloaded zip file
    os.remove(download_path)

    # check if SUMO bin folder exists
    sumo_bin_path = os.path.join(extract_path, sumo_version, "bin")
    if not os.path.exists(sumo_bin_path):
        print(f"Error: bin folder not found in extracted SUMO directory: {sumo_bin_path}")
        return False

    # Add the SUMO bin folder to the system PATH
    if sumo_bin_path not in os.environ['PATH']:
        add_path = subprocess.run(["setx", "PATH", f"%PATH%;{sumo_bin_path}"], shell=True, check=True)
        if add_path.returncode == 0:
            print("  :SUMO is installed successfully.")
        else:
            print("  :Error: Failed to add SUMO bin folder to system PATH.")

    return True


def install_sumo_linux() -> bool:
    """Install SUMO onto the linux system.

    Returns:
        bool: True if the SUMO is installed successfully on Linux, False otherwise
    """

    return False
