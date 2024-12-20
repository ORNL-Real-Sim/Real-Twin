'''
##############################################################
# Created Date: Friday, December 20th 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
import os
import subprocess
import pyufunc as pf


def create_virtualenv(folder_path: str, env_name: str, verbose: bool = True) -> None:
    """Create a virtual environment in the specified folder with the specified name.

    Args:
        folder_path (str): the path to the folder where the virtual environment will be created
        env_name (str): the name of the virtual environment

    Returns:
        None
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Full path to the virtual environment
    env_path = os.path.join(folder_path, env_name)

    # Create the virtual environment
    subprocess.run(["python", "-m", "venv", env_path])

    if verbose:
        print(f"Virtual environment '{env_name}' created at '{env_path}")

    # Activate the virtual environment
    activate_path = pf.path2linux(os.path.join(env_path, "Scripts", "activate"))

    if pf.is_windows():
        subprocess.run([activate_path], shell=True)
    elif pf.is_linux():
        subprocess.run(["source", activate_path], shell=True)
    elif pf.is_mac():
        pass
        # activate_path = os.path.join(env_path, "bin", "activate")
    else:
        print("OS not supported")
        return

    if verbose:
        print(f"Virtual environment '{env_name}' activated")

    # install the required packages, install the realtwin with dependencies
    subprocess.run(["pip", "install", "realtwin"])

    if verbose:
        print("Required packages installed in the virtual environment")

    return None
