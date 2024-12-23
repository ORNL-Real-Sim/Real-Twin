'''
##############################################################
# Created Date: Friday, December 20th 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
import os
import subprocess
import shutil
import pyufunc as pf


def venv_create(*, env_name: str = "", folder_path: str = "", verbose: bool = True) -> None:
    """Create a virtual environment in the specified folder with the specified name.

    Args:
        env_name (str): the name of the virtual environment
        folder_path (str): the path to the folder where the virtual environment will be created
        verbose (bool): whether to print the progress

    Raises:
        Exception: if env_name is not a string, or folder_path is not a string

    Returns:
        None
    """
    # Default values for env_name and folder_path if not provided
    if not env_name:
        env_name = "env_rt"

    if not folder_path:
        folder_path = os.getcwd()

    # TDD for env_name
    if not isinstance(env_name, str):
        raise Exception("env_name must be a string")
    if not isinstance(folder_path, str):
        raise Exception("folder_path must be a string")

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Full path to the virtual environment
    env_path = os.path.join(folder_path, env_name)

    # Create the virtual environment
    subprocess.run(["python", "-m", "venv", env_path])

    if verbose:
        print(f"Virtual environment '{env_name}' created at: '{env_path}")

    # Activate the virtual environment
    if pf.is_windows():
        activate_path = pf.path2linux(os.path.join(env_path, "Scripts", "activate"))
        subprocess.run([activate_path], shell=True)

    elif pf.is_linux():
        activate_path = os.path.join(env_path, "bin", "activate")
        subprocess.run(["source", activate_path], shell=True)

    elif pf.is_mac():
        pass

    else:
        print("  :OS not supported")
        return

    if verbose:
        print(f"  :Virtual environment '{env_name}' activated")

    # install the realtwin package in the virtual environment
    exe_pip = subprocess.run(["pip", "install", "realtwin"])
    if exe_pip.returncode != 0:
        print("  :Failed to install the package realtwin in the virtual environment")
        return None
    if verbose:
        print("  :Package realtwin installed in the virtual environment")

    return None


def venv_delete(*, env_name: str = "", folder_path: str = "", verbose: bool = True) -> None:
    """Delete the virtual environment in the specified folder with the specified name.

    Args:
        env_name (str): the name of the virtual environment
        folder_path (str): the path to the folder where the virtual environment will be deleted
        verbose (bool): whether to print the progress

    Returns:
        None
    """
    # Full path to the virtual environment
    env_path = os.path.join(folder_path, env_name)

    # Delete the virtual environment
    shutil.rmtree(env_path)

    if verbose:
        print(f"Virtual environment '{env_name}' deleted from '{env_path}")

    return None
