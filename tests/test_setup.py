'''
##############################################################
# Created Date: Monday, December 23rd 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


import os
import sys
from pathlib import Path


def add_pkg_to_sys_path(pkg_name: str, verbose: bool = True) -> None:
    """Automatically finds an importable Python package by its name in the current directory,
    parent directories, or child directories, and adds it to the system path.
    This is useful when writing test functions and needing to import a package that is not installed.

    Args:
        package_name (str): The name of the package to locate and add.
        verbose (bool): Whether to print the process info. Defaults to True.

    Location:
        pyufunc/util_pathio/_path.py

    Examples:
        >>> import pyufunc as pf
        >>> pf.add_pkg_to_sys_path('my_package', False)

    Returns:
        None: None

    """

    # TDD: check if the package path is a string
    if not isinstance(pkg_name, str):
        raise ValueError("pkg_path should be a string.")

    # Helper function to check if a directory is an importable Python package
    def is_importable_package(path):
        return os.path.isdir(path) and (
            # Check for traditional package
            os.path.isfile(os.path.join(path, '__init__.py'))
            # Accept single-module package
            or any(fname.endswith('.py') for fname in os.listdir(path))
        )

    # Helper function to locate the package in the directory tree
    def locate_package(start_path, package_name):
        for root, dirs, _ in os.walk(start_path):
            if package_name in dirs:
                package_path = os.path.join(root, package_name)
                if is_importable_package(package_path):
                    return package_path
        return None

    # Start searching in the current directory and parent directories
    current_dir = os.getcwd()
    while True:
        # Look for the package in the current directory and its subdirectories
        package_path = locate_package(current_dir, pkg_name)
        if package_path:
            break
        # Move to the parent directory
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Stop if we've reached the root
            break
        current_dir = parent_dir

    # If found, add to the system path
    if package_path:
        absolute_path = Path(os.path.abspath(package_path))
        absolute_path_parent = absolute_path.parent.absolute()

        if absolute_path_parent not in sys.path:
            sys.path.insert(0, str(absolute_path_parent))
            if verbose:
                print(f"Added {absolute_path_parent} to system path.")
        else:
            if verbose:
                print(f"{absolute_path_parent} is already in the system path.")
        return absolute_path

    print(f"  :Importable package '{pkg_name}' not found"
          "in the current directory, parent directories, or children.")
