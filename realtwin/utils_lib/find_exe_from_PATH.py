'''
##############################################################
# Created Date: Tuesday, December 31st 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
import os
import pyufunc as pf


def find_executable_from_PATH_on_win(exe_name: str,
                                     ext: str = "exe",
                                     sel_dir: list = None,
                                     verbose: bool = True) -> list | None:
    """Find the executable from the system PATH.

    Args:
        exe_name (str): The executable name to search for.
        ext (str): The extension of the executable. Defaults to "exe" for executable files.
        sel_dir (list): A list of directories to search for the executable. Defaults to [].
        verbose (bool): Whether to print the process info. Defaults to True.

    Returns:
        list or None: A list of full path of the executable if found, otherwise None.
    """

    # check if the executable name is a string
    if not isinstance(exe_name, str):
        raise ValueError("exe_name should be a string.")

    # check if extension is str
    if not isinstance(ext, str):
        raise ValueError("ext should be a string.")

    # check if sel_dir is a list
    if not isinstance(sel_dir, (list, type(None))):
        raise ValueError("sel_dir should be a list.")

    # Add the selected directories to the PATH environment
    if sel_dir:
        for path in sel_dir:
            if os.path.isdir(path):
                os.environ["PATH"] += os.pathsep + path
            elif verbose:
                print(f"  :The directory: {path} does not exist. Skipped.")

    # check if exe_name has the extension
    _, ext_str = os.path.splitext(exe_name)
    if not ext_str:
        if verbose:
            print(f"  :The executable: {exe_name} has no extension. Added {ext} as the extension.")
        exe_name = f"{exe_name}.{ext}"

    # get the full environment PATH
    env_paths = os.getenv("PATH").split(os.pathsep)

    res = []
    for path in env_paths:
        abs_path = pf.path2linux(os.path.join(path, exe_name))

        # check if the file exists and is executable
        if os.path.isfile(abs_path) and os.access(abs_path, os.X_OK):
            res.append(abs_path)

    if not res:
        if verbose:
            print(f"  :Could not find {exe_name} in the system PATH."
                  " Please make sure the executable is installed."
                  " please include executable extension in the name.")
        return None

    if verbose:
        print(f"  :Found {exe_name} in the system PATH:")
        for path in res:
            print(f"    :{path}")
    return res
