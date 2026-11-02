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

"""Find executables on Windows and Linux."""

import os
import re
import subprocess

import pyufunc as pf


def find_executable_on_win(
    exe_name: str | re.Pattern,
    ext: str = "exe",
    sel_dir: list | None = None,
    verbose: bool = True,
) -> list | None:
    """Find an executable from PATH or the Windows system drive.

    PATH and ``sel_dir`` are checked first. If they do not contain the
    executable, the Windows system drive is searched recursively. Regex
    searches use common Windows installation directories on that drive to
    avoid an unnecessarily expensive full-drive scan.

    Args:
        exe_name (str or re.Pattern): The executable name or regular expression
            to search for.
        ext (str): The extension of the executable. Defaults to "exe" for executable files.
        sel_dir (list): Additional directories to search. Defaults to None.
        verbose (bool): Whether to print the process info. Defaults to True.

    Returns:
        list or None: Full paths of matching executables, or None if not found.
    """

    # check if the executable name is a string or regular expression
    if not isinstance(exe_name, (str, re.Pattern)):
        raise ValueError(
            "exe_name should be a string. A regular expression is also accepted."
        )  # noqa: TRY004

    # check if extension is str
    if not isinstance(ext, str):
        raise ValueError("ext should be a string.")  # noqa: TRY004

    # check if sel_dir is a list
    if not isinstance(sel_dir, (list, type(None))):
        raise ValueError("sel_dir should be a list.")  # noqa: TRY004

    # Collect additional directories without changing the process environment.
    selected_paths = []
    if sel_dir:
        for path in sel_dir:
            if os.path.isdir(path):
                selected_paths.append(path)
            elif verbose:
                print(f"  :The directory: {path} does not exist. Skipped.")

    executable_pattern = None
    if isinstance(exe_name, str):
        # check if exe_name has the extension
        _, ext_str = os.path.splitext(exe_name)
        if not ext_str:
            if verbose:
                print(
                    f"  :The executable: {exe_name} has no extension. Added {ext} as the extension."
                )
            exe_name = f"{exe_name}.{ext}"
    else:
        executable_pattern = re.compile(
            exe_name.pattern, exe_name.flags | re.IGNORECASE
        )

    def matches_executable(file_name: str) -> bool:
        """Return whether a directory entry matches the requested executable."""
        if executable_pattern is not None:
            return executable_pattern.fullmatch(file_name) is not None
        return file_name.casefold() == exe_name.casefold()

    # Search PATH first, followed by explicitly selected directories.
    env_paths = os.environ.get("PATH", "").split(os.pathsep)
    search_paths = [*env_paths, *selected_paths]

    res = []
    seen_paths = set()
    for path in search_paths:
        if not path:
            continue

        # Quoted PATH entries are accepted by Windows shells but need to be
        # unquoted before they can be used with file-system APIs.
        path = os.fspath(path).strip().strip('"')
        if executable_pattern is None:
            candidate_paths = [os.path.join(path, exe_name)]
        else:
            try:
                with os.scandir(path) as entries:
                    candidate_paths = [
                        entry.path
                        for entry in entries
                        if entry.is_file()
                        and matches_executable(entry.name)
                    ]
            except OSError:
                candidate_paths = []

        for candidate_path in candidate_paths:
            abs_path = pf.path2linux(candidate_path)
            normalized_path = os.path.normcase(os.path.abspath(abs_path))

            # check if the file exists and is executable
            if (
                normalized_path not in seen_paths
                and os.path.isfile(abs_path)
                and os.access(abs_path, os.X_OK)
            ):
                res.append(abs_path)
                seen_paths.add(normalized_path)

    # A Windows installation directory is not always added to PATH. Fall back
    # to the system drive and stop after the first recursive match. Regex
    # searches are limited to common installation roots because scanning every
    # directory on a drive is prohibitively slow for a pattern.
    if not res and os.name == "nt":
        system_drive = os.environ.get("SystemDrive", "C:")
        drive_root = os.path.abspath(f"{system_drive}{os.sep}")

        if os.path.isdir(drive_root):
            preferred_directories = {
                "program files": 0,
                "program files (x86)": 1,
                "programdata": 2,
                "users": 3,
                "windows": 4,
            }
            if executable_pattern is None:
                search_roots = [drive_root]
            else:
                search_roots = [
                    os.path.join(drive_root, directory)
                    for directory in sorted(
                        os.listdir(drive_root),
                        key=lambda name: (
                            preferred_directories.get(name.casefold(), 5),
                            name.casefold(),
                        ),
                    )
                    if directory.casefold() in preferred_directories
                    and os.path.isdir(os.path.join(drive_root, directory))
                ]

            if verbose:
                print(
                    f"  :{exe_name} was not found in PATH. "
                    f"Searching {drive_root} recursively."
                )

            for search_root in search_roots:
                for current_dir, dir_names, file_names in os.walk(
                    search_root,
                    topdown=True,
                    onerror=lambda _error: None,
                    followlinks=False,
                ):
                    dir_names.sort(key=str.casefold)

                    matching_name = next(
                        (name for name in file_names if matches_executable(name)), None
                    )
                    if matching_name is None:
                        continue

                    abs_path = os.path.join(current_dir, matching_name)
                    if os.path.isfile(abs_path) and os.access(abs_path, os.X_OK):
                        res.append(pf.path2linux(abs_path))
                        break
                if res:
                    break

    if not res:
        if verbose:
            print(
                f"  :Could not find {exe_name} in PATH or on the system drive. "
                "Please make sure the executable is installed."
            )
        return None

    if verbose:
        print(f"  :Found {exe_name}:")
        for path in res:
            print(f"    :{path}")
    return res


def find_executable_on_linux(
    exe_name: str, verbose: bool = True
) -> list[str]:
    """Use the system `which -a` to list all matches for exe_name on Linux.

    Args:
        exe_name (str): The name of the executable to search for.
        verbose (bool): Whether to print the process info. Defaults to True.

    Example:
        >>> find_executable_on_linux("python3", verbose=True)
        >>> ['/usr/bin/python3', '/usr/local/bin/python3']

    Returns:
        Optional[List[str]]: A list of paths where the executable is found, or None if not found.
    """

    try:
        # -a: list all matches, not just the first
        proc = subprocess.run(
            ["which", "-a", exe_name], capture_output=True, text=True, check=False
        )
        paths = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if verbose:
            if paths:
                for p in paths:
                    print(f"[which] Found: {p}")
            else:
                print(f"[which] No matches for {exe_name}")
        return paths or None

    except FileNotFoundError:
        if verbose:
            print("[which] `which` command not found on this system.")
        return None
