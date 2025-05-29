##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a MIT               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################

from pathlib import Path
import shutil


def prepare_config_file(dest_dir: str = "") -> bool:
    """ Create a default/demo configuration file for the RealTwin package.

    Args:
        dest_dir (str): save the configuration file to destination dir.
            Defaults to "", which means the current working directory.

    Example:
        >>> import realtwin as rt
        >>> rt.create_configuration_file()  # will create a configuration file in the current working directory
        >>> rt.create_configuration_file(dest_dir="path/to/directory")  # will create in the specified directory

    Return:
        bool: True if the configuration file is created successfully.
    """
    # TDD

    path_config = Path(__file__).parent.parent / "data_lib/public_configs.yaml"

    if not dest_dir:
        dest_dir = Path.cwd()

    # check if the destination directory exists
    if not Path(dest_dir).exists():
        raise FileNotFoundError(f"Destination directory does not exist: {dest_dir}")

    # copy the configuration file to the destination directory
    dest_config = Path(dest_dir) / "public_configs.yaml"
    shutil.copy(path_config, dest_config)
    print(f"  :Configuration file created at {dest_config}")
    return True
