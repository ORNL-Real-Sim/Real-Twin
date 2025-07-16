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


import os
import yaml
import re
from pathlib import Path
from zipfile import ZipFile

import pyufunc as pf
from rich.console import Console
console = Console()

from realtwin.util_lib.check_lon_lat_from_list_of_coords import detect_coord_order
from realtwin.util_lib.get_bbox_from_list_of_coords import get_bounding_box_from_vertices


def load_input_config(path_config: str) -> dict:
    """load input configuration from yaml file

    Args:
        path_config (str): the path of the configuration file in yaml format

    Raises:
        FileNotFoundError: if the file is not found
        ValueError: if the file is not in yaml format

    Returns:
        dict: the dictionary of the configuration data
    """

    if isinstance(path_config, Path):
        path_config = str(path_config)

    # TDD check whether the file exists and is a yaml file
    if not os.path.exists(path_config):
        raise FileNotFoundError(f"  :File not found: {path_config}")

    if not (path_config.endswith('.yaml') or path_config.endswith('.yml')):
        raise ValueError(f"  :File is not in yaml format: {path_config}")

    # read the yaml file and return the configuration dictionary
    with open(path_config, 'r', encoding="utf-8") as yaml_data:
        config = yaml.safe_load(yaml_data)

    # check whether input_dir exists
    if config.get('input_dir') is None:
        # set input_dir to current working directory if not specified
        config['input_dir'] = pf.path2linux(os.getcwd())
    else:

        # check if the input_dir is a valid path
        if not Path(config['input_dir']).exists():
            config['input_dir'] = pf.path2linux(os.getcwd())

        # convert input_dir to linux format
        config['input_dir'] = pf.path2linux(config['input_dir'])

    # check output_dir from input configuration file
    if config.get('output_dir') is None:
        # set output_dir to input_dir/output if not specified
        config['output_dir'] = pf.path2linux(os.path.join(config['input_dir'], 'output'))
    elif not os.path.exists(config['output_dir']):
        config['output_dir'] = pf.path2linux(os.path.join(config['input_dir'], 'output'))

    # check whether key sections exist in the configuration file
    key_sections = ["Traffic", 'Network', 'Control']
    for key in key_sections:
        if key not in config:
            console.log(f"[bold]{key} section is not found in the configuration file.")

    # update network bbox if vertices are provided in the input configuration file
    if vertices := config.get('Network', {}).get('NetworkVertices'):

        # update vertices to a list of tuples in (lon, lat) format
        if isinstance(vertices, str):
            # Regular expression to extract the coordinate pairs
            pattern = r"\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)"
            matches = re.findall(pattern, vertices)
            vertices = [(float(match[0]), float(match[1])) for match in matches]
        elif isinstance(vertices, (list, tuple)):
            # Check if the list contains tuples
            if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in vertices):
                vertices = [(float(item[0]), float(item[1])) for item in vertices]
            else:
                raise ValueError("Invalid format: List must contain list/tuple of [lon, lat] or (lat, lon) pairs.")
        else:
            raise ValueError("Invalid format: vertices must be a string or list/tuple.")

        # detect the coordinate order
        coord_order = detect_coord_order(vertices)
        if coord_order == "lat, lon":
            vertices = [[float(lon), float(lat)] for (lat, lon) in vertices]
        elif coord_order == "ambiguous":
            console.log("  :Coordinate order is ambiguous. Assuming (lon, lat) format.")

        # update the vertices in the configuration file
        config['Network']['NetworkVertices'] = vertices

        # update the bounding box if it is not provided
        bbox = config.get('Network', {}).get('Net_BBox')
        if not bbox:
            config['Network']['Net_BBox'] = get_bounding_box_from_vertices(vertices)

    # check whether demo mode is enabled
    config = check_demo_data_enabled(config)

    return config


def check_demo_data_enabled(config: dict) -> dict:
    """Check if demo data is enabled in the configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        dict: The configuration dictionary with demo data check.
    """
    # check whether demo mode is enabled
    AVAILABLE_DEMO_DATA = ["chattanooga"]
    if demo_data := config.get('demo_data'):
        if not isinstance(demo_data, str):
            config['demo_data'] = None
            console.log("  :Demo data is not a string. Demo mode is disabled.")

        elif demo_data.lower() in AVAILABLE_DEMO_DATA:
            try:
                # copy demo data to the input directory
                demo_data_path = pf.path2linux(
                    Path(__file__).parent.parent.parent / "data_lib" / f"{demo_data.lower()}.zip")

                with ZipFile(demo_data_path, 'r') as zip_ref:
                    extract_path = os.path.splitext(demo_data_path)[0]
                    os.makedirs(extract_path, exist_ok=True)
                    zip_ref.extractall(config['input_dir'])
                console.log(
                    f"  :Demo data {demo_data} extracted to {config['input_dir']}.")
                # update input directory to the extracted demo data
                config["input_dir"] = pf.path2linux(
                    Path(config['input_dir']) / f"{demo_data.lower()}")
                config["Network"]["NetworkName"] = demo_data
                # use dummy coordinates to make sure program works (in generate_inputs)
                config["Network"]["NetworkVertices"] = [[-85.14977588011192, 35.040346288414916],
                                                        [-85.15823020212477,
                                                            35.04345144844759],
                                                        [-85.15829457513502,
                                                            35.043293338482925],
                                                        [-85.14986171079225, 35.04018378032611]]
            except Exception as e:
                console.log(
                    f"  :Demo data {demo_data} extraction failed for {e}. Demo mode is disabled.")
                config['demo_data'] = None
        else:
            config['demo_data'] = None
            console.log(
                f"Demo data {demo_data} currently not available. Available demo data: {AVAILABLE_DEMO_DATA}")
            console.log("Demo mode is disabled.")

    return config
