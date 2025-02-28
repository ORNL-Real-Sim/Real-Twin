'''
##############################################################
# Created Date: Wednesday, January 22nd 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


import os
import yaml
import re

import pyufunc as pf


def get_bounding_box_from(vertices: str) -> tuple:
    """get the bounding box from the vertices string

    Args:
        vertices (str): the vertices of the network in string format
            "(lon, lat),(lon, lat),..."

    Notes:
        The vertices format can be found in configuration file

    Returns:
        tuple: the bounding box of the network: (min_lon, min_lat, max_lon, max_lat)
    """

    # Regular expression to extract the coordinate pairs
    pattern = r"\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)"
    matches = re.findall(pattern, vertices)

    lon_lst = [float(match[0]) for match in matches]
    lat_lst = [float(match[1]) for match in matches]

    return (min(lon_lst), min(lat_lst), max(lon_lst), max(lat_lst))


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
        # convert input_dir to linux format
        config['input_dir'] = pf.path2linux(config['input_dir'])

    # check output_dir from input configuration file
    if config.get('output_dir') is None:
        # set output_dir to input_dir/output if not specified
        config['output_dir'] = pf.path2linux(
            os.path.join(config['input_dir'], 'output'))

    # check whether key sections exist in the configuration file
    key_sections = ["Traffic", 'Network', 'Control']
    for key in key_sections:
        if key not in config:
            print(f"  :{key} section is not found in the configuration file.")

    # update network bbox if vertices are provided in the input configuration file
    if vertices := config.get('Network', {}).get('NetworkVertices'):
        bbox = config.get('Network', {}).get('Net_BBox')

        # update the bounding box if it is not provided
        if not bbox:
            config['Network']['Net_BBox'] = get_bounding_box_from(vertices)

    return config
