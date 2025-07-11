'''
##############################################################
# Created Date: Friday, July 11th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

import re


def get_bounding_box_from_vertices(vertices: str | list) -> tuple:
    """get the bounding box from the vertices string

    Args:
        vertices (str): the vertices of the network in string format
            "(lon, lat),(lon, lat),..."

    Notes:
        The vertices format can be found in configuration file

    Returns:
        tuple: the bounding box of the network: (min_lon, min_lat, max_lon, max_lat)
    """
    if isinstance(vertices, str):
        # Regular expression to extract the coordinate pairs
        pattern = r"\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)"
        matches = re.findall(pattern, vertices)

        lon_lst = [float(match[0]) for match in matches]
        lat_lst = [float(match[1]) for match in matches]
    elif isinstance(vertices, (list, tuple)):
        # Check if the list contains tuples
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in vertices):
            lon_lst = [float(item[0]) for item in vertices]
            lat_lst = [float(item[1]) for item in vertices]
        else:
            raise ValueError(
                "Invalid format: List must contain list/tuple of [lon, lat].")
    else:
        raise ValueError("Invalid format: vertices must be a string or list.")

    return (min(lon_lst), min(lat_lst), max(lon_lst), max(lat_lst))
