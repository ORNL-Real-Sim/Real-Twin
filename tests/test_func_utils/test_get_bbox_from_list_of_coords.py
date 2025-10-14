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

""" Test the get_bbox_from_vertices function """

import pytest
from pathlib import Path
from tests.proj_add_sys_path import add_pkg_to_sys_path
add_pkg_to_sys_path("realtwin")

from realtwin.util_lib.get_bbox_from_list_of_coords import get_bbox_from_vertices


def test_bbox_from_string():
    """ Test bounding box extraction from a string of coordinates """
    vertices = "(-83.9, 35.9),(-84.1, 36.1),(-84.0, 36.0)"
    bbox = get_bbox_from_vertices(vertices)
    assert bbox == (-84.1, 35.9, -83.9, 36.1)


def test_bbox_from_list_of_tuples():
    """ Test bounding box extraction from a list of tuples """
    vertices = [(-83.9, 35.9), (-84.1, 36.1), (-84.0, 36.0)]
    bbox = get_bbox_from_vertices(vertices)
    assert bbox == (-84.1, 35.9, -83.9, 36.1)


def test_bbox_from_list_of_lists():
    """ Test bounding box extraction from a list of lists """
    vertices = [[-83.9, 35.9], [-84.1, 36.1], [-84.0, 36.0]]
    bbox = get_bbox_from_vertices(vertices)
    assert bbox == (-84.1, 35.9, -83.9, 36.1)


def test_bbox_with_negative_coords():
    """ Test bounding box extraction with negative coordinates """
    vertices = "(-120.5, -45.2),(-121.0, -44.8),(-120.7, -45.5)"
    bbox = get_bbox_from_vertices(vertices)
    assert bbox == (-121.0, -45.5, -120.5, -44.8)


def test_invalid_string_format():
    """ Test error handling for invalid string format """
    with pytest.raises(ValueError):
        get_bbox_from_vertices("invalid string")


def test_invalid_list_format():
    """ Test error handling for invalid list format """
    with pytest.raises(ValueError):
        get_bbox_from_vertices([1, 2, 3])


def test_invalid_type():
    """ Test error handling for invalid type """
    with pytest.raises(ValueError):
        get_bbox_from_vertices(12345)
