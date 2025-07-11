'''
##############################################################
# Created Date: Friday, July 11th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


import pytest
from pathlib import Path
try:
    import realtwin
except ImportError:
    # If realtwin is not installed, use the local path
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import realtwin

from realtwin.util_lib.get_bbox_from_list_of_coords import get_bounding_box_from_vertices


def test_bbox_from_string():
    vertices = "(-83.9, 35.9),(-84.1, 36.1),(-84.0, 36.0)"
    bbox = get_bounding_box_from_vertices(vertices)
    assert bbox == (-84.1, 35.9, -83.9, 36.1)


def test_bbox_from_list_of_tuples():
    vertices = [(-83.9, 35.9), (-84.1, 36.1), (-84.0, 36.0)]
    bbox = get_bounding_box_from_vertices(vertices)
    assert bbox == (-84.1, 35.9, -83.9, 36.1)


def test_bbox_from_list_of_lists():
    vertices = [[-83.9, 35.9], [-84.1, 36.1], [-84.0, 36.0]]
    bbox = get_bounding_box_from_vertices(vertices)
    assert bbox == (-84.1, 35.9, -83.9, 36.1)


def test_bbox_with_negative_coords():
    vertices = "(-120.5, -45.2),(-121.0, -44.8),(-120.7, -45.5)"
    bbox = get_bounding_box_from_vertices(vertices)
    assert bbox == (-121.0, -45.5, -120.5, -44.8)


def test_invalid_string_format():
    with pytest.raises(ValueError):
        get_bounding_box_from_vertices("invalid string")


def test_invalid_list_format():
    with pytest.raises(ValueError):
        get_bounding_box_from_vertices([1, 2, 3])


def test_invalid_type():
    with pytest.raises(ValueError):
        get_bounding_box_from_vertices(12345)