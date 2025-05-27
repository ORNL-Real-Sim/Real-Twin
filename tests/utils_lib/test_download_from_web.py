##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a __TBD__           #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################
import os
from tests.proj_add_sys_path import add_pkg_to_sys_path
add_pkg_to_sys_path("realtwin")

from realtwin.util_lib.download_file_from_web import download_single_file_from_web


def test_download_failure():
    """Test the download_single_file_from_web function"""
    url = "https://www.google.com"
    dest_filename = "not_valid.zip"
    assert not download_single_file_from_web(url, dest_filename)

    os.remove(dest_filename)


def test_download_success():
    """Test the download_single_file_from_web function"""
    url = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_92x30dp.png"
    dest_filename = "test.png"
    assert download_single_file_from_web(url, dest_filename)

    os.remove(dest_filename)
