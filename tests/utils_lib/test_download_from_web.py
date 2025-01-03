'''
##############################################################
# Created Date: Friday, January 3rd 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
import os
import pytest
from tests.proj_add_sys_path import add_pkg_to_sys_path
add_pkg_to_sys_path("realtwin")

from realtwin.utils_lib.download_file_from_web import download_single_file_from_web


def test_download_failure():
    url = "https://www.google.com"
    dest_filename = "not_valid.zip"
    assert not download_single_file_from_web(url, dest_filename)

    os.remove(dest_filename)


def test_download_success():
    url = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_92x30dp.png"
    dest_filename = "test.png"
    assert download_single_file_from_web(url, dest_filename)

    os.remove(dest_filename)
