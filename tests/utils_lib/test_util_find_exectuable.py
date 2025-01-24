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

from realtwin.utils_lib.find_exe_from_PATH import find_executable_from_PATH_on_win


class TestFindExeOnWin:
    """Test the find_executable_from_PATH_on_win function"""

    def setup_class(self):
        """Set up the class"""
        self.EXE_NAME = "sumo.exe"
        self.EXE_NAME_NOT_FOUND = "fake_exe"
        self.SEL_DRI = os.getcwd()

    def test_error_exe_name_error(self):
        """Test the error when exe_name is not a string"""
        with pytest.raises(ValueError,
                           match="exe_name should be a string."):
            find_executable_from_PATH_on_win(exe_name=123)

    def test_ext_error(self):
        """Test the error when ext is not a string"""
        with pytest.raises(ValueError,
                           match="ext should be a string."):
            find_executable_from_PATH_on_win(exe_name=self.EXE_NAME,
                                             ext=123)

    def test_sel_dir_error(self):
        """Test the error when sel_dir is not a list"""
        with pytest.raises(ValueError,
                           match="sel_dir should be a list."):
            find_executable_from_PATH_on_win(exe_name=self.EXE_NAME,
                                             sel_dir=123)

    def test_exe_not_found(self):
        """Test the case when the exe is not found"""
        res = find_executable_from_PATH_on_win(exe_name=self.EXE_NAME_NOT_FOUND)
        assert res is None

    def test_exe_found(self):
        """Test the case when the exe is found"""
        res = find_executable_from_PATH_on_win(exe_name=self.EXE_NAME)
        print("exe found: ", res)
        assert res is not None
