'''
##############################################################
# Created Date: Monday, December 30th 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
import os
import pytest
import pyufunc as pf
from .test_setup import add_pkg_to_sys_path
add_pkg_to_sys_path("realtwin")

from realtwin import REALTWIN


class TestREALTWIN:
    """Test the REALTWIN class"""

    def setup_class(self):
        """Set up the class"""
        self.INPUT_DIR = pf.path2linux(os.getcwd())
        self.INPUT_DIR_NOT_FOUND = "datasets/fake_dir/"

    def test_input_dir_not_found(self):
        """REALTWIN object should be created successfully"""

        with pytest.raises(FileNotFoundError):
            twin = REALTWIN(input_dir=self.INPUT_DIR_NOT_FOUND)

    def test_input_dir_found(self):
        """REALTWIN object should be created successfully"""

        twin = REALTWIN(input_dir=self.INPUT_DIR)
        assert twin._input_dir == self.INPUT_DIR

    def test_output_dir_default(self):
        """REALTWIN object should be created successfully"""

        twin = REALTWIN(input_dir=self.INPUT_DIR)
        assert twin._output_dir == pf.path2linux(os.path.join(self.INPUT_DIR, 'output'))

    def test_output_dir_custom(self):
        """REALTWIN object should be created successfully"""

        OUTPUT_DIR = pf.path2linux(os.getcwd())
        twin = REALTWIN(input_dir=self.INPUT_DIR, output_dir=OUTPUT_DIR)
        assert twin._output_dir == OUTPUT_DIR
