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
import pytest
import pyufunc as pf
from tests.proj_add_sys_path import add_pkg_to_sys_path
add_pkg_to_sys_path("realtwin")

from realtwin import RealTwin


class TestRealTwin:
    """Test the REALTWIN class"""

    def setup_class(self):
        """Set up the class"""
        self.INPUT_CONFIG = "realtwin_config.yaml"
        self.INPUT_DIR_NOT_FOUND = "datasets/fake_dir/"

    def test_input_dir_not_found(self):
        """REALTWIN object should be created successfully"""

        with pytest.raises(FileNotFoundError):
            RealTwin(input_config_file=self.INPUT_DIR_NOT_FOUND)

    def test_input_config_found(self):
        """REALTWIN object should be created successfully"""

        twin = RealTwin(input_config_file=self.INPUT_CONFIG)
        assert isinstance(twin.input_config, dict)

    def test_output_dir_default(self):
        """REALTWIN object should be created successfully"""

        twin = RealTwin(input_config_file=self.INPUT_CONFIG)
        assert twin.input_config["output_dir"] == pf.path2linux(os.path.join(
            twin.input_config["input_dir"], 'output'))
