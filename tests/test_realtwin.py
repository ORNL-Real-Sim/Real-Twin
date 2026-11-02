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
import os
import shutil
import tempfile
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
import pyufunc as pf

from tests.proj_add_sys_path import add_pkg_to_sys_path

add_pkg_to_sys_path("realtwin")

import realtwin  # noqa: E402
from realtwin import RealTwinSUMO  # noqa: E402


class TestRealTwinSUMO:
    """Test the REALTWIN class"""

    def setup_class(self):
        """Set up the class"""
        self.INPUT_CONFIG = "realtwin_config.yaml"
        self.INPUT_DIR_NOT_FOUND = "datasets/fake_dir/"

        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

        # Create a minimal test config
        self.test_config_template = {
            "input_dir": os.path.join(self.test_dir, "input"),
            "output_dir": os.path.join(self.test_dir, "output"),
            "demo_data": None,
            "Calibration": {
                "turn_inflow": {"is_calibration": True},
                "behavior": {"is_calibration": True},
            },
            "Network": {
                "NetworkName": "test_net",
                "NetworkVertices": [
                    [-85.14977588011192, 35.040346288414916],
                    [-85.15823020212477, 35.04345144844759],
                ],
            },
        }

        # Create input directory
        os.makedirs(self.test_config_template["input_dir"], exist_ok=True)

    @pytest.fixture(autouse=True)
    def reset_test_config(self):
        self.test_config_content = deepcopy(self.test_config_template)

    def teardown_class(self):
        """Clean up after tests"""
        if hasattr(self, "test_dir") and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_init_no_config_file(self):
        """Test initialization without config file should raise exception"""
        with pytest.raises(Exception, match="Input configuration file is not provided"):
            RealTwinSUMO(input_config_file="")

    def test_init_invalid_config_file(self):
        """Test initialization with invalid config file"""
        with pytest.raises(FileNotFoundError):
            RealTwinSUMO(input_config_file=self.INPUT_DIR_NOT_FOUND)

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    def test_init_success(self, mock_load_config):
        """Test successful initialization"""
        mock_load_config.return_value = self.test_config_content

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)

        assert twin.input_config == self.test_config_content
        assert twin.verbose is False
        assert twin._input_confirm is True
        assert twin._venv_name == "venv_rt"
        assert twin._proj_dir == os.getcwd()
        mock_load_config.assert_called_once_with(self.INPUT_CONFIG)

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    def test_init_with_kwargs(self, mock_load_config):
        """Test initialization with additional kwargs"""
        mock_load_config.return_value = self.test_config_content

        twin = RealTwinSUMO(
            input_config_file=self.INPUT_CONFIG, verbose=True, input_confirm=False
        )

        assert twin.verbose is True
        assert twin._input_confirm is False

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    def test_init_config_found(self, mock_load_config):
        """Test REALTWIN object creation with valid config"""
        mock_load_config.return_value = self.test_config_content

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)
        assert isinstance(twin.input_config, dict)

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    @patch("realtwin.rt_sumo.rt_sumo.install_sumo")
    def test_env_setup_default_sumo(self, mock_install_sumo, mock_load_config):
        """Test environment setup with default SUMO simulator"""
        mock_load_config.return_value = self.test_config_content
        mock_install_sumo.return_value = True

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)
        result = twin.env_setup()

        assert twin.sel_sim == ["sumo"]
        assert "Selected simulators: ['sumo'] are installed successfully" in result
        mock_install_sumo.assert_called_once()

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    @patch("realtwin.rt_sumo.rt_sumo.install_sumo")
    def test_env_setup_multiple_simulators(self, mock_install_sumo, mock_load_config):
        """Test environment setup with multiple simulators"""
        mock_load_config.return_value = self.test_config_content
        mock_install_sumo.return_value = True

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)
        twin.env_setup(sel_sim=["SUMO", "VISSIM"])

        # Only SUMO should succeed (VISSIM installation returns None)
        assert "sumo" in twin.sel_sim
        assert len(twin.sel_sim) >= 1

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    @patch("realtwin.rt_sumo.rt_sumo.install_sumo")
    def test_env_setup_strict_version(self, mock_install_sumo, mock_load_config):
        """Test environment setup with strict version requirements"""
        mock_load_config.return_value = self.test_config_content
        mock_install_sumo.return_value = True

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)
        twin.env_setup(
            sel_sim=["SUMO"], strict_sumo_version="1.21.0", sel_dir=["/custom/path"]
        )

        assert "sumo" in twin.sel_sim
        mock_install_sumo.assert_called_once()
        # Check that kwargs were passed correctly
        call_args = mock_install_sumo.call_args
        assert call_args[1]["strict_sumo_version"] == "1.21.0"
        assert call_args[1]["sel_dir"] == ["/custom/path"]

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    @patch("realtwin.rt_sumo.rt_sumo.install_sumo")
    def test_env_setup_no_simulators_available(
        self, mock_install_sumo, mock_load_config
    ):
        """Test environment setup when no simulators are available"""
        mock_load_config.return_value = self.test_config_content
        mock_install_sumo.return_value = False

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)

        with pytest.raises(Exception, match="No simulator is available"):
            twin.env_setup(sel_sim=["SUMO"])

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    @patch("realtwin.rt_sumo.rt_sumo.check_abstract_inputs")
    @patch("realtwin.rt_sumo.rt_sumo.update_matchup_table")
    @patch("realtwin.rt_sumo.rt_sumo.generate_turn_demand")
    @patch("realtwin.rt_sumo.rt_sumo.parse_SUMO_TLS_ID")
    def test_generate_abstract_scenario(
        self,
        mock_parse_tls,
        mock_gen_demand,
        mock_update_matchup,
        mock_check_inputs,
        mock_load_config,
    ):
        """Test generate_abstract_scenario method"""
        mock_load_config.return_value = self.test_config_content
        mock_update_matchup.return_value = MagicMock()
        mock_gen_demand.return_value = (MagicMock(), MagicMock())

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG, input_confirm=False)

        # Mock abstract_scenario
        twin.abstract_scenario = MagicMock()
        twin.abstract_scenario.Traffic = MagicMock()
        twin.abstract_scenario.Network = MagicMock()
        twin.abstract_scenario.Network.OpenDriveNetwork = MagicMock()
        twin.abstract_scenario.Network.OpenDriveNetwork.OpenDrive_network = [
            "test.net.xml",
            "",
        ]

        with patch("builtins.input", return_value=""):
            twin.generate_abstract_scenario()

        mock_check_inputs.assert_called_once()
        mock_update_matchup.assert_called_once()
        mock_gen_demand.assert_called_once()

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    @patch("realtwin.rt_sumo.rt_sumo.ConcreteScenario")
    def test_generate_concrete_scenario(self, mock_concrete_scenario, mock_load_config):
        """Test generate_concrete_scenario method"""
        mock_load_config.return_value = self.test_config_content
        mock_concrete_instance = MagicMock()
        mock_concrete_scenario.return_value = mock_concrete_instance

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)
        twin.abstract_scenario = MagicMock()  # Mock abstract scenario exists

        twin.generate_concrete_scenario()

        assert hasattr(twin, "concrete_scenario")
        mock_concrete_instance.get_unified_scenario.assert_called_once_with(
            twin.abstract_scenario
        )

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    def test_generate_concrete_scenario_no_abstract(self, mock_load_config):
        """Test generate_concrete_scenario without abstract scenario"""
        mock_load_config.return_value = self.test_config_content

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)
        # Don't set abstract_scenario

        twin.generate_concrete_scenario()  # Should handle gracefully

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    @patch("realtwin.rt_sumo.rt_sumo.SimPrep")
    def test_prepare_simulation(self, mock_sim_prep, mock_load_config):
        """Test prepare_simulation method"""
        mock_load_config.return_value = self.test_config_content
        mock_sim_instance = MagicMock()
        mock_sim_prep.return_value = mock_sim_instance

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)
        twin.sel_sim = ["sumo"]
        twin.concrete_scenario = MagicMock()

        result = twin.prepare_simulation(
            start_time=28800, end_time=32400, seed=101, step_length=0.1
        )

        assert result is True
        mock_sim_instance.create_sumo_sim.assert_called_once_with(
            twin.concrete_scenario,
            start_time=28800,
            end_time=32400,
            seed=101,
            step_length=0.1,
        )

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    @patch("realtwin.rt_sumo.rt_sumo.cali_sumo")
    def test_calibrate_default_params(self, mock_cali_sumo, mock_load_config):
        """Test calibrate method with default parameters"""
        mock_load_config.return_value = self.test_config_content

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)
        twin.sel_sim = ["sumo"]

        result = twin.calibrate()

        assert result is True
        mock_cali_sumo.assert_called_once()
        call_args = mock_cali_sumo.call_args
        assert call_args[1]["sel_algo"] == {"turn_inflow": "ga", "behavior": "ga"}

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    @patch("realtwin.rt_sumo.rt_sumo.cali_sumo")
    def test_calibrate_custom_params(self, mock_cali_sumo, mock_load_config):
        """Test calibrate method with custom parameters"""
        mock_load_config.return_value = self.test_config_content

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)
        twin.sel_sim = ["sumo"]

        custom_algo = {"turn_inflow": "sa", "behavior": "ts"}
        custom_routes = {"route_1": {"time": 20, "edge_list": ["edge1", "edge2"]}}

        result = twin.calibrate(sel_algo=custom_algo, sel_behavior_routes=custom_routes)

        assert result is True
        call_args = mock_cali_sumo.call_args
        assert call_args[1]["sel_algo"] == custom_algo

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    def test_calibrate_invalid_algorithm(self, mock_load_config):
        """Test calibrate method with invalid algorithm"""
        mock_load_config.return_value = self.test_config_content

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)
        twin.sel_sim = ["sumo"]

        invalid_algo = {"turn_inflow": "invalid_algo", "behavior": "ga"}

        result = twin.calibrate(sel_algo=invalid_algo)

        assert result is False

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    def test_post_process(self, mock_load_config):
        """Test post_process method (currently just a placeholder)"""
        mock_load_config.return_value = self.test_config_content

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)

        # Should not raise any exception
        twin.post_process()

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    def test_visualize(self, mock_load_config):
        """Test visualize method (currently just a placeholder)"""
        mock_load_config.return_value = self.test_config_content

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)

        # Should not raise any exception
        twin.visualize()

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    def test_venv_methods_attached(self, mock_load_config):
        """Test that venv methods are attached to the object"""
        mock_load_config.return_value = self.test_config_content

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)

        assert hasattr(twin, "venv_create")
        assert hasattr(twin, "venv_delete")
        assert twin._venv_name == "venv_rt"

    @patch("realtwin.rt_sumo.rt_sumo.load_input_configs")
    def test_output_dir_default(self, mock_load_config):
        """Test default output directory setting"""
        config_without_output = self.test_config_content.copy()
        config_without_output["output_dir"] = pf.path2linux(
            os.path.join(config_without_output["input_dir"], "output")
        )
        mock_load_config.return_value = config_without_output

        twin = RealTwinSUMO(input_config_file=self.INPUT_CONFIG)

        expected_output_dir = pf.path2linux(
            os.path.join(twin.input_config["input_dir"], "output")
        )
        assert twin.input_config["output_dir"] == expected_output_dir


def test_declared_public_exports_are_importable():
    missing = [name for name in realtwin.__all__ if not hasattr(realtwin, name)]
    assert missing == []
