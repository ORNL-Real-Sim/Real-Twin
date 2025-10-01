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

""" Test the autonomous vehicle module """

import os
import tempfile
import shutil
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
# from tests.proj_add_sys_path import add_pkg_to_sys_path
# add_pkg_to_sys_path("realtwin")

try:
    import realtwin
except ImportError:
    # If realtwin is not installed, use the local path
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import realtwin

from realtwin.autonomous_veh import (
    SimAV,
    prepare_av_configs,
    load_av_configs,
    generate_sumo_loop_detector_add_xml
)
from realtwin.autonomous_veh.sim_av import check_inputs_from_config, name_without_suffixes


class TestLoadAVConfigs:
    """Test the load_av_configs function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_config_path = Path(self.test_dir) / "test_config.yaml"

        # Valid test configuration
        self.valid_config = {
            'pct_penetration': [20, 30, 30, 20],
            'veh_types': ['Human', 'AVnormal', 'AVsafe', 'AVaggressive'],
            'KraussParameters': {
                'minGap': [0.5, 1.0, 1.5, 2.0],
                'accel': [3.8, 3.5, 3.0, 2.5]
            }
        }

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_av_configs_valid_file(self):
        """Test loading a valid configuration file"""
        # Create valid config file
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.valid_config, f)

        result = load_av_configs(self.test_config_path)

        assert isinstance(result, dict)
        assert result['pct_penetration'] == [20, 30, 30, 20]
        assert result['veh_types'] == ['Human', 'AVnormal', 'AVsafe', 'AVaggressive']
        assert 'CFmodel' in result

    def test_load_av_configs_file_not_found(self):
        """Test error handling for non-existent file"""
        non_existent_path = Path(self.test_dir) / "non_existent.yaml"

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_av_configs(non_existent_path)

    def test_load_av_configs_invalid_extension(self):
        """Test error handling for invalid file extension"""
        invalid_file = Path(self.test_dir) / "config.txt"
        invalid_file.touch()

        with pytest.raises(ValueError, match="Configuration file must be a YAML file"):
            load_av_configs(invalid_file)

    def test_load_av_configs_invalid_penetration_sum(self):
        """Test error handling for penetration percentages that don't sum to 100"""
        invalid_config = self.valid_config.copy()
        invalid_config['pct_penetration'] = [20, 30, 30, 30]  # Sum = 110

        with open(self.test_config_path, 'w') as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError, match=r"pct_penetration must sum to 100%\."):
            load_av_configs(self.test_config_path)

    def test_load_av_configs_invalid_penetration_values(self):
        """Test error handling for penetration values outside 0-100 range"""
        invalid_config = self.valid_config.copy()
        invalid_config['pct_penetration'] = [20, 30, 60, -10]  # Negative value

        with open(self.test_config_path, 'w') as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError, match=r"pct_penetration values must be between 0 and 100\."):
            load_av_configs(self.test_config_path)

    def test_load_av_configs_non_list_penetration(self):
        """Test error handling for non-list penetration parameter"""
        invalid_config = self.valid_config.copy()
        invalid_config['pct_penetration'] = 100  # Not a list

        with open(self.test_config_path, 'w') as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(TypeError, match=r"pct_penetration must be a list of percentages\."):
            load_av_configs(self.test_config_path)

    def test_load_av_configs_no_veh_types(self):
        """Test error handling for missing vehicle types"""
        invalid_config = self.valid_config.copy()
        del invalid_config['veh_types']

        with open(self.test_config_path, 'w') as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError, match=r"Vehicle types must be specified in the configuration file\."):
            load_av_configs(self.test_config_path)

    def test_load_av_configs_invalid_path_type(self):
        """Test error handling for invalid path type"""
        with pytest.raises(TypeError, match=r"path_config must be a string or Path object\."):
            load_av_configs(123)

    def test_load_av_configs_path_object(self):
        """Test loading config with Path object"""
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.valid_config, f)

        result = load_av_configs(Path(self.test_config_path))

        assert isinstance(result, dict)
        assert result['pct_penetration'] == [20, 30, 30, 20]


class TestPrepareAVConfigs:
    """Test the prepare_av_configs function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('realtwin.autonomous_veh.sim_av.shutil.copy')
    @patch('realtwin.autonomous_veh.sim_av.Path')
    def test_prepare_av_configs_existing_dir(self, mock_path_class, mock_copy):
        """Test prepare_av_configs with existing directory"""
        # Mock Path behavior
        mock_dest_path = MagicMock()
        mock_dest_path.exists.return_value = True
        mock_path_class.return_value = mock_dest_path

        mock_path_class.__file__ = "/fake/path/sim_av.py"

        result = prepare_av_configs(self.test_dir)

        assert result is True
        mock_copy.assert_called_once()

    @patch('realtwin.autonomous_veh.sim_av.shutil.copy')
    @patch('realtwin.autonomous_veh.sim_av.Path')
    def test_prepare_av_configs_nonexistent_dir(self, mock_path_class, mock_copy):
        """Test prepare_av_configs with non-existent directory (uses current dir)"""
        mock_dest_path = MagicMock()
        mock_dest_path.exists.return_value = False
        mock_path_class.return_value = mock_dest_path
        mock_path_class.cwd.return_value = Path("/current/dir")

        result = prepare_av_configs("/nonexistent/dir")

        assert result is True
        mock_copy.assert_called_once()


class TestSimAV:
    """Test the SimAV class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_config_path = Path(self.test_dir) / "test_config.yaml"

        # Valid test configuration
        self.valid_config = {
            'pct_penetration': [25, 25, 25, 25],
            'veh_types': ['Human', 'AVnormal', 'AVsafe', 'AVaggressive'],
            'input': {
                'input_dir': self.test_dir,
                'net_file': 'test.net.xml',
                'flow_file': 'test.flow.xml',
                'turn_file': 'test.turn.xml'
            },
            'sim_name': 'test_sim',
            'sim_start': 0,
            'sim_time': 3600
        }

        # Create test input files
        self.net_file = Path(self.test_dir) / 'test.net.xml'
        self.flow_file = Path(self.test_dir) / 'test.flow.xml'
        self.turn_file = Path(self.test_dir) / 'test.turn.xml'

        # Create minimal XML files
        for file_path in [self.net_file, self.flow_file, self.turn_file]:
            with open(file_path, 'w') as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n<root></root>')

        # Create config file
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.valid_config, f)

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_sim_av_init_success(self):
        """Test successful SimAV initialization"""
        sim_av = SimAV(self.test_config_path, verbose=True)

        assert sim_av.path_config == Path(self.test_config_path)
        assert sim_av.verbose is True

    def test_sim_av_init_file_not_found(self):
        """Test SimAV initialization with non-existent config file"""
        with pytest.raises(FileNotFoundError):
            SimAV("non_existent_config.yaml")

    def test_sim_av_init_invalid_path_type(self):
        """Test SimAV initialization with invalid path type"""
        with pytest.raises(ValueError, match=r".*path_config must be a string or Path object\."):
            SimAV(123)

    @patch('realtwin.autonomous_veh.sim_av.load_av_configs')
    def test_sim_av_load_config_success(self, mock_load_configs):
        """Test successful config loading"""
        mock_load_configs.return_value = self.valid_config

        sim_av = SimAV(self.test_config_path)
        result = sim_av.load_config(str(self.test_config_path))

        assert result is True
        assert hasattr(sim_av, 'av_config')
        mock_load_configs.assert_called_once_with(str(self.test_config_path))

    @patch('realtwin.autonomous_veh.sim_av.load_av_configs')
    def test_sim_av_load_config_failure(self, mock_load_configs):
        """Test config loading failure"""
        mock_load_configs.side_effect = Exception("Loading failed")

        sim_av = SimAV(self.test_config_path)
        result = sim_av.load_config(str(self.test_config_path))

        assert result is False

    @patch('realtwin.autonomous_veh.sim_av.prepare_av_configs')
    def test_sim_av_prepare_av_configs(self, mock_prepare):
        """Test SimAV prepare_av_configs method"""
        mock_prepare.return_value = True

        sim_av = SimAV(self.test_config_path)
        result = sim_av.prepare_av_configs(self.test_dir)

        assert result is True
        mock_prepare.assert_called_once_with(self.test_dir)

    @patch('realtwin.autonomous_veh.sim_av.run_sumo_simulation')
    @patch('realtwin.autonomous_veh.sim_av.create_sumo_config')
    @patch('realtwin.autonomous_veh.sim_av.add_veh_types_to_rou')
    @patch('realtwin.autonomous_veh.sim_av.create_sumo_rou_xml')
    @patch('realtwin.autonomous_veh.sim_av.update_sumo_flow_xml')
    @patch('realtwin.autonomous_veh.sim_av.create_veh_type_attributes')
    @patch('realtwin.autonomous_veh.sim_av.generate_sumo_loop_detector_add_xml')
    @patch('realtwin.autonomous_veh.sim_av.shutil')
    @patch('realtwin.autonomous_veh.sim_av.load_av_configs')
    def test_sim_av_run_simulation_with_config_file(self, mock_load_configs, mock_shutil,
                                                    mock_gen_detector, mock_create_veh_attrs,
                                                    mock_update_flow, mock_create_rou,
                                                    mock_add_veh_types, mock_create_config,
                                                    mock_run_sumo):
        """Test running simulation with config file"""
        mock_load_configs.return_value = self.valid_config
        mock_create_veh_attrs.return_value = {}

        sim_av = SimAV(self.test_config_path)
        result = sim_av.run_simulation(self.test_config_path)

        assert result is True
        mock_load_configs.assert_called()
        mock_run_sumo.assert_called_once()

    def test_sim_av_run_simulation_no_config(self):
        """Test running simulation without config"""
        sim_av = SimAV(self.test_config_path)

        # with pytest.raises(Exception, match=r".*No AV configuration provided or loaded\."):
        #     sim_av.run_simulation()

        with pytest.raises(Exception, match=r" No such file or directory"):
            sim_av.run_simulation()

    @patch('realtwin.autonomous_veh.sim_av.check_inputs_from_config')
    def test_sim_av_run_simulation_invalid_inputs(self, mock_check_inputs):
        """Test running simulation with invalid inputs"""
        mock_check_inputs.return_value = False

        sim_av = SimAV(self.test_config_path)

        with pytest.raises(Exception, match=".*Invalid inputs in the configuration file"):
            sim_av.run_simulation(self.valid_config)


class TestCheckInputsFromConfig:
    """Test the check_inputs_from_config function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()

        # Create test files
        self.net_file = Path(self.test_dir) / 'test.net.xml'
        self.flow_file = Path(self.test_dir) / 'test.flow.xml'
        self.turn_file = Path(self.test_dir) / 'test.turn.xml'

        for file_path in [self.net_file, self.flow_file, self.turn_file]:
            file_path.touch()

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_check_inputs_valid_config(self):
        """Test checking valid input configuration"""
        config = {
            'input': {
                'input_dir': self.test_dir,
                'net_file': 'test.net.xml',
                'flow_file': 'test.flow.xml',
                'turn_file': 'test.turn.xml'
            }
        }

        result = check_inputs_from_config(config)
        assert result is True

    def test_check_inputs_no_input_dict(self):
        """Test checking config without input dictionary"""
        config = {}

        result = check_inputs_from_config(config)
        assert result is False

    def test_check_inputs_no_input_dir(self):
        """Test checking config without input directory"""
        config = {
            'input': {
                'net_file': 'test.net.xml',
                'flow_file': 'test.flow.xml',
                'turn_file': 'test.turn.xml',
            }
        }

        result = check_inputs_from_config(config)
        assert result is False

    def test_check_inputs_nonexistent_dir(self):
        """Test checking config with non-existent directory"""
        config = {
            'input': {
                'input_dir': 'test.net.xml',
                'net_file': 'test.net.xml',
                'flow_file': 'test.flow.xml',
                'turn_file': 'test.turn.xml'
            }
        }

        result = check_inputs_from_config(config)
        assert result is False

    def test_check_inputs_missing_net_file(self):
        """Test checking config with missing network file"""
        config = {
            'input': {
                'input_dir': self.test_dir,
                'flow_file': 'test.flow.xml',
                'turn_file': 'test.turn.xml'
            }
        }

        result = check_inputs_from_config(config)
        assert result is False

    def test_check_inputs_nonexistent_net_file(self):
        """Test checking config with non-existent network file"""
        config = {
            'input': {
                'input_dir': self.test_dir,
                'net_file': 'nonexistent.net.xml',
                'flow_file': 'test.flow.xml',
                'turn_file': 'test.turn.xml'
            }
        }

        result = check_inputs_from_config(config)
        assert result is False


class TestNameWithoutSuffixes:
    """Test the name_without_suffixes function"""

    def test_name_without_suffixes_single_suffix(self):
        """Test removing single suffix"""
        path = Path("test.xml")
        result = name_without_suffixes(path)
        assert result == "test"

    def test_name_without_suffixes_multiple_suffixes(self):
        """Test removing multiple suffixes"""
        path = Path("test.net.xml")
        result = name_without_suffixes(path)
        assert result == "test"

    def test_name_without_suffixes_no_suffix(self):
        """Test with no suffix"""
        path = Path("test")
        result = name_without_suffixes(path)
        assert result == "test"


class TestGenerateSumoLoopDetectorAddXml:
    """Test the generate_sumo_loop_detector_add_xml function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.net_file = Path(self.test_dir) / 'test.net.xml'

        # Create a minimal SUMO network XML with traffic lights and lanes
        net_content = '''<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16" junctionCornerDetail="5" limitTurnSpeed="5.50"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <edge id="edge1" from="junction1" to="tl_junction" priority="1">
        <lane id="edge1_0" index="0" speed="13.89" length="100.0"/>
        <lane id="edge1_1" index="1" speed="13.89" length="100.0"/>
    </edge>
    <edge id="edge2" from="junction2" to="tl_junction" priority="1">
        <lane id="edge2_0" index="0" speed="13.89" length="150.0"/>
    </edge>
    <tlLogic id="tl_junction" type="static" programID="0">
        <phase duration="31" state="GGr"/>
        <phase duration="6"  state="yyr"/>
        <phase duration="31" state="rrG"/>
        <phase duration="6"  state="rry"/>
    </tlLogic>
</net>'''

        with open(self.net_file, 'w') as f:
            f.write(net_content)

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_generate_detector_e1_default(self):
        """Test generating E1 detectors with default parameters"""
        result = generate_sumo_loop_detector_add_xml(
            self.net_file,
            detector_type="E1",
            dest_dir=self.test_dir
        )

        assert result is True
        detector_file = Path(self.test_dir) / "detector.add.xml"
        assert detector_file.exists()

        # Check content
        with open(detector_file, 'r') as f:
            content = f.read()
            assert "inductionLoop" in content
            assert "edge1_0_detector" in content
            assert "edge1_1_detector" in content
            assert "edge2_0_detector" in content

    def test_generate_detector_e2(self):
        """Test generating E2 detectors"""
        result = generate_sumo_loop_detector_add_xml(
            self.net_file,
            detector_type="E2",
            add_fname="e2_detectors.add.xml",
            dest_dir=self.test_dir
        )

        assert result is True
        detector_file = Path(self.test_dir) / "e2_detectors.add.xml"
        assert detector_file.exists()

        with open(detector_file, 'r') as f:
            content = f.read()
            assert "laneAreaDetector" in content

    def test_generate_detector_e0(self):
        """Test generating E0 detectors"""
        result = generate_sumo_loop_detector_add_xml(
            self.net_file,
            detector_type="E0",
            dest_dir=self.test_dir
        )

        assert result is True
        detector_file = Path(self.test_dir) / "detector.add.xml"
        assert detector_file.exists()

        with open(detector_file, 'r') as f:
            content = f.read()
            assert "instantInductionLoop" in content

    def test_generate_detector_invalid_type(self):
        """Test error handling for invalid detector type"""
        with pytest.raises(ValueError, match=r"Unknown detector type.*Accepted types are E1, E2, E0\."):
            generate_sumo_loop_detector_add_xml(
                self.net_file,
                detector_type="INVALID",
                dest_dir=self.test_dir
            )

    def test_generate_detector_custom_output_filename(self):
        """Test generating detectors with custom output filename"""
        result = generate_sumo_loop_detector_add_xml(
            self.net_file,
            detector_type="E1",
            detector_output_fname="custom_output.xml",
            dest_dir=self.test_dir
        )

        assert result is True
        detector_file = Path(self.test_dir) / "detector.add.xml"
        assert detector_file.exists()

        with open(detector_file, 'r') as f:
            content = f.read()
            assert "custom_output.xml" in content

    def test_generate_detector_auto_add_xml_extension(self):
        """Test automatic addition of .add.xml extension"""
        result = generate_sumo_loop_detector_add_xml(
            self.net_file,
            detector_type="E1",
            add_fname="detectors",  # No extension
            dest_dir=self.test_dir
        )

        assert result is True
        detector_file = Path(self.test_dir) / "detectors.add.xml"
        assert detector_file.exists()

    def test_generate_detector_auto_xml_extension_output(self):
        """Test automatic addition of .xml extension for output filename"""
        result = generate_sumo_loop_detector_add_xml(
            self.net_file,
            detector_type="E1",
            detector_output_fname="output_file",  # No extension
            dest_dir=self.test_dir
        )

        assert result is True
        detector_file = Path(self.test_dir) / "detector.add.xml"
        assert detector_file.exists()

        with open(detector_file, 'r') as f:
            content = f.read()
            assert "output_file.xml" in content

    def test_generate_detector_path_object_input(self):
        """Test using Path object as input"""
        result = generate_sumo_loop_detector_add_xml(
            Path(self.net_file),
            detector_type="E1",
            dest_dir=self.test_dir
        )

        assert result is True
        detector_file = Path(self.test_dir) / "detector.add.xml"
        assert detector_file.exists()


class TestIntegration:
    """Integration tests for the autonomous vehicle module"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('realtwin.autonomous_veh.sim_av.shutil.copy')
    @patch('realtwin.autonomous_veh.sim_av.Path')
    def test_prepare_av_configs_integration(self, mock_path_class, mock_copy):
        """Test integration of prepare_av_configs"""
        mock_dest_path = MagicMock()
        mock_dest_path.exists.return_value = True
        mock_path_class.return_value = mock_dest_path

        # Test both function and class method
        result1 = prepare_av_configs(self.test_dir)

        # Create a dummy config file for SimAV initialization
        dummy_config_path = Path(self.test_dir) / "dummy_config.yaml"
        dummy_config_path.touch()

        sim_av = SimAV(str(dummy_config_path))
        result2 = sim_av.prepare_av_configs(self.test_dir)

        assert result1 is True
        assert result2 is True