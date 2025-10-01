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
import tempfile
import shutil
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from tests.proj_add_sys_path import add_pkg_to_sys_path
add_pkg_to_sys_path("realtwin")

from realtwin.autonomous_veh.sim_av import (
    SimAV,
    prepare_av_configs,
    check_inputs_from_config,
    name_without_suffixes
)


class TestSimAVWorkflow:
    """Comprehensive tests for SimAV simulation workflow"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()

        # Create comprehensive test configuration
        self.complete_av_config = {
            'veh_types': ['Human', 'AVnormal', 'AVsafe', 'AVaggressive'],
            'pct_penetration': [40, 30, 20, 10],
            'car_follow_model': 'Krauss',
            'lane_change_model': 'LC2013',
            'sim_name': 'test_simulation',
            'sim_start': 7200,  # 2 hours
            'sim_time': 1800,   # 30 minutes
            'input': {
                'input_dir': str(self.test_dir),
                'net_file': 'network.net.xml',
                'flow_file': 'flows.flow.xml',
                'turn_file': 'turns.turn.xml'
            },
            'CFmodel': {
                'Human': {
                    'Krauss': {
                        'accel': 2.6, 'decel': 4.5, 'minGap': 2.5,
                        'sigma': 0.5, 'tau': 1.0, 'emergencyDecel': 9.0
                    }
                },
                'AVnormal': {
                    'Krauss': {
                        'accel': 3.0, 'decel': 5.0, 'minGap': 2.0,
                        'sigma': 0.3, 'tau': 0.8, 'emergencyDecel': 9.5
                    }
                },
                'AVsafe': {
                    'Krauss': {
                        'accel': 2.2, 'decel': 5.5, 'minGap': 3.0,
                        'sigma': 0.1, 'tau': 1.2, 'emergencyDecel': 10.0
                    }
                },
                'AVaggressive': {
                    'Krauss': {
                        'accel': 3.5, 'decel': 6.0, 'minGap': 1.5,
                        'sigma': 0.2, 'tau': 0.6, 'emergencyDecel': 11.0
                    }
                }
            }
        }

        # Create test input files
        self.create_test_files()

        # Create config file
        self.config_path = Path(self.test_dir) / 'test_config.yaml'
        with open(self.config_path, 'w') as f:
            yaml.dump(self.complete_av_config, f)

    def create_test_files(self):
        """Create test SUMO files"""
        # Network file
        net_content = '''<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16">
    <edge id="edge1" from="node1" to="node2" priority="1">
        <lane id="edge1_0" index="0" speed="13.89" length="200.0"/>
    </edge>
    <edge id="edge2" from="node2" to="node3" priority="1">
        <lane id="edge2_0" index="0" speed="13.89" length="150.0"/>
    </edge>
</net>'''

        # Flow file
        flow_content = '''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <flow id="flow1" from="edge1" to="edge2" vehsPerHour="200" begin="0" end="3600"/>
</routes>'''

        # Turn file
        turn_content = '''<?xml version="1.0" encoding="UTF-8"?>
<turns>
    <interval begin="0" end="3600">
        <fromEdge id="edge1">
            <toEdge id="edge2" probability="1.0"/>
        </fromEdge>
    </interval>
</turns>'''

        # Write files
        files_content = {
            'network.net.xml': net_content,
            'flows.flow.xml': flow_content,
            'turns.turn.xml': turn_content
        }

        for filename, content in files_content.items():
            with open(Path(self.test_dir) / filename, 'w') as f:
                f.write(content)

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('realtwin.autonomous_veh.sim_av.run_sumo_simulation')
    @patch('realtwin.autonomous_veh.sim_av.create_sumo_config')
    @patch('realtwin.autonomous_veh.sim_av.add_veh_types_to_rou')
    @patch('realtwin.autonomous_veh.sim_av.create_sumo_rou_xml')
    @patch('realtwin.autonomous_veh.sim_av.update_sumo_flow_xml')
    @patch('realtwin.autonomous_veh.sim_av.create_veh_type_attributes')
    @patch('realtwin.autonomous_veh.sim_av.generate_sumo_loop_detector_add_xml')
    def test_run_simulation_complete_workflow(self, mock_gen_detector, mock_create_veh_attrs,
                                              mock_update_flow, mock_create_rou,
                                              mock_add_veh_types, mock_create_config,
                                              mock_run_sumo):
        """Test complete simulation workflow with all steps"""
        # Setup mocks
        mock_create_veh_attrs.return_value = {
            'Human': {'id': 'Human', 'probability': 0.4},
            'AVnormal': {'id': 'AVnormal', 'probability': 0.3}
        }
        mock_gen_detector.return_value = True
        mock_update_flow.return_value = True
        mock_create_rou.return_value = True
        mock_add_veh_types.return_value = True
        mock_create_config.return_value = True
        mock_run_sumo.return_value = True

        # Create SimAV instance and run simulation
        sim_av = SimAV(self.config_path)
        result = sim_av.run_simulation()

        # Verify workflow completion
        assert result is True

        # Verify all functions were called
        mock_create_veh_attrs.assert_called_once()
        mock_gen_detector.assert_called_once()
        mock_update_flow.assert_called_once()
        mock_create_rou.assert_called_once()
        mock_add_veh_types.assert_called_once()
        mock_create_config.assert_called_once()
        mock_run_sumo.assert_called_once()

        # Verify output directory creation
        output_dir = Path(self.test_dir) / 'output_AV'
        assert output_dir.exists()

    def test_run_simulation_custom_config_dict(self):
        """Test running simulation with configuration dictionary"""
        sim_av = SimAV(self.config_path)

        with patch('realtwin.autonomous_veh.sim_av.check_inputs_from_config') as mock_check:
            mock_check.return_value = False

            with pytest.raises(Exception, match=r".*Invalid inputs in the configuration file"):
                sim_av.run_simulation(self.complete_av_config)

    def test_run_simulation_file_operations(self):
        """Test file copying and directory creation during simulation"""
        with patch('realtwin.autonomous_veh.sim_av.shutil.copy') as mock_copy, \
             patch('realtwin.autonomous_veh.sim_av.shutil.rmtree') as mock_rmtree, \
             patch('realtwin.autonomous_veh.sim_av.run_sumo_simulation') as mock_run_sumo, \
             patch('realtwin.autonomous_veh.sim_av.create_sumo_config') as mock_config, \
             patch('realtwin.autonomous_veh.sim_av.add_veh_types_to_rou') as mock_add_types, \
             patch('realtwin.autonomous_veh.sim_av.create_sumo_rou_xml') as mock_create_rou, \
             patch('realtwin.autonomous_veh.sim_av.update_sumo_flow_xml') as mock_update_flow, \
             patch('realtwin.autonomous_veh.sim_av.create_veh_type_attributes') as mock_create_attrs, \
             patch('realtwin.autonomous_veh.sim_av.generate_sumo_loop_detector_add_xml') as mock_detector:

            # Setup mocks
            for mock_func in [mock_run_sumo, mock_config, mock_add_types,
                             mock_create_rou, mock_update_flow, mock_detector]:
                mock_func.return_value = True
            mock_create_attrs.return_value = {}

            sim_av = SimAV(self.config_path)

            # Pre-create output directory to test removal
            output_dir = Path(self.test_dir) / 'output_AV'
            output_dir.mkdir(exist_ok=True)

            result = sim_av.run_simulation()

            assert result is True

            # Verify file operations
            assert mock_copy.call_count >= 2  # At least network, flow, and turn files

            # Verify directory removal and recreation
            mock_rmtree.assert_called_once()

    def test_run_simulation_default_parameters(self):
        """Test simulation with default parameters when not specified in config"""
        # Remove optional parameters from config
        minimal_config = self.complete_av_config.copy()
        del minimal_config['sim_name']
        del minimal_config['sim_start']
        del minimal_config['sim_time']

        with open(self.config_path, 'w') as f:
            yaml.dump(minimal_config, f)

        with patch('realtwin.autonomous_veh.sim_av.run_sumo_simulation') as mock_run_sumo, \
             patch('realtwin.autonomous_veh.sim_av.create_sumo_config') as mock_config, \
             patch('realtwin.autonomous_veh.sim_av.add_veh_types_to_rou') as mock_add_types, \
             patch('realtwin.autonomous_veh.sim_av.create_sumo_rou_xml') as mock_create_rou, \
             patch('realtwin.autonomous_veh.sim_av.update_sumo_flow_xml') as mock_update_flow, \
             patch('realtwin.autonomous_veh.sim_av.create_veh_type_attributes') as mock_create_attrs, \
             patch('realtwin.autonomous_veh.sim_av.generate_sumo_loop_detector_add_xml') as mock_detector:

            # Setup mocks
            for mock_func in [mock_run_sumo, mock_add_types, mock_create_rou,
                              mock_update_flow, mock_detector]:
                mock_func.return_value = True
            mock_create_attrs.return_value = {}

            sim_av = SimAV(self.config_path)
            result = sim_av.run_simulation()

            assert result is True

            # Check that default values were used
            config_call_args = mock_config.call_args
            assert 'sim_name' in config_call_args[1]
            assert config_call_args[1]['sim_name'] == 'chatt'  # Default
            assert config_call_args[1]['sim_start'] == 0       # Default
            assert config_call_args[1]['sim_end'] == 3600      # Default sim_time

    @patch('realtwin.autonomous_veh.sim_av.load_av_configs')
    def test_run_simulation_with_config_file_path(self, mock_load_config):
        """Test running simulation with config file path"""
        mock_load_config.return_value = self.complete_av_config

        sim_av = SimAV(self.config_path)

        with patch('realtwin.autonomous_veh.sim_av.check_inputs_from_config') as mock_check:
            mock_check.return_value = False

            with pytest.raises(Exception, match=r".*Invalid inputs in the configuration file"):
                sim_av.run_simulation(str(self.config_path))

        # Verify config was loaded from file
        mock_load_config.assert_called_with(str(self.config_path))

    def test_run_simulation_path_handling(self):
        """Test proper path handling for input and output files"""
        with patch('realtwin.autonomous_veh.sim_av.run_sumo_simulation') as mock_run_sumo, \
             patch('realtwin.autonomous_veh.sim_av.create_sumo_config') as mock_config, \
             patch('realtwin.autonomous_veh.sim_av.add_veh_types_to_rou') as mock_add_types, \
             patch('realtwin.autonomous_veh.sim_av.create_sumo_rou_xml') as mock_create_rou, \
             patch('realtwin.autonomous_veh.sim_av.update_sumo_flow_xml') as mock_update_flow, \
             patch('realtwin.autonomous_veh.sim_av.create_veh_type_attributes') as mock_create_attrs, \
             patch('realtwin.autonomous_veh.sim_av.generate_sumo_loop_detector_add_xml') as mock_detector:

            # Setup mocks
            for mock_func in [mock_run_sumo, mock_config, mock_add_types,
                             mock_create_rou, mock_update_flow, mock_detector]:
                mock_func.return_value = True
            mock_create_attrs.return_value = {}

            sim_av = SimAV(self.config_path)
            result = sim_av.run_simulation()

            assert result is True

            # Verify that paths are correctly constructed
            rou_call_args = mock_create_rou.call_args[0]

            # All paths should be in output directory
            assert str(Path(self.test_dir) / 'output_AV') in str(rou_call_args[0])  # net file
            assert str(Path(self.test_dir) / 'output_AV') in str(rou_call_args[1])  # flow file
            assert str(Path(self.test_dir) / 'output_AV') in str(rou_call_args[2])  # turn file
            assert str(Path(self.test_dir) / 'output_AV') in str(rou_call_args[3])  # rou file


class TestCheckInputsFromConfigExtended:
    """Extended tests for check_inputs_from_config function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_check_inputs_input_dir_is_file(self):
        """Test when input_dir points to a file instead of directory"""
        # Create a file instead of directory
        input_file = Path(self.test_dir) / 'not_a_dir.txt'
        input_file.touch()

        config = {
            'input': {
                'input_dir': str(input_file),
                'net_file': 'test.net.xml',
                'flow_file': 'test.flow.xml',
                'turn_file': 'test.turn.xml'
            }
        }

        result = check_inputs_from_config(config)
        assert result is False

    def test_check_inputs_all_files_missing(self):
        """Test when all required files are missing"""
        config = {
            'input': {
                'input_dir': self.test_dir,
                'net_file': 'missing_network.net.xml',
                'flow_file': 'missing_flow.flow.xml',
                'turn_file': 'missing_turn.turn.xml'
            }
        }

        result = check_inputs_from_config(config)
        assert result is False

    def test_check_inputs_partial_files_missing(self):
        """Test when some files exist and some don't"""
        # Create only network file
        net_file = Path(self.test_dir) / 'network.net.xml'
        net_file.touch()

        config = {
            'input': {
                'input_dir': self.test_dir,
                'net_file': 'network.net.xml',
                'flow_file': 'missing_flow.flow.xml',
                'turn_file': 'missing_turn.turn.xml'
            }
        }

        result = check_inputs_from_config(config)
        assert result is False

    def test_check_inputs_empty_filenames(self):
        """Test with empty filename strings"""
        config = {
            'input': {
                'input_dir': self.test_dir,
                'net_file': '',
                'flow_file': '',
                'turn_file': ''
            }
        }

        result = check_inputs_from_config(config)
        assert result is False

    def test_check_inputs_missing_file_keys(self):
        """Test when file keys are missing from config"""
        config = {
            'input': {
                'input_dir': self.test_dir
                # Missing net_file, flow_file, turn_file keys
            }
        }

        result = check_inputs_from_config(config)
        assert result is False

    def test_check_inputs_success_case(self):
        """Test successful input validation"""
        # Create all required files
        files = ['network.net.xml', 'flows.flow.xml', 'turns.turn.xml']
        for filename in files:
            (Path(self.test_dir) / filename).touch()

        config = {
            'input': {
                'input_dir': self.test_dir,
                'net_file': 'network.net.xml',
                'flow_file': 'flows.flow.xml',
                'turn_file': 'turns.turn.xml'
            }
        }

        result = check_inputs_from_config(config)
        assert result is True


class TestNameWithoutSuffixesExtended:
    """Extended tests for name_without_suffixes function"""

    def test_name_without_suffixes_complex_extensions(self):
        """Test with complex multi-part extensions"""
        test_cases = [
            ("file.tar.gz", "file"),
            ("archive.tar.bz2", "archive"),
            ("data.csv.backup", "data"),
            ("model.h5.old", "model"),
            ("config.yaml.template", "config")
        ]

        for input_name, expected in test_cases:
            path = Path(input_name)
            result = name_without_suffixes(path)
            assert result == expected

    def test_name_without_suffixes_hidden_files(self):
        """Test with hidden files (starting with dot)"""
        test_cases = [
            (".hidden", ".hidden"),
            (".config.json", ".config"),
            (".bashrc.backup", ".bashrc")
        ]

        for input_name, expected in test_cases:
            path = Path(input_name)
            result = name_without_suffixes(path)
            assert result == expected


class TestPrepareAVConfigsExtended:
    """Extended tests for prepare_av_configs function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('realtwin.autonomous_veh.sim_av.Path')
    @patch('realtwin.autonomous_veh.sim_av.shutil.copy')
    def test_prepare_av_configs_path_construction(self, mock_copy, mock_path_class):
        """Test proper path construction in prepare_av_configs"""
        mock_dest_path = MagicMock()
        mock_dest_path.exists.return_value = True
        mock_path_class.return_value = mock_dest_path

        # Mock __file__ path construction
        mock_file_path = MagicMock()
        mock_file_path.parent.parent = MagicMock()
        mock_path_class.__file__ = mock_file_path

        result = prepare_av_configs(self.test_dir)

        assert result is True
        mock_copy.assert_called_once()

    @patch('realtwin.autonomous_veh.sim_av.Path')
    def test_prepare_av_configs_cwd_fallback(self, mock_path_class):
        """Test fallback to current working directory"""
        mock_dest_path = MagicMock()
        mock_dest_path.exists.return_value = False
        mock_cwd_path = MagicMock()

        mock_path_class.side_effect = lambda x: mock_dest_path if x == "nonexistent" else mock_cwd_path
        mock_path_class.cwd.return_value = mock_cwd_path

        with patch('realtwin.autonomous_veh.sim_av.shutil.copy'):
            result = prepare_av_configs("nonexistent")

            assert result is True
            mock_path_class.cwd.assert_called_once()


class TestSimAVErrorHandling:
    """Test error handling scenarios in SimAV"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = Path(self.test_dir) / 'config.yaml'

        # Create minimal config
        minimal_config = {
            'veh_types': ['Human'],
            'pct_penetration': [100],
            'input': {
                'input_dir': self.test_dir,
                'net_file': 'test.net.xml',
                'flow_file': 'test.flow.xml',
                'turn_file': 'test.turn.xml'
            }
        }

        with open(self.config_path, 'w') as f:
            yaml.dump(minimal_config, f)

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_config_exception_handling(self):
        """Test exception handling in load_config method"""
        sim_av = SimAV(self.config_path)

        with patch('realtwin.autonomous_veh.sim_av.load_av_configs') as mock_load:
            mock_load.side_effect = Exception("Loading failed")

            result = sim_av.load_config(str(self.config_path))

            assert result is False
            mock_load.assert_called_once()

    def test_run_simulation_no_config_fallback(self):
        """Test run_simulation fallback to path_config when no av_config exists"""
        sim_av = SimAV(self.config_path)

        # Ensure no av_config attribute exists
        assert not hasattr(sim_av, 'av_config')

        with patch('realtwin.autonomous_veh.sim_av.load_av_configs') as mock_load:
            mock_load.side_effect = Exception("Failed to load from path_config")

            with pytest.raises(Exception, match=r".*No AV configuration provided or loaded\."):
                sim_av.run_simulation()

    @patch('realtwin.autonomous_veh.sim_av.run_sumo_simulation')
    @patch('realtwin.autonomous_veh.sim_av.create_sumo_config')
    @patch('realtwin.autonomous_veh.sim_av.add_veh_types_to_rou')
    @patch('realtwin.autonomous_veh.sim_av.create_sumo_rou_xml')
    @patch('realtwin.autonomous_veh.sim_av.update_sumo_flow_xml')
    @patch('realtwin.autonomous_veh.sim_av.create_veh_type_attributes')
    @patch('realtwin.autonomous_veh.sim_av.generate_sumo_loop_detector_add_xml')
    def test_run_simulation_function_exceptions(self, mock_gen_detector, mock_create_veh_attrs,
                                                mock_update_flow, mock_create_rou,
                                                mock_add_veh_types, mock_create_config,
                                                mock_run_sumo):
        """Test behavior when simulation functions raise exceptions"""
        # Create input files first
        for filename in ['test.net.xml', 'test.flow.xml', 'test.turn.xml']:
            (Path(self.test_dir) / filename).touch()

        # Setup mocks - make one function fail
        mock_create_veh_attrs.return_value = {}
        mock_gen_detector.return_value = True
        mock_update_flow.side_effect = Exception("Flow update failed")

        sim_av = SimAV(self.config_path)

        # Should propagate the exception
        with pytest.raises(Exception, match="Flow update failed"):
            sim_av.run_simulation()


class TestSimAVIntegration:
    """Integration tests combining multiple SimAV components"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_simav_lifecycle_methods(self):
        """Test complete SimAV lifecycle: init -> load_config -> prepare_config -> run"""
        # Create config file
        config_path = Path(self.test_dir) / 'config.yaml'
        config_path.touch()

        config_data = {
            'veh_types': ['Human', 'AV'],
            'pct_penetration': [70, 30],
            'input': {
                'input_dir': self.test_dir,
                'net_file': 'net.xml',
                'flow_file': 'flow.xml',
                'turn_file': 'turn.xml'
            }
        }

        # Initialize SimAV
        sim_av = SimAV(config_path)

        # Test prepare_av_configs
        with patch('realtwin.autonomous_veh.sim_av.prepare_av_configs') as mock_prepare:
            mock_prepare.return_value = True
            result = sim_av.prepare_av_configs(self.test_dir)
            assert result is True

        # Test load_config
        with patch('realtwin.autonomous_veh.sim_av.load_av_configs') as mock_load:
            mock_load.return_value = config_data
            result = sim_av.load_config(str(config_path))
            assert result is True
            assert hasattr(sim_av, 'av_config')

        # Test run_simulation with loaded config
        with patch('realtwin.autonomous_veh.sim_av.check_inputs_from_config') as mock_check:
            mock_check.return_value = False

            with pytest.raises(Exception, match=r".*Invalid inputs in the configuration file"):
                sim_av.run_simulation()

    def test_multiple_simav_instances(self):
        """Test that multiple SimAV instances work independently"""
        config1_path = Path(self.test_dir) / 'config1.yaml'
        config2_path = Path(self.test_dir) / 'config2.yaml'

        for config_path in [config1_path, config2_path]:
            config_path.touch()

        # Create two instances
        sim_av1 = SimAV(config1_path, verbose=True)
        sim_av2 = SimAV(config2_path, verbose=False)

        # Verify they're independent
        assert sim_av1.path_config != sim_av2.path_config
        assert sim_av1.verbose != sim_av2.verbose

        # Test that they can load different configs
        config1_data = {'type': 'config1', 'veh_types': ['Human']}
        config2_data = {'type': 'config2', 'veh_types': ['AV']}

        with patch('realtwin.autonomous_veh.sim_av.load_av_configs') as mock_load:
            mock_load.side_effect = lambda path: (
                config1_data if str(config1_path) in str(path) else config2_data
            )

            sim_av1.load_config(str(config1_path))
            sim_av2.load_config(str(config2_path))

            assert sim_av1.av_config['type'] == 'config1'
            assert sim_av2.av_config['type'] == 'config2'