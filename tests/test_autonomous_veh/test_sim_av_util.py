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
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

try:
    import realtwin
except ImportError:
    # If realtwin is not installed, use the local path
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import realtwin

from realtwin.autonomous_veh._sim_av_util import (
    prettify_xml,
    generate_rgb_colors,
    create_veh_type_attributes,
    update_sumo_flow_xml,
    create_sumo_rou_xml,
    add_veh_types_to_rou,
    create_sumo_config,
    run_sumo_simulation
)


class TestPrettifyXml:
    """Test the prettify_xml function"""

    def test_prettify_xml_simple(self):
        """Test prettifying a simple XML structure"""
        root = ET.Element("root")
        child = ET.SubElement(root, "child")
        child.text = "content"
        tree = ET.ElementTree(root)

        result = prettify_xml(tree)

        assert isinstance(result, str)
        assert "<?xml version=" in result
        assert "<root>" in result
        assert "<child>content</child>" in result
        assert "    " in result  # Check for indentation

    def test_prettify_xml_nested(self):
        """Test prettifying a nested XML structure"""
        root = ET.Element("config")
        input_elem = ET.SubElement(root, "input")
        net_file = ET.SubElement(input_elem, "net-file")
        net_file.set("value", "test.net.xml")
        tree = ET.ElementTree(root)

        result = prettify_xml(tree)

        assert "<config>" in result
        assert "<input>" in result
        assert '<net-file value="test.net.xml"' in result

    def test_prettify_xml_empty(self):
        """Test prettifying an empty XML structure"""
        root = ET.Element("empty")
        tree = ET.ElementTree(root)

        result = prettify_xml(tree)

        assert "<empty/>" in result or "<empty></empty>" in result


class TestGenerateRgbColors:
    """Test the generate_rgb_colors function"""

    def test_generate_rgb_colors_basic(self):
        """Test generating a basic set of RGB colors"""
        num_colors = 3
        result = generate_rgb_colors(num_colors)

        assert len(result) == num_colors
        assert all(isinstance(color, str) for color in result)

        # Check format (r,g,b)
        for color in result:
            parts = color.split(',')
            assert len(parts) == 3
            for part in parts:
                value = int(part)
                assert 0 <= value <= 255

    def test_generate_rgb_colors_single(self):
        """Test generating a single RGB color"""
        result = generate_rgb_colors(1)

        assert len(result) == 1
        assert result[0] == "255,0,0"  # First color should be pure red

    def test_generate_rgb_colors_zero(self):
        """Test generating zero RGB colors"""
        result = generate_rgb_colors(0)

        assert len(result) == 0
        assert result == []

    def test_generate_rgb_colors_large(self):
        """Test generating a large number of RGB colors"""
        num_colors = 10
        result = generate_rgb_colors(num_colors)

        assert len(result) == num_colors
        # Check that all colors are different (should be with HSV distribution)
        assert len(set(result)) == num_colors

    def test_generate_rgb_colors_format(self):
        """Test that RGB colors are in correct format"""
        result = generate_rgb_colors(5)

        for color in result:
            # Should be in format "r,g,b" where r,g,b are integers 0-255
            assert isinstance(color, str)
            parts = color.split(',')
            assert len(parts) == 3

            for part in parts:
                value = int(part)  # Should not raise ValueError
                assert 0 <= value <= 255


class TestCreateVehTypeAttributes:
    """Test the create_veh_type_attributes function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.valid_av_config = {
            'veh_types': ['Human', 'AVnormal', 'AVsafe'],
            'pct_penetration': [50, 30, 20],
            'car_follow_model': 'Krauss',
            'lane_change_model': 'LC2013',
            'CFmodel': {
                'Human': {
                    'Krauss': {
                        'accel': 2.6,
                        'decel': 4.5,
                        'minGap': 2.5,
                        'sigma': 0.5,
                        'tau': 1.0,
                        'emergencyDecel': 9.0
                    }
                },
                'AVnormal': {
                    'Krauss': {
                        'accel': 3.0,
                        'decel': 5.0,
                        'minGap': 2.0,
                        'sigma': 0.2,
                        'tau': 0.8,
                        'emergencyDecel': 9.5
                    }
                },
                'AVsafe': {
                    'Krauss': {
                        'accel': 2.0,
                        'decel': 6.0,
                        'minGap': 3.0,
                        'sigma': 0.1,
                        'tau': 1.2,
                        'emergencyDecel': 10.0
                    }
                }
            }
        }

    def test_create_veh_type_attributes_success(self):
        """Test successful creation of vehicle type attributes"""
        result = create_veh_type_attributes(self.valid_av_config)

        assert isinstance(result, dict)
        assert len(result) == 3

        # Check that all vehicle types are present
        for veh_type in ['Human', 'AVnormal', 'AVsafe']:
            assert veh_type in result

            # Check required attributes
            attrs = result[veh_type]
            assert attrs['id'] == veh_type
            assert attrs['vClass'] == 'passenger'
            assert attrs['carFollowModel'] == 'Krauss'
            assert attrs['laneChangeModel'] == 'LC2013'

            # Check probability calculation
            expected_prob = self.valid_av_config['pct_penetration'][
                self.valid_av_config['veh_types'].index(veh_type)] * 0.01
            assert attrs['probability'] == expected_prob

    def test_create_veh_type_attributes_default_models(self):
        """Test creation with default car follow and lane change models"""
        config = self.valid_av_config.copy()
        del config['car_follow_model']
        del config['lane_change_model']

        result = create_veh_type_attributes(config)

        for veh_type in result:
            assert result[veh_type]['carFollowModel'] == 'Krauss'  # Default
            assert result[veh_type]['laneChangeModel'] == 'LC2013'  # Default

    def test_create_veh_type_attributes_empty_config(self):
        """Test with empty or missing configuration"""
        empty_config = {}
        result = create_veh_type_attributes(empty_config)

        assert result == {}

    def test_create_veh_type_attributes_cf_parameters(self):
        """Test that car-following parameters are correctly mapped"""
        result = create_veh_type_attributes(self.valid_av_config)

        human_attrs = result['Human']
        cf_params = self.valid_av_config['CFmodel']['Human']['Krauss']

        assert human_attrs['accel'] == cf_params['accel']
        assert human_attrs['decel'] == cf_params['decel']
        assert human_attrs['minGap'] == cf_params['minGap']
        assert human_attrs['sigma'] == cf_params['sigma']
        assert human_attrs['tau'] == cf_params['tau']
        assert human_attrs['emergencyDecel'] == cf_params['emergencyDecel']


class TestUpdateSumoFlowXml:
    """Test the update_sumo_flow_xml function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.flow_file = Path(self.test_dir) / 'test_flow.xml'

        # Create a basic flow XML file
        flow_content = '''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <flow id="flow1" from="edge1" to="edge2" vehsPerHour="100" begin="0" end="3600"/>
    <flow id="flow2" from="edge2" to="edge3" vehsPerHour="150" begin="0" end="3600"/>
</routes>'''

        with open(self.flow_file, 'w') as f:
            f.write(flow_content)

        self.veh_types = ['Human', 'AV']
        self.veh_type_attributes = {
            'Human': {
                'id': 'Human',
                'vClass': 'passenger',
                'carFollowModel': 'Krauss',
                'probability': 0.7,
                'accel': 2.6,
                'decel': 4.5
            },
            'AV': {
                'id': 'AV',
                'vClass': 'passenger',
                'carFollowModel': 'IDM',
                'probability': 0.3,
                'accel': 3.0,
                'decel': 5.0
            }
        }

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_update_sumo_flow_xml_success(self):
        """Test successful update of SUMO flow XML"""
        result = update_sumo_flow_xml(str(self.flow_file), self.veh_types, self.veh_type_attributes)

        assert result is True

        # Parse the updated file
        tree = ET.parse(self.flow_file)
        root = tree.getroot()

        # Check vTypeDistribution element
        vtype_dist = root.find('vTypeDistribution')
        assert vtype_dist is not None
        assert vtype_dist.get('id') == 'vdis'

        # Check vehicle types
        vtypes = vtype_dist.findall('vType')
        assert len(vtypes) == 2

        # Check that flows have type='vdis'
        flows = root.findall('flow')
        assert len(flows) == 2
        for flow in flows:
            assert flow.get('type') == 'vdis'

    def test_update_sumo_flow_xml_colors(self):
        """Test that colors are assigned to vehicle types"""
        update_sumo_flow_xml(str(self.flow_file), self.veh_types, self.veh_type_attributes)

        tree = ET.parse(self.flow_file)
        root = tree.getroot()

        vtypes = root.find('vTypeDistribution').findall('vType')
        for vtype in vtypes:
            color = vtype.get('color')
            assert color is not None
            # Check color format (r,g,b)
            parts = color.split(',')
            assert len(parts) == 3
            for part in parts:
                assert 0 <= int(part) <= 255

    def test_update_sumo_flow_xml_attributes(self):
        """Test that vehicle type attributes are correctly set"""
        update_sumo_flow_xml(str(self.flow_file), self.veh_types, self.veh_type_attributes)

        tree = ET.parse(self.flow_file)
        root = tree.getroot()

        vtypes = root.find('vTypeDistribution').findall('vType')
        vtype_dict = {vt.get('id'): vt for vt in vtypes}

        # Check Human vehicle type
        human_vtype = vtype_dict['Human']
        assert human_vtype.get('vClass') == 'passenger'
        assert human_vtype.get('carFollowModel') == 'Krauss'
        assert abs(float(human_vtype.get('probability')) - 0.7) < 0.001


class TestCreateSumoRouXml:
    """Test the create_sumo_rou_xml function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.net_file = Path(self.test_dir) / 'test.net.xml'
        self.flow_file = Path(self.test_dir) / 'test.flow.xml'
        self.turn_file = Path(self.test_dir) / 'test.turn.xml'
        self.rou_file = Path(self.test_dir) / 'test.rou.xml'

        # Create dummy files
        for file_path in [self.net_file, self.flow_file, self.turn_file]:
            file_path.touch()

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('realtwin.autonomous_veh._sim_av_util.os.system')
    def test_create_sumo_rou_xml_success(self, mock_system):
        """Test successful creation of SUMO route XML"""
        mock_system.return_value = 0

        result = create_sumo_rou_xml(
            str(self.net_file),
            str(self.flow_file),
            str(self.turn_file),
            str(self.rou_file)
        )

        assert result is True
        mock_system.assert_called_once()

        # Check that jtrrouter command was called with correct parameters
        call_args = mock_system.call_args[0][0]
        assert 'jtrrouter' in call_args
        assert f'--route-files={self.flow_file}' in call_args
        assert f'--turn-ratio-files={self.turn_file}' in call_args
        assert f'--net-file={self.net_file}' in call_args
        assert f'--output-file={self.rou_file}' in call_args


class TestAddVehTypesToRou:
    """Test the add_veh_types_to_rou function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.rou_file = Path(self.test_dir) / 'test.rou.xml'

        # Create a basic route XML file
        rou_content = '''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <route id="route1" edges="edge1 edge2 edge3"/>
    <vehicle id="veh1" route="route1" depart="0"/>
</routes>'''

        with open(self.rou_file, 'w') as f:
            f.write(rou_content)

        self.veh_types = ['Human', 'AV']
        self.veh_type_attributes = {
            'Human': {'id': 'Human', 'accel': 2.6, 'decel': 4.5},
            'AV': {'id': 'AV', 'accel': 3.0, 'decel': 5.0}
        }

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_add_veh_types_to_rou_success(self):
        """Test successful addition of vehicle types to route file"""
        result = add_veh_types_to_rou(str(self.rou_file), self.veh_types, self.veh_type_attributes)

        assert result is True

        # Parse the updated file
        tree = ET.parse(self.rou_file)
        root = tree.getroot()

        # Check that vTypes were added at the beginning
        vtypes = root.findall('vType')
        assert len(vtypes) == 4

        # Check that original elements are still there
        routes = root.findall('route')
        vehicles = root.findall('vehicle')
        assert len(routes) == 1
        assert len(vehicles) == 1

    def test_add_veh_types_to_rou_attributes(self):
        """Test that vehicle type attributes are correctly added"""
        add_veh_types_to_rou(str(self.rou_file), self.veh_types, self.veh_type_attributes)

        tree = ET.parse(self.rou_file)
        root = tree.getroot()

        vtypes = root.findall('vType')
        vtype_dict = {vt.get('id'): vt for vt in vtypes}

        # Check attributes
        human_vtype = vtype_dict['Human']
        assert human_vtype.get('id') == 'Human'
        assert abs(float(human_vtype.get('accel')) - 2.6) < 0.001

        # Check colors are assigned
        for vtype in vtypes:
            assert vtype.get('color') is not None

    def test_add_veh_types_to_rou_insertion_order(self):
        """Test that vehicle types are inserted at the beginning"""
        add_veh_types_to_rou(str(self.rou_file), self.veh_types, self.veh_type_attributes)

        tree = ET.parse(self.rou_file)
        root = tree.getroot()

        # First elements should be vTypes
        first_elements = list(root)[:2]
        assert all(elem.tag == 'vType' for elem in first_elements)


class TestCreateSumoConfig:
    """Test the create_sumo_config function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = Path(self.test_dir) / 'test.sumocfg'

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_create_sumo_config_default_params(self):
        """Test creation of SUMO config with default parameters"""
        result = create_sumo_config(str(self.config_file))

        assert result is True
        assert self.config_file.exists()

        # Parse the created config file
        tree = ET.parse(self.config_file)
        root = tree.getroot()

        assert root.tag == 'configuration'

        # Check input section
        input_elem = root.find('input')
        assert input_elem is not None

        net_file = input_elem.find('net-file')
        assert net_file.get('value') == 'chatt.net.xml'  # Default sim_name

        rou_file = input_elem.find('route-files')
        assert rou_file.get('value') == 'chatt.rou.xml'

    def test_create_sumo_config_custom_params(self):
        """Test creation of SUMO config with custom parameters"""
        result = create_sumo_config(
            str(self.config_file),
            sim_name='custom_sim',
            sim_start=1800,
            sim_end=5400
        )

        assert result is True

        tree = ET.parse(self.config_file)
        root = tree.getroot()

        # Check custom parameters
        input_elem = root.find('input')
        net_file = input_elem.find('net-file')
        assert net_file.get('value') == 'custom_sim.net.xml'

        time_elem = root.find('time')
        begin_elem = time_elem.find('begin')
        end_elem = time_elem.find('end')
        assert begin_elem.get('value') == '1800'
        assert end_elem.get('value') == '5400'

    def test_create_sumo_config_structure(self):
        """Test that all required sections are created in config"""
        create_sumo_config(str(self.config_file))

        tree = ET.parse(self.config_file)
        root = tree.getroot()

        # Check all major sections exist
        assert root.find('input') is not None
        assert root.find('output') is not None
        assert root.find('time') is not None
        assert root.find('gui_only') is not None
        assert root.find('report') is not None

    def test_create_sumo_config_output_files(self):
        """Test that output files are correctly configured"""
        create_sumo_config(str(self.config_file), sim_name='test')

        tree = ET.parse(self.config_file)
        root = tree.getroot()

        output_elem = root.find('output')

        full_output = output_elem.find('full-output')
        assert full_output.get('value') == 'test_Full_Output.xml'

        amitran_output = output_elem.find('amitran-output')
        assert amitran_output.get('value') == 'test_Amitran_Output.xml'

    def test_create_sumo_config_time_settings(self):
        """Test time configuration settings"""
        create_sumo_config(str(self.config_file))

        tree = ET.parse(self.config_file)
        root = tree.getroot()

        time_elem = root.find('time')
        step_length = time_elem.find('step-length')
        assert step_length.get('value') == '0.1'


class TestRunSumoSimulation:
    """Test the run_sumo_simulation function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = Path(self.test_dir) / 'test.sumocfg'
        self.config_file.touch()

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('realtwin.autonomous_veh._sim_av_util.traci')
    def test_run_sumo_simulation_success(self, mock_traci):
        """Test successful SUMO simulation run"""
        # Mock traci behavior
        mock_traci.simulation.getTime.side_effect = [0, 1800, 3600]  # Simulation progresses

        result = run_sumo_simulation(str(self.config_file), sim_time=3600)

        assert result is True

        # Check traci calls
        mock_traci.start.assert_called_once_with(['sumo-gui', '-c', str(self.config_file)])
        assert mock_traci.simulationStep.call_count == 2  # Called twice before reaching 3600
        mock_traci.close.assert_called_once()

    @patch('realtwin.autonomous_veh._sim_av_util.traci')
    def test_run_sumo_simulation_custom_time(self, mock_traci):
        """Test SUMO simulation with custom simulation time"""
        mock_traci.simulation.getTime.side_effect = [0, 900, 1800]  # Shorter simulation

        result = run_sumo_simulation(str(self.config_file), sim_time=1800)

        assert result is True
        mock_traci.start.assert_called_once()
        mock_traci.close.assert_called_once()

    @patch('realtwin.autonomous_veh._sim_av_util.traci')
    def test_run_sumo_simulation_default_time(self, mock_traci):
        """Test SUMO simulation with default time parameter"""
        mock_traci.simulation.getTime.side_effect = [0, 1800, 3600]

        result = run_sumo_simulation(str(self.config_file))  # No sim_time specified

        assert result is True
        # Should use default 3600 seconds
        mock_traci.start.assert_called_once()

    @patch('realtwin.autonomous_veh._sim_av_util.traci')
    def test_run_sumo_simulation_zero_time(self, mock_traci):
        """Test SUMO simulation with zero simulation time"""
        mock_traci.simulation.getTime.return_value = 0

        result = run_sumo_simulation(str(self.config_file), sim_time=0)

        assert result is True
        mock_traci.start.assert_called_once()
        mock_traci.close.assert_called_once()
        # simulationStep should not be called since time is already >= 0
        mock_traci.simulationStep.assert_not_called()


class TestIntegration:
    """Integration tests for sim_av_util functions"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_xml_workflow_integration(self):
        """Test integrated XML workflow"""
        # Create vehicle type attributes
        av_config = {
            'veh_types': ['Human', 'AV'],
            'pct_penetration': [70, 30],
            'car_follow_model': 'Krauss',
            'CFmodel': {
                'Human': {
                    'Krauss': {
                        'accel': 2.6, 'decel': 4.5, 'minGap': 2.5,
                        'sigma': 0.5, 'tau': 1.0, 'emergencyDecel': 9.0
                    }
                },
                'AV': {
                    'Krauss': {
                        'accel': 3.0, 'decel': 5.0, 'minGap': 2.0,
                        'sigma': 0.2, 'tau': 0.8, 'emergencyDecel': 9.5
                    }
                }
            }
        }

        veh_type_attrs = create_veh_type_attributes(av_config)

        # Create flow file
        flow_file = Path(self.test_dir) / 'test.flow.xml'
        flow_content = '''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <flow id="flow1" from="edge1" to="edge2" vehsPerHour="100"/>
</routes>'''
        with open(flow_file, 'w') as f:
            f.write(flow_content)

        # Update flow file
        result = update_sumo_flow_xml(str(flow_file), av_config['veh_types'], veh_type_attrs)
        assert result is True

        # Create route file
        rou_file = Path(self.test_dir) / 'test.rou.xml'
        rou_content = '''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <route id="route1" edges="edge1 edge2"/>
</routes>'''
        with open(rou_file, 'w') as f:
            f.write(rou_content)

        # Add vehicle types to route file
        result = add_veh_types_to_rou(str(rou_file), av_config['veh_types'], veh_type_attrs)
        assert result is True

        # Create config file
        config_file = Path(self.test_dir) / 'test.sumocfg'
        result = create_sumo_config(str(config_file), sim_name='integration_test')
        assert result is True

        # Verify all files exist and have correct content
        assert flow_file.exists()
        assert rou_file.exists()
        assert config_file.exists()

        # Check that flow file has vTypeDistribution
        flow_tree = ET.parse(flow_file)
        assert flow_tree.getroot().find('vTypeDistribution') is not None

        # Check that route file has vTypes
        rou_tree = ET.parse(rou_file)
        vtypes = rou_tree.getroot().findall('vType')
        assert len(vtypes) == 4

    def test_color_consistency(self):
        """Test that colors are consistent across functions"""
        num_colors = 3
        colors1 = generate_rgb_colors(num_colors)
        colors2 = generate_rgb_colors(num_colors)

        # Same number of colors should generate same sequence
        assert colors1 == colors2

        # Different number should generate different sequences
        colors3 = generate_rgb_colors(num_colors + 1)
        assert colors1 != colors3[:num_colors]  # First n colors should be different
