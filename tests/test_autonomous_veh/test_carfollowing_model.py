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

""" Test the car-following and lane-changing model parameters """

from pathlib import Path
import pytest
# from tests.proj_add_sys_path import add_pkg_to_sys_path
# add_pkg_to_sys_path("realtwin")

try:
    import realtwin
except ImportError:
    # If realtwin is not installed, use the local path
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import realtwin

from realtwin.autonomous_veh._carfollowing_lanechanging_model import (
    CFmodel,
    CF_DEFAULT_PARAMETERS
)


class TestCarFollowingLaneChangingModel:
    """Test car-following and lane-changing model parameters"""

    def test_cfmodel_structure(self):
        """Test that CFmodel has the expected structure"""
        expected_vehicle_types = ['Human', 'AVnormal', 'AVsafe', 'AVaggressive']

        assert isinstance(CFmodel, dict)
        for veh_type in expected_vehicle_types:
            assert veh_type in CFmodel
            assert isinstance(CFmodel[veh_type], dict)

    def test_cf_default_parameters_structure(self):
        """Test that CF_DEFAULT_PARAMETERS has the expected structure"""
        expected_models = ['Wied99', 'Krauss', 'IDM', 'CoopVisWied99', 'LCVisWied99']

        assert isinstance(CF_DEFAULT_PARAMETERS, dict)
        for model in expected_models:
            assert model in CF_DEFAULT_PARAMETERS
            assert isinstance(CF_DEFAULT_PARAMETERS[model], dict)

    def test_wied99_parameters(self):
        """Test Wied99 model parameters"""
        wied99_params = CF_DEFAULT_PARAMETERS['Wied99']
        expected_params = ['cc0', 'cc1', 'cc2', 'cc3', 'cc4', 'cc5', 'cc6', 'cc7', 'cc8', 'cc9']

        for param in expected_params:
            assert param in wied99_params
            assert isinstance(wied99_params[param], (int, float))

    def test_krauss_parameters(self):
        """Test Krauss model parameters"""
        krauss_params = CF_DEFAULT_PARAMETERS['Krauss']
        expected_params = ['minGap', 'accel', 'decel', 'sigma', 'tau', 'emergencyDecel']

        for param in expected_params:
            assert param in krauss_params
            assert isinstance(krauss_params[param], (int, float))

        # Test specific parameter ranges/values
        assert krauss_params['minGap'] >= 0
        assert krauss_params['accel'] > 0
        assert krauss_params['decel'] > 0
        assert 0 <= krauss_params['sigma'] <= 1
        assert krauss_params['tau'] > 0
        assert krauss_params['emergencyDecel'] > 0

    def test_idm_parameters(self):
        """Test IDM model parameters"""
        idm_params = CF_DEFAULT_PARAMETERS['IDM']
        expected_params = ['minGap', 'accel', 'decel', 'tau', 'emergencyDecel', 'delta']

        for param in expected_params:
            assert param in idm_params
            assert isinstance(idm_params[param], (int, float))

        # Test specific parameter ranges/values
        assert idm_params['minGap'] >= 0
        assert idm_params['accel'] > 0
        assert idm_params['decel'] > 0
        assert idm_params['tau'] > 0
        assert idm_params['emergencyDecel'] > 0
        assert idm_params['delta'] > 0

    def test_coopviswied99_parameters(self):
        """Test CoopVisWied99 model parameters"""
        coop_params = CF_DEFAULT_PARAMETERS['CoopVisWied99']
        expected_params = ['minLookAheadDist', 'maxLookAheadDist', 'minLookBackDist',
                           'maxLookBackDist', 'noOfInteractObjects']

        for param in expected_params:
            assert param in coop_params
            assert isinstance(coop_params[param], (int, float))

        # Test logical parameter relationships
        assert coop_params['minLookAheadDist'] <= coop_params['maxLookAheadDist']
        assert coop_params['minLookBackDist'] <= coop_params['maxLookBackDist']
        assert coop_params['noOfInteractObjects'] >= 0

    def test_lcviswied99_parameters(self):
        """Test LCVisWied99 model parameters"""
        lc_params = CF_DEFAULT_PARAMETERS['LCVisWied99']

        assert 'safetyDistRedFact' in lc_params
        assert 'coopLaneChange' in lc_params

        assert isinstance(lc_params['safetyDistRedFact'], (int, float))
        assert isinstance(lc_params['coopLaneChange'], bool)

        # Test parameter ranges
        assert 0 <= lc_params['safetyDistRedFact'] <= 1

    def test_parameter_types(self):
        """Test that all parameters have appropriate types"""
        for model_name, model_params in CF_DEFAULT_PARAMETERS.items():
            assert isinstance(model_params, dict), f"Model {model_name} should be a dictionary"

            for param_name, param_value in model_params.items():
                assert isinstance(param_value, (int, float, bool)), \
                    f"Parameter {param_name} in {model_name} should be numeric or boolean"

    def test_cfmodel_initialization(self):
        """Test that CFmodel can be properly initialized for vehicle types"""
        # Test that we can create instances for each vehicle type
        for veh_type in CFmodel.keys():
            # Each vehicle type should be able to hold model parameters
            test_model = CFmodel[veh_type].copy()
            assert isinstance(test_model, dict)

    def test_parameter_consistency(self):
        """Test consistency of parameters across models"""
        # Test that emergency deceleration is always higher than normal deceleration
        for model_name in ['Krauss', 'IDM']:
            if model_name in CF_DEFAULT_PARAMETERS:
                params = CF_DEFAULT_PARAMETERS[model_name]
                if 'emergencyDecel' in params and 'decel' in params:
                    assert params['emergencyDecel'] >= params['decel'], \
                        f"Emergency deceleration should be >= normal deceleration in {model_name}"

    def test_model_completeness(self):
        """Test that all models have the minimum required parameters"""
        # Test Krauss model (most commonly used)
        krauss_required = ['minGap', 'accel', 'decel', 'tau']
        for param in krauss_required:
            assert param in CF_DEFAULT_PARAMETERS['Krauss'], \
                f"Required parameter {param} missing from Krauss model"

    def test_parameter_bounds(self):
        """Test that parameters are within reasonable bounds"""
        # Test acceleration values are positive and reasonable
        for model_name in ['Krauss', 'IDM']:
            if model_name in CF_DEFAULT_PARAMETERS:
                params = CF_DEFAULT_PARAMETERS[model_name]
                if 'accel' in params:
                    assert 0 < params['accel'] <= 10, \
                        f"Acceleration in {model_name} should be between 0 and 10 m/s²"
                if 'decel' in params:
                    assert 0 < params['decel'] <= 15, \
                        f"Deceleration in {model_name} should be between 0 and 15 m/s²"

    def test_vehicle_type_availability(self):
        """Test that all expected vehicle types are available"""
        expected_types = ['Human', 'AVnormal', 'AVsafe', 'AVaggressive']
        for veh_type in expected_types:
            assert veh_type in CFmodel, f"Vehicle type {veh_type} should be in CFmodel"