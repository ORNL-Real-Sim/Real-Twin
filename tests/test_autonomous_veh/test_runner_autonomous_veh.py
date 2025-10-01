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
"""
Test runner for autonomous vehicle module tests

This module provides a convenient way to run all autonomous vehicle related tests.
"""

import pytest
import sys
from pathlib import Path


def run_autonomous_veh_tests():
    """
    Run all autonomous vehicle tests
    
    Returns:
        int: pytest exit code (0 for success, non-zero for failure)
    """
    
    # Get the directory where this file is located
    test_dir = Path(__file__).parent
    
    # Define test files for autonomous vehicle module
    test_files = [
        "test_autonomous_veh.py",
        "test_carfollowing_model.py"
    ]
    
    # Check if test files exist
    existing_files = []
    for test_file in test_files:
        file_path = test_dir / test_file
        if file_path.exists():
            existing_files.append(str(file_path))
        else:
            print(f"Warning: Test file {test_file} not found at {file_path}")
    
    if not existing_files:
        print("Error: No autonomous vehicle test files found!")
        return 1
    
    # Run pytest on the existing test files
    print(f"Running autonomous vehicle tests: {existing_files}")
    
    # Configure pytest arguments
    pytest_args = [
        "-v",  # verbose output
        "--tb=short",  # shorter traceback format
        "--color=yes",  # colored output
        *existing_files
    ]
    
    # Run the tests
    return pytest.main(pytest_args)


def run_specific_test_class(test_class_name: str):
    """
    Run a specific test class from the autonomous vehicle tests
    
    Args:
        test_class_name (str): Name of the test class to run
        
    Returns:
        int: pytest exit code
    """
    
    test_dir = Path(__file__).parent
    
    # Map of test class names to their files
    class_to_file = {
        "TestLoadAVConfigs": "test_autonomous_veh.py",
        "TestPrepareAVConfigs": "test_autonomous_veh.py", 
        "TestSimAV": "test_autonomous_veh.py",
        "TestCheckInputsFromConfig": "test_autonomous_veh.py",
        "TestNameWithoutSuffixes": "test_autonomous_veh.py",
        "TestGenerateSumoLoopDetectorAddXml": "test_autonomous_veh.py",
        "TestIntegration": "test_autonomous_veh.py",
        "TestCarFollowingLaneChangingModel": "test_carfollowing_model.py"
    }
    
    if test_class_name not in class_to_file:
        print(f"Error: Test class '{test_class_name}' not found!")
        print(f"Available test classes: {list(class_to_file.keys())}")
        return 1
    
    test_file = test_dir / class_to_file[test_class_name]
    if not test_file.exists():
        print(f"Error: Test file {test_file} not found!")
        return 1
    
    # Run specific test class
    pytest_args = [
        "-v",
        "--tb=short", 
        "--color=yes",
        f"{test_file}::{test_class_name}"
    ]
    
    print(f"Running test class: {test_class_name}")
    return pytest.main(pytest_args)


if __name__ == "__main__":
    """
    Command line interface for running autonomous vehicle tests
    
    Usage:
        python test_runner_autonomous_veh.py                    # Run all AV tests
        python test_runner_autonomous_veh.py TestSimAV          # Run specific test class
    """
    
    if len(sys.argv) == 1:
        # Run all autonomous vehicle tests
        exit_code = run_autonomous_veh_tests()
    elif len(sys.argv) == 2:
        # Run specific test class
        test_class = sys.argv[1]
        exit_code = run_specific_test_class(test_class)
    else:
        print("Usage:")
        print("  python test_runner_autonomous_veh.py                    # Run all AV tests")
        print("  python test_runner_autonomous_veh.py <TestClassName>    # Run specific test class")
        exit_code = 1
    
    sys.exit(exit_code)