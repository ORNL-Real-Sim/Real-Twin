# Autonomous Vehicle Module Tests

This directory contains comprehensive tests for the RealTwin autonomous vehicle simulation module.

## Test Files Overview

### `test_autonomous_veh.py`
Main test suite for the autonomous vehicle simulation functionality:

- **TestLoadAVConfigs**: Tests for configuration loading (`load_av_configs`)
  - Valid/invalid configuration files
  - Parameter validation (penetration percentages, vehicle types)
  - Error handling for missing files, invalid formats

- **TestPrepareAVConfigs**: Tests for configuration file generation (`prepare_av_configs`)
  - Directory handling (existing/non-existent)
  - File copying functionality

- **TestSimAV**: Tests for the main SimAV class
  - Initialization with various parameters
  - Configuration loading and validation
  - Simulation execution workflow
  - Error handling for invalid inputs

- **TestCheckInputsFromConfig**: Tests for input validation
  - Directory existence checks
  - Required file validation (network, flow, turn files)
  - Error reporting for missing components

- **TestNameWithoutSuffixes**: Tests for filename utility function
  - Single and multiple suffix removal
  - Edge cases with no suffixes

- **TestGenerateSumoLoopDetectorAddXml**: Tests for loop detector generation
  - Different detector types (E1, E2, E0)
  - XML file generation and content validation
  - Custom output filenames and extensions
  - Traffic light intersection detection

- **TestIntegration**: Integration tests combining multiple components

### `test_carfollowing_model.py`
Tests for car-following and lane-changing model parameters:

- **TestCarFollowingLaneChangingModel**: Comprehensive tests for model parameters
  - Model structure validation
  - Parameter type checking
  - Boundary value testing
  - Consistency checks across models
  - Vehicle type availability

## Running Tests

### Run All Autonomous Vehicle Tests
```bash
# From the tests directory
python test_runner_autonomous_veh.py

# Or using pytest directly
python -m pytest test_autonomous_veh.py test_carfollowing_model.py -v
```

### Run Specific Test Classes
```bash
# Run specific test class
python test_runner_autonomous_veh.py TestSimAV

# Or using pytest
python -m pytest test_autonomous_veh.py::TestSimAV -v
```

### Run Individual Test Methods
```bash
# Run specific test method
python -m pytest test_autonomous_veh.py::TestSimAV::test_sim_av_init_success -v
```

## Test Coverage Areas

### Configuration Management
- ✅ YAML file loading and validation
- ✅ Parameter type checking and boundary validation
- ✅ Vehicle type and penetration rate validation
- ✅ Error handling for malformed configurations

### Simulation Setup
- ✅ SimAV class initialization
- ✅ Input file validation (network, flow, turn files)
- ✅ Directory structure handling
- ✅ Configuration file generation

### Loop Detector Generation
- ✅ SUMO network parsing for traffic light intersections
- ✅ XML generation for different detector types
- ✅ Lane identification and detector placement
- ✅ Output file formatting

### Car-Following Models
- ✅ Parameter structure validation for all models:
  - Wied99 (10 parameters: cc0-cc9)
  - Krauss (6 parameters: minGap, accel, decel, sigma, tau, emergencyDecel)
  - IDM (6 parameters: minGap, accel, decel, tau, emergencyDecel, delta)
  - CoopVisWied99 (5 parameters: lookAhead/Back distances, interaction objects)
  - LCVisWied99 (2 parameters: safety factor, cooperation flag)
- ✅ Parameter bounds and consistency checking
- ✅ Vehicle type availability (Human, AVnormal, AVsafe, AVaggressive)

### Error Handling
- ✅ File not found scenarios
- ✅ Invalid configuration parameters
- ✅ Missing required inputs
- ✅ Type validation and conversion

## Mock Dependencies

The tests use extensive mocking to isolate functionality:

- **File System Operations**: `pathlib.Path`, `shutil`, `os` operations
- **YAML Loading**: Configuration file parsing
- **SUMO Utilities**: Network parsing and XML generation
- **External Dependencies**: `pyufunc` and other utility libraries

## Test Data

Tests create temporary directories and files to avoid dependencies on external data files:

- Minimal SUMO network XML files with traffic lights
- Test configuration YAML files
- Temporary input/output directories

## Continuous Integration

These tests are designed to run in CI/CD environments:

- No external dependencies on SUMO installation
- Self-contained test data
- Platform-independent file operations
- Comprehensive error condition coverage

## Adding New Tests

When adding new functionality to the autonomous vehicle module:

1. Add corresponding test methods to the appropriate test class
2. Follow the existing naming convention: `test_<functionality>_<scenario>`
3. Include both positive and negative test cases
4. Mock external dependencies appropriately
5. Update this documentation with new test coverage

## Dependencies

Required packages for running tests:
- `pytest`
- `unittest.mock` (built-in)
- `yaml` 
- `pathlib` (built-in)
- `tempfile` (built-in)

The tests are designed to be self-contained and not require SUMO or other external simulation tools to be installed.