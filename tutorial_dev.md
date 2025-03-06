- [Run the realtwin package on your device (Implementation and Explanation)](#run-the-realtwin-package-on-your-device-implementation-and-explanation)
  - [Code Health \& Standards Check (Optional - for developers)](#code-health--standards-check-optional---for-developers)
  - [Installation](#installation)
  - [Create \& Delete Virtual Environment (Optional)](#create--delete-virtual-environment-optional)
    - [Create venv](#create-venv)
    - [Activate venv](#activate-venv)
    - [Delete venv](#delete-venv)
  - [Simulation Environment Setup](#simulation-environment-setup)
    - [General Setup](#general-setup)
    - [Check simulator under additonal directories: Optional](#check-simulator-under-additonal-directories-optional)
    - [Strict simulator version: if not found, will install the strick version then (SUMO Only): Optional](#strict-simulator-version-if-not-found-will-install-the-strick-version-then-sumo-only-optional)
    - [Available arguments for environment setup](#available-arguments-for-environment-setup)
  - [Abstract Scenario Generation](#abstract-scenario-generation)
  - [Concrete Scenario Generation](#concrete-scenario-generation)
  - [Simulation (Preparation)](#simulation-preparation)
  - [Calibration](#calibration)

# Run the realtwin package on your device (Implementation and Explanation)

**This tutorial will eventually be embedded in README.md (currently maintained as a single file for development).**

## Code Health & Standards Check (Optional - for developers)

Ensure the project code is well-maintained and adheres to established standards.

```python
# Step 1: navigate to the file: tests/proj_std_check.py

# Step 2: run the file and then the report file will be generated under foldedr: docs/code_evaluation

# Step 3: fix bugs or refactor codes based on the generated report and push onto GitHub
```

## Installation

Please note, the packatge is not yet available on PyPI (Stay tuned to be noticed by ARMS group from ORNL).

`pip install realtwin `

NOTE: For developers, you should clone the repository from GitHub (private repository): https://github.com/Real-Sim-XIL/Real-Twin

and install dependencies:

`pip install -r requirements.txt`

## Create & Delete Virtual Environment (Optional)

If you want to run the realtwin in an isolated virtual environment that not affecting your existiong working environment, please follow the following steps.

### Create venv

```python
import realtwin as rt

# venv_dir: the directory to install virtual env. Default to be current folder
rt.venv_create(venv_name="venv_rt", venv_dir="")
```

### Activate venv

In order to activate your venv, please be aware the different IDE (**integrated development environment**) may need different actions.

For cmd:

    https://python.land/virtual-environments/virtualenv

For VS Code:

    https://code.visualstudio.com/docs/python/environments

For PyCharm:

    https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html

### Delete venv

```python
import realtwin as rt

# venv_dir: if the directory not provided, the function will find the venv under current folder
rt.venv_delete(venv_name="venv_rt", venv_dir="")
```

## Simulation Environment Setup

### General Setup

```python
import realtwin as rt

if __name__ == "__main__":

    # Prepare your configuration file (in YAML format)
    CONFIG_FILE = "./public_configs.yaml"

    # initialize the realtwin object
    twin = rt.RealTwin(input_config_file=CONFIG_FILE, verbose=True)

    # environment setup
    # Check if SUMO, VISSIM, AIMSUN, etc... are installed
    twin.env_setup(sel_sim=["SUMO", "VISSIM"])


```

### Check simulator under additonal directories: Optional

```python
import realtwin as rt

if __name__ == "__main__":

    # Prepare your configuration file (in YAML format)
    CONFIG_FILE = "./public_configs.yaml"

    # initialize the realtwin object
    twin = rt.RealTwin(input_config_file=CONFIG_FILE, verbose=True)

    # environment setup: Check if SUMO, VISSIM, AIMSUN, etc... are installed

    # NOTE: change the new_dir to your own directory where the SUMO is installed (multiple versions)
    new_dir = [f"~/SUMO/sumo-1.20.0/bin", "path-of-different-sumo-versons/bin",]

    twin.env_setup(sel_sim=["SUMO", "VISSIM"], sel_dir=new_dir) # Pls note, sel_dir should be a list
```

### Strict simulator version: if not found, will install the strick version then (SUMO Only): Optional

```python
import realtwin as rt

if __name__ == "__main__":

    # Prepare your configuration file (in YAML format)
    CONFIG_FILE = "./public_configs.yaml"

    # initialize the realtwin object
    twin = rt.RealTwin(input_config_file=CONFIG_FILE, verbose=True)

    # environment setup: Check if SUMO, VISSIM, AIMSUN, etc... are installed

    # NOTE: change the new_dir to your own directory where the SUMO is installed (multiple versions)
    new_dir = [f"~/SUMO/sumo-1.20.0/bin", "path-of-different-sumo-versons/bin",]

    twin.env_setup(sel_sim=["SUMO", "VISSIM"], sel_dir=new_dir)

    # if you have SUMO version 1.20.0 installed on your machine, and you want to run using version 1.21.0 (Not installed yet)
    twin.env_setup(sel_sim=["SUMO", "VISSIM"], sel_dir=new_dir, strict_sumo_version="1.21.0")
```

### Available arguments for environment setup

```python
import realtwin as rt

if __name__ == "__main__":

    # ~

    twin.env_setup(
        sel_sim = [],  # select simulator to run
        sel_dir = [],  # add additional dir that installed simulator
        strict_sumo_version="1.21.0",  # strick the simulator version to run
        strict_vissim_version="",  # strick the simulator version to run
        strict_aimsun_version="",  # strick the simulator version to run
        create_venv=False,  # create an isolate virtual environment (Navigate to: Create & Delete Virtual Environment)
        verbose=True,  # print out process message
    )

```

## Abstract Scenario Generation

```python
import realtwin as rt

if __name__ == "__main__":

    # ~ previous implementations

    # generate abstract scenario
    twin.generate_abstract_scenario(incl_elevation_tif=True)

    # incl_elevation_tif: if ture, will automatically download elevation file (.tif)
```

## Concrete Scenario Generation

```python
import realtwin as rt

if __name__ == "__main__":

    # ~ previous implementations

    # generate abstract scenario
    twin.generate_concrete_scenario()
```

## Simulation (Preparation)

```python
import realtwin as rt

if __name__ == "__main__":

    # ~ previous implementations

    # perform calibration
    # keyword arguments can be passed to specify the calibration options
    # or change from internal and external configuration files
    twin.calibrate(sel_algo={"turn_inflow": "GA", "behavior": "SA"})  # Available algos: GA, SA, TS
```

## Calibration
