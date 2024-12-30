- [Run the realtwin package on your device (Code Implementation and Explanation)](#run-the-realtwin-package-on-your-device-code-implementation-and-explanation)
  - [Installation](#installation)
  - [Create and delete virtual environment (Optional)](#create-and-delete-virtual-environment-optional)

    - [Create venv](#create-venv)
    - [Activate venv](#activate-venv)
    - [Delete venv](#delete-venv)
  - [Setup simulation environment](#setup-simulation-environment)


# Run the realtwin package on your device (Code Implementation and Explanation)

## Installation

Please note, the packatge is not yet available on PyPI (please wait to be noticed by ARMS group from ORNL).

`pip install realtwin `

NOTE: For developers, you should clone the repository from GitHub (private repository): https://github.com/Real-Sim-XIL/Real-Twin

and install dependencies:

pip install -r requirements.txt

## Create and delete virtual environment (Optional)

If you wanted to run the realtwin in an isolated virtual environment that not appecting your existiong working environment, please follow the following steps.

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

## Setup simulation environment

```python
import realtwin as rt

if __name__ == "__main__":

    INPUT_DIR = "path_to_your_input_data"

    # initialize the realtwin object
    twin = rt.REALTWIN(input_dir=INPUT_DIR)

    # environment setup
    # Check if SUMO, VISSIM, AIMSUN, etc... are installed
    twin.env_setup(sel_sim=["SUMO", "VISSIM"])


```
