# Real-Twin

**(Feel free to revise the document to enhance the accuracy and readability)**

The real-twin developed by ORNL Applied Research and Mobility System (ARMS) group that enables the simulation of twin-structured cities.

## Change Log

### 2024-12-27

* add .pylintrc file for project-wide coding control
* refactored code to reach production-level
* test the current functionalities: including create venv, delete venv, check SUMO and install SUMO

TODO:

* Discussion the repository license: MIT, Apache, BSD etc..., as we are writing code, we need copyright licence text as Docstring at the top of each file created.
* Add test functions perform object safety check

### 2024-12-26

* enable pylint for checking / evaluating coding for GitHub Action
* add code evaluation fuction (local checking) to perform code checking and evaluation (navigate to tests folder: pylint_proj_code_checker.py)
* update requirements.txt
* add official copyright for each document for the requirement of software development under ORNL
* add test functions

#### TODO:

* discuss the repository license: MIT, Apache, BSD etc..., as we are writing code, we need copyright licence text as Docstring at the top of each file created.
* add more test functions perform object safety check

### 2024-12-23

* re-design the realtwin development framework
* re-design the overall package framework
* add utility functions: is_sumo_installed
* add utility functions: venv_create, venv_delete
* add func_lib in install_simulator: install_sumo_windows
* add test function: test_is_sumo_installed
* ![1734989040637](docs/readme/1734989040637.png)

#### TODO:

* Test the overall functionalities
