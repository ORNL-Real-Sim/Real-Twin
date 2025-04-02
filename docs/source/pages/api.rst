=====================
Classes and Functions
=====================

.. currentmodule:: realtwin


RealTwin API
==============
.. autosummary::
    :toctree: api/

    _realtwin.RealTwin


Utility Functions
=================
.. autosummary::
    :toctree: api/

    utils_lib.venv_create
    utils_lib.venv_delete
    utils_lib.download_elevation_tif_by
    utils_lib.download_single_file_from_web
    utils_lib.find_executable_from_PATH_on_win

Installation and Environment
============================
.. autosummary::
    :toctree: api/

    func_lib._a_install_simulator.install_sumo
    func_lib._a_install_simulator.install_sumo_windows
    func_lib._a_install_simulator.install_sumo_linux
    func_lib._a_install_simulator.install_sumo_macos

    func_lib._a_install_simulator.is_sumo_installed
    func_lib._a_install_simulator.is_aimsun_installed
    func_lib._a_install_simulator.is_vissim_installed

Load Inputs
===========
.. autosummary::
    :toctree: api/

    func_lib._b_load_inputs.load_input_config
    func_lib._b_load_inputs.get_bounding_box_from

Abstract Scenario Generation
============================
.. autosummary::
    :toctree: api/

    func_lib._c_abstract_scenario.AbstractScenario
    func_lib._c_abstract_scenario.load_traffic_volume
    func_lib._c_abstract_scenario.load_control_signal
    func_lib._c_abstract_scenario.load_traffic_turning_ratio
    func_lib._c_abstract_scenario.OpenDriveNetwork
    func_lib._c_abstract_scenario.OSMRoad

Concrete Scenario Generation
============================
.. autosummary::
    :toctree: api/

    func_lib._d_concrete_scenario.ConcreteScenario

Prepare Simulation Documents
============================
.. autosummary::
    :toctree: api/

    func_lib._e_simulation.SimPrep
    func_lib._e_simulation.SUMOPrep
    func_lib._e_simulation.AimsunPrep
    func_lib._e_simulation.VissimPrep

Calibration
===========
.. autosummary::
    :toctree: api/

    func_lib._f_calibration.cali_sumo
    func_lib._f_calibration.cali_aimsun
    func_lib._f_calibration.cali_vissim
    func_lib._f_calibration.BehaviorCalib
    func_lib._f_calibration.TurnInflowCalib
