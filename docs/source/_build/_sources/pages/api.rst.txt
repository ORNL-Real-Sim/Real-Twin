======================
API Reference
======================

.. currentmodule:: realtwin


RealTwin API
==============

.. autosummary::
    :toctree: api/

    RealTwin


Utility Functions
=================

.. autosummary::
    :toctree: api/

    util_lib.venv_create
    util_lib.venv_delete
    util_lib.download_elevation_tif_by_bbox
    util_lib.download_single_file_from_web
    util_lib.find_executable_from_PATH_on_win
    util_lib.prepare_config_file


Installation and Environment
============================

.. autosummary::
    :toctree: api/

    func_lib.install_sumo
    func_lib.install_sumo_windows
    func_lib.install_sumo_linux
    func_lib.install_sumo_macos

    func_lib.is_sumo_installed
    func_lib.is_aimsun_installed
    func_lib.is_vissim_installed

Load Inputs
===========

.. autosummary::
    :toctree: api/

    func_lib.load_input_config
    func_lib.get_bounding_box_from

Abstract Scenario Generation
============================

.. autosummary::
    :toctree: api/

    func_lib.AbstractScenario
    func_lib.load_traffic_volume
    func_lib.load_control_signal
    func_lib.load_traffic_turning_ratio
    func_lib.OpenDriveNetwork
    func_lib.OSMRoad

Concrete Scenario Generation
============================

.. autosummary::
    :toctree: api/

    func_lib.ConcreteScenario

Prepare Simulation Documents
============================

.. autosummary::
    :toctree: api/

    func_lib.SimPrep
    func_lib.SUMOPrep
    func_lib.AimsunPrep
    func_lib.VissimPrep

Calibration
===========

.. autosummary::
    :toctree: api/

    func_lib.cali_sumo
    func_lib.cali_aimsun
    func_lib.cali_vissim
    func_lib.BehaviorCalib
    func_lib.TurnInflowCalib
