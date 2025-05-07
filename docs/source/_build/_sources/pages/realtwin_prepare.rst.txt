=====================
realtwin Preparation
=====================

.. contents:: There are several steps to prepare before the use of Real-Twin package. This includes:
   :depth: 3
   :local:
   :backlinks: none

How To Prepare Python Environment (:blue:`Optional`)
====================================================

If you want to run the realtwin in an isolated virtual environment that not affecting your existing working environment, please follow the following steps.

Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several ways to manager a Python virtual environment.
    - Using `venv`_ module in Python
    - Using `conda manage environments`_ package manager (`conda install`_)
    - Using `virtualenv`_ package

realtwin package provides a simple way to create a virtual environment using the `venv` module in Python.

.. code-block:: python
    :linenos:
    :emphasize-lines: 5

    import realtwin as rt

    # venv_name: the name of the virtual environment. Default to be "venv_rt"
    # venv_dir: the directory to install virtual env. Default to be current folder
    rt.venv_create(venv_name="venv_rt", venv_dir="")


Activate Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to activate your venv, please be aware the different IDE (**integrated development environment**) may need different actions.

For cmd:
    - https://python.land/virtual-environments/virtualenv

For VS Code:
    - https://code.visualstudio.com/docs/python/environments

For PyCharm:
    - https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html


Delete Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several ways to manager a Python virtual environment.
    - Using `venv`_ module in Python
    - Using `conda manage environments`_ package manager (`conda install`_)
    - Using `virtualenv`_ package

.. code-block:: python
    :linenos:
    :emphasize-lines: 5

    import realtwin as rt

    # venv_name: the name of the virtual environment. Default to be "venv_rt"
    # venv_dir: the directory to install virtual env. Default to be current folder
    rt.venv_delete(venv_name="venv_rt", venv_dir="")

.. _How To Prepare Elevation Map In TIFF Format:

How To Prepare Elevation Map In TIFF Format (:blue:`Optional`)
==============================================================


.. note::
    - The elevation map is used to generate the elevation data for the simulation, which can perform real-world scenario generation. If no elevation map is provided, the simulation will consider the network as flat.
    - You need to add the elevation map to the configuration file. For details on how to add the elevation map to the configuration file, please refer to the section: :ref:`Update Configuration File`.

.. code-block:: python
    :linenos:
    :emphasize-lines: 6, 7, 8

    import realtwin as rt

    # prepare the elevation map
    # This will create a sample elevation map in the current working directory: `elevation_map.tif`

    bbox = "the bounding box of the area you want to download the elevation data for, in the format: (lon_min, lat_min, lon_max, lat_max)"
    output_file = "elevation_map.tif"  # the name/path of the output file
    rt.download_elevation_tif_by_bbox(bbox, output_file)

How To Prepare / Update Configuration File (:blue:`Required`)
=============================================================

The configuration file is a `YAML`_ file that contains all the necessary parameters for the realtwin package to run.

Prepare From `realtwin` Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

User can use the sample configuration file provided in the `realtwin` package.

.. code-block:: python
    :linenos:
    :emphasize-lines: 5

    import realtwin as rt

    # prepare the configuration file
    # This will create a sample configuration file in the current working directory: `realtwin_config.yaml`
    rt.prepare_config_file()

User can also specify the output directory for the configuration file.

.. code-block:: python
    :linenos:
    :emphasize-lines: 4, 5

    import realtwin as rt

    # This will create a sample configuration file in the specified directory: `realtwin_config.yaml`
    output_dir = "Directory/where/you/want/to/save/the/config/file"
    rt.prepare_config_file(dest_dir=output_dir)

If the configuration file generated, user can modify the file according to their needs. For details on how to modify the configuration file, please refer to the section: :ref:`Update Configuration File`.

Prepare From GitHub
~~~~~~~~~~~~~~~~~~~~
Download the sample configuration file from the `Real-Twin GitHub Config`_ and modify it according to your needs. For details on how to modify the configuration file, please refer to the section: :ref:`Update Configuration File`.


Update Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~

There are several parameters in the configuration file that need to be updated according to your needs. The following is a list of parameters that need to be updated.

- :orange:`demo_data`:

  whether to use demo data or not. Default is false. It could be boolean or string type.

  - `false`: use data from the configuration file
  - `true`: use default demo data from the package (chattanooga)
  - `chattanooga`: use chattanooga data from the package, which is the default value, optional data: chattanooga, nashville, knoxville

- :orange:`input_dir`:

  the directory where the input files are located. This is a **required parameter**.

- :orange:`output_dir`:

  the directory where the output files are located. This is an optional parameter. If not specified, the output files will be saved in the input directory.

- :orange:`Network`:

  - `NetworkName`:

    the name of the network. This is a **required parameter**. The name should not have space between words.

  - `NetworkVertices`:

    the vertices of the network in the format: (lon_min, lat_min), (lon_max, lat_max). This is a **required parameter**.

  - `Net_BBox`:

    the bounding box of the network. This is an optional parameter. If not specified, the bounding box will be generated automatically based on the network vertices.

  - `ElevationMap`:

    the elevation map in TIFF format. This is an optional parameter. User can prepare the elevation map using :ref:`How To Prepare Elevation Map In TIFF Format` to generate the elevation map, and save the elevation map in the input_dir directory. If not specified, the simulation will consider the network as flat.

- :orange:`Traffic`:

  - `Volume`:

    the traffic volume data file. This is a **required parameter**. The file should be in CSV format.

  - `TurningRatio`:

    the turning ratio data file. This is a **required parameter**. The file should be in CSV format.

  - `GridSmart_lookup`:

    the GridSmart lookup table. This is a **required parameter**. The file should be in CSV format.

- :orange:`Control`:

  - `Signal`:

    the signal data file. This is a **required parameter**. The file should be in CSV format.

  - `Synchro_lookup`:

    the Synchro lookup table. This is a **required parameter**. The file should be in CSV format.

- :orange:`Calibration`:

  - **scenario_config**:

    the overall configuration for the calibration.

    - `sim_start_time`:

      the start time of the simulation in seconds. This is a **required parameter**. Default is 28800 (8:00 AM).

    - `sim_end_time`:

      the end time of the simulation in seconds. This is a **required parameter**. Default is 32400 (9:00 AM).

    - `calibration_target`:

      the target for the calibration. This is a **required parameter**. The target can be GEH or GEHPercent.

      - `GEH`:

        the GEH statistic. This is a **required parameter**. Default is 5.

      - `GEHPercent`:

        the accepted match. This is a **required parameter**. Default is 0.85.

    - `calibration_interval`:

      the calibration interval in minutes. This is a **required parameter**. Default is 60.
    - `demand_interval`:

      the demand interval in minutes. This is a **required parameter**. Default is 15.

    - `path_turn`:

      the path to the turning data file. This is a **required parameter**.

    - `path_inflow`:

      the path to the inflow data file. This is a **required parameter**.

    - `path_summary`:

      the path to the summary data file. This is a **required parameter**.

    - `path_updated_signal`:

      the path to the updated signal data file. This is a **required parameter**.

    - `path_Edge_add`:

      the path to the Edge add data file. This is a **required parameter**.

    - `path_EdgeData`:

      the path to the Edge data file. This is a **required parameter**.

  - **turn_inflow**:

    - `initial_params`:

      the initial parameters for the turning inflow calibration. This is a **required parameter**.

    - `params_ranges`:

      the parameter ranges for the turning inflow calibration. This is a **required parameter**.

    - `max_epoch`:

      the maximum number of iterations. This is a **required parameter**. Default is 1000.

    - `max_fe`:

      the maximum number of function evaluations. This is a **required parameter**. Default is 10000.

    - `max_time`:

      the maximum time in seconds. This is a **required parameter**. Default is 3600.

    - `max_early_stop`:

      the maximum number of early stop iterations. This is a **required parameter**. Default is 20.

  - **behavior**:

    - `EB_tt`:

     the EB travel time in minutes. This is a **required parameter**. Default is 240.

    - `WB_tt`:

      the WB travel time in minutes. This is a **required parameter**. Default is 180.

    - `EB_edge_list`:

      the EB edge list. This is a **required parameter**.

    - `WB_edge_list`:

      the WB edge list. This is a **required parameter**.

    - `initial_params`:

      the initial parameters for the behavior calibration. This is a **required parameter**.

      - `min_gap`: the minimum gap between vehicles in meters. This is a **required parameter**. Default is 2.5.

      - `acceleration`: the maximum acceleration in m/s^2. This is a **required parameter**. Default is 2.6.

      - `deceleration`: the maximum deceleration in m/s^2. This is a **required parameter**. Default is 4.5.

      - `sigma`: the driver imperfection. This is a **required parameter**. Default is 0.5.

      - `tau`: the desired headway. This is a **required parameter**. Default is 1.00.

      - `emergencyDecel`: the emergency deceleration in m/s^2. This is a **required parameter**. Default is 9.0

    - `params_ranges`: the parameter ranges for the behavior calibration. This is a **required parameter**.

      - `min_gap`: the minimum gap between vehicles in meters. This is a **required parameter**. Default is [1.0, 3.0].

      - `acceleration`: the maximum acceleration in m/s^2. This is a **required parameter**. Default is [2.5, 3.0].

      - `deceleration`: the maximum deceleration in m/s^2. This is a **required parameter**. Default is [4.0, 5.3].

      - `sigma`: the driver imperfection. This is a **required parameter**. Default is [0.0, 1.0].

      - `tau`: the desired headway. This is a **required parameter**. Default is [0.25, 1.25].

      - `emergencyDecel`: the emergency deceleration in m/s^2. This is a **required parameter**. Default is [5.0, 9.3].

    - `max_epoch`:

      the maximum number of iterations. This is a **required parameter**. Default is 1000.

    - `max_fe`:

      the maximum number of function evaluations. This is a **required parameter**. Default is 10000.

    - `max_time`:

      the maximum time in seconds. This is a **required parameter**. Default is 3600.

    - `max_early_stop`:

      the maximum number of early stop iterations. This is a **required parameter**. Default is 20.

  - **ga_config**: the configuration for the genetic algorithm.

    - `num_generations`: the number of generations. This is a **required parameter**. Default is 10.
    - `num_variables`: the number of variables. This is a **required parameter**. Default is 16.
    - `num_turning_ratio`: the number of turning ratios. This is a **required parameter**. Default is 12.
    - `ubc`: the upper bound of inflow. This is a **required parameter**. Default is 200.
    - `population_size`: the population size. This is a **required parameter**. Default is 2.
    - `crossover_rate`: the crossover rate. This is a **required parameter**. Default is 0.75.
    - `mutation_rate`: the mutation rate. This is a **required parameter**. Default is 0.1.
    - `elitism_size`: the number of elite individuals to carry over. This is a **required parameter**. Default is 1.
    - `best_fitness_value`: the best fitness value. This is a **required parameter**. Default is 999999.
    - `max_no_improvement`: the maximum number of early stop iterations. This is a **required parameter**. Default is 5.

    - `epoch`: the number of generations. This is a **required parameter**. Default is 1000.
    - `pop_size`: the population size. This is a **required parameter**. Default is 30.
    - `pc`: the crossover probability. This is a **required parameter**. Default is 0.75.
    - `pm`: the mutation probability. This is a **required parameter**. Default is 0.1.
    - `selection`: the selection method. This is a **required parameter**. Default is roulette.
    - `key_way`: the key way for tournament selection. This is a **required parameter**. Default is 0.2.
    - `crossover`: the crossover method. This is a **required parameter**. Default is uniform.
    - `mutation`: the mutation method. This is a **required parameter**. Default is swap.
    - `elite_best`: the percentage of the best in elite group. This is a **required parameter**. Default is 0.1.
    - `elite_worst`: the percentage of the worst in elite group. This is a **required parameter**. Default is 0.3.
    - `model_selection`: the model selection method. This is a **required parameter**. Default is BaseGA. Optional values: BaseGA, EliteSingleGA, EliteMultiGA, MultiGA, SingleGA.

  - **sa_config**: the configuration for the simulated annealing.

    - `num_variables`: the number of variables. This is a **required parameter**. Default is 16.
    - `num_turning_ratio`: the number of turning ratios. This is a **required parameter**. Default is 12.
    - `initial_temperature`: the initial temperature. This is a **required parameter**. Default is 100.
    - `ubc`: the upper bound of inflow. This is a **required parameter**. Default is 200.
    - `cost_difference`: the cost difference. This is a **required parameter**. Default is 2.
    - `accept_prob`: the acceptance probability. This is a **required parameter**. Default is 0.5.
    - `stopping_temperature`: the stopping temperature. This is a **required parameter**. Default is 0.001.

    - `epoch`: the number of generations. This is a **required parameter**. Default is 1000.
    - `temp_init`: the initial temperature. This is a **required parameter**. Default is 100.
    - `cooling_rate`: the cooling rate. This is a **required parameter**. Default is 0.891.
    - `scale`: the scale for the temperature. This is a **required parameter**. Default is 0.1.
    - `model_selection`: the model selection method. This is a **required parameter**. Default is OriginalSA. Optional values: OriginalSA, GaussianSA, SwarmSA.

  - **ts_config**: the configuration for the tabu search.

    - `iterations`: the number of iterations. This is a **required parameter**. Default is 3.
    - `tabu_size`: the size of the tabu list. This is a **required parameter**. Default is 120.
    - `neighborhood_size`: the size of the neighbourhood for generating candidate solutions. This is a **required parameter**. Default is 32.
    - `move_range`: the initial move range. This is a **required parameter**. Default is 0.5.
    - `num_turning_ratio`: the number of turning ratios. This is a **required parameter**. Default is 12.
    - `max_no_improvement_local`: the maximum number of early stop iterations for local search. This is a **required parameter**. Default is 5.
    - `max_no_improvement_global`: the maximum number of early stop iterations for global search. This is a **required parameter**. Default is 30.
    - `lower_bound`: the lower bound for inflow counts. This is a **required parameter**. Default is 0.
    - `upper_bound`: the upper bound for inflow counts. This is a **required parameter**. Default is 1.
    - `lbc`: the lower bound for inflow counts. This is a **required parameter**. Default is 0.
    - `ubc`: the upper bound for inflow counts. This is a **required parameter**. Default is 200.

    - `epoch`: the number of generations. This is a **required parameter**. Default is 1000.
    - `tabu_size`: the size of the tabu list. This is a **required parameter**. Default is 10.
    - `neighbour_size`: the size of the neighbourhood for generating candidate solutions. This is a **required parameter**. Default is 10.
    - `perturbation_scale`: the scale of perturbation for the solution. This is a **required parameter**. Default is 0.05.


How To Create / Update MatchupTable (:blue:`Required`)
======================================================


How To Prepare Control (Signal) File (:blue:`Required`)
=======================================================

.. note::
    - The control file is used to define the signal control data for the simulation. The control file **must** be in CSV format using `Synchro`_ `UTDF`_ format.

    - In your input_dir directory, you need to create a folder named "Control" and put the control file in the folder. The control file should be named "Synchro_signal.csv".

The UTDF control file includes the following information:
    - Network
    - Nodes
    - Links
    - Lanes
    - Timeplans
    - Phases

Sample Control File: https://github.com/ORNL-Real-Sim/Real-Twin-Dev/blob/main/datasets/tss/Control/Synchro_signal.csv

How To Prepare Traffic Demand File (:blue:`Required`)
=====================================================

Traffic demand data is used to define the traffic demand (turning counts / ratios) for the simulation. Demand data may come from various sources, the realtwin package accept the following formats:
    - GridSmart format:
    - Synchro format:
    - more...

From GradSmart Demands
~~~~~~~~~~~~~~~~~~~~~~



From Synchro UTDF Demands
~~~~~~~~~~~~~~~~~~~~~~~~~




.. _`Real-Twin GitHub Config`: https://github.com/ORNL-Real-Sim/Real-Twin-Dev/blob/main/public_configs.yaml
.. _`venv`: https://docs.python.org/3/library/venv.html
.. _`conda install`: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _`conda manage environments`: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _`virtualenv`: https://virtualenv.pypa.io/en/latest/user_guide.html
.. _`YAML`: https://en.wikipedia.org/wiki/YAML
.. _`Synchro`: https://online.trafficware.com/downloads/
.. _`UTDF`: https://docs.aimsun.com/next/22.0.1/UsersManual/SynchroImporter.html#:~:text=Synchro%20is%20software%20from%20Trafficware,Combined%20format%20defined%20by%20Trafficware