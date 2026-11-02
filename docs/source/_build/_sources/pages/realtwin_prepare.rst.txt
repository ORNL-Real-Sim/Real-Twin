Preparation
===========

Real-Twin currently supports Windows only. Prepare the YAML configuration,
traffic observations, and the simulator's
matchup table before generating scenarios. External signal data is optional.
See :doc:`install` for Python dependencies and
:doc:`realtwin_simulator` for simulator requirements.

.. contents:: On this page
   :depth: 2
   :local:
   :backlinks: none

Choose a configuration
----------------------

For the repository examples, start with the
:download:`tutorial configuration <../../../tutorials/realtwin_config.yaml>`.
It contains the current shared optimizer settings and native Vissim behavior
configuration. Run relative paths from the repository root.

You can also copy the packaged configuration into an existing directory:

.. code-block:: python

    import realtwin as rt

    rt.prepare_realtwin_configs(dest_dir=".")

This creates ``realtwin_config.yaml``; it does not prepare traffic observations
or simulator models. The packaged starter and repository tutorial configuration
are different examples. Review their paths, stage switches, and simulator
settings before use. The packaged starter disables both calibration stages;
the current tutorial enables behavior only.

.. _Update Configuration File:

Configuration settings
----------------------

Input directories and network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``demo_data``: use ``false`` or an empty value for your own inputs.
  The currently supported packaged dataset is the string ``"chattanooga"``.
  A boolean ``true`` does not select this dataset.
* ``input_dir``: an existing directory for SUMO/Vissim observations. If omitted
  or nonexistent, the loader falls back to the current working directory.
* ``output_dir``: an existing output directory. If omitted or nonexistent,
  the loader uses ``<input_dir>/output``. Create a custom directory before
  setting this field.
* ``Network.NetworkName``: the name used for generated files, without spaces.
* ``Network.NetworkVertices``: polygon points in ``[longitude, latitude]``
  order when generating a SUMO network from the selected area.
  ``Net_BBox`` can be derived from those vertices.
* ``Network.ElevationMap``: an optional elevation raster; see
  :ref:`How To Prepare Elevation Map In TIFF Format`.

The tutorial currently uses ``input_dir: datasets/example1`` and an unset
``output_dir``, giving ``datasets/example1/output``. Its SUMO and Vissim
scripts separately import ``datasets/example2/updated.net.xml``. Keep network
paths and observation paths distinct when changing datasets.

Traffic and matchup tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tutorial uses ``Traffic: GridSmart`` and
``Control: Synchro_signal.csv``. Actual observations are linked to network
movements through ``MatchupTable.xlsx``.

.. list-table:: Input locations
   :header-rows: 1
   :widths: 15 35 50

   * - Simulator
     - Demand and control folders
     - Matchup table
   * - SUMO
     - ``<input_dir>/Traffic`` and ``<input_dir>/Control``
     - ``<input_dir>/MatchupTable.xlsx``
   * - Vissim
     - ``<input_dir>/Traffic`` and ``<input_dir>/Control``
     - ``<output_dir>/VISSIM/MatchupTable.xlsx``
   * - Aimsun
     - ``Traffic`` and ``Control`` beside ``AIMSUN.model_fname``
     - ``MatchupTable.xlsx`` beside the ``.ang`` model

Place GridSmart turning-count workbooks in the Traffic folder. Enter their
filenames for the surveyed junctions in the generated table. Preserve the
network movement IDs and check that observation directions match the generated
bearings. Save and close the workbook before continuing past a manual prompt.

Vissim's table contains native ``FromLinkNo_Vissim``, ``ToLinkNo_Vissim``,
and ``InternalLinks_Vissim`` values. Use the exported
``<NetworkName>_junctions.csv`` to identify junctions. Fill ``File_GridSmart``
and, when control is enabled, ``File_Synchro`` and
``IntersectionID_Synchro``. Abstract generation derives the remaining fields.
A SUMO workbook cannot replace this table; use stage 2's ``--seed-from``
option to translate its observation references. See :doc:`realtwin_generation`.

Optional external signal control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the flag in the relevant simulator section while keeping its other settings:

.. code-block:: yaml

    SUMO:
        is_control_data: false
    VISSIM:
        is_control_data: false
    AIMSUN:
        is_control_data: false

When external control is enabled, prepare a Synchro UTDF signal CSV in the
simulator's Control folder and fill its file/intersection references in the
matchup table. The CSV may contain Network, Nodes, Links, Lanes, Timeplans, and
Phases sections. Prepare it before running input generation if you want
control enabled throughout the script.

Omitting the flag keeps loading enabled when data is available. An explicit
``false`` skips external signal data even when CSVs exist. SUMO input generation
sets the live flag to false for a missing/empty Control folder. Vissim and
Aimsun input, abstract, and concrete generation do so when their Control folder
has no signal CSV files.

Disabled control skips Synchro processing and external signal import while
retaining demand generation and existing source/model control. Vissim uses
``<NetworkName>_demand.inpx`` for simulation and calibration; Aimsun creates
its scenario without assigning the RealTwin master control plan. Traffic data
and the matchup table remain required.

Adding signal CSVs later does not re-enable the live flag. Update the table's
control references, then, for an existing Vissim instance:

.. code-block:: python

    twin.input_config["VISSIM"]["is_control_data"] = True
    twin.generate_abstract_scenario()
    twin.generate_concrete_scenario()
    twin.prepare_simulation()

Use ``"SUMO"`` or ``"AIMSUN"`` for the corresponding class.

Simulator-specific paths
~~~~~~~~~~~~~~~~~~~~~~~~

For Vissim, ``VISSIM.sumo_xml_file`` identifies the SUMO network to import.
In the non-demo workflow, passing ``incl_sumo_net`` to ``generate_inputs()``
takes precedence. Replace the checked-in machine-specific path if using the
configuration directly.

For Aimsun, configure:

* ``model_fname``: the ``.ang`` model path. Preparation saves changes to this
  model, so use a working copy.
* ``model_xdor``: the OpenDRIVE input path used by network import.
* ``site_packages``: packages compatible with Aimsun's Python runtime.
* ``aimsun_exe_path``: the full licensed console executable path, such as
  ``C:\Program Files\Aimsun\Aimsun Next 23\aconsole.exe``.

Calibration switches, times, and algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable stages independently in the existing Calibration section:

.. code-block:: yaml

    Calibration:
        turn_inflow:
            is_calibration: false
        behavior:
            is_calibration: true

Keep the other calibration settings. Reverse these flags for turn/inflow only,
or enable both stages to run them in sequence. Selecting an algorithm through
``sel_algo`` does not enable its stage. Both flags false make ``calibrate()``
return ``False`` without running an optimizer.

``Calibration.scenario_config`` supplies the calibration window in seconds
after midnight. The tutorial uses ``sim_start_time: 28800`` and
``sim_end_time: 32400`` (08:00-09:00), ``GEH: 5``, and
``GEHPercent: 0.85``. ``calibration_time_step`` is seconds per simulation
step. Vissim stage overrides use 0.1 seconds for the tutorial's 10/s controller.

Available algorithms are genetic algorithm (GA), simulated annealing (SA),
tabu search (TS), and Bayesian optimization (BO).
Current optimizer blocks are ``ga_config``, ``sa_config``, ``ts_config``,
and ``bo_config``. GA uses ``model_selection``, ``epoch``, ``pop_size``,
``pc``, and ``pm``; SA uses ``model_selection``, ``epoch``,
``temp_init``, ``cooling_rate``, ``step_size``, and ``scale``;
TS uses ``epoch``, ``tabu_size``, ``neighbour_size``, and
``perturbation_scale``. The tutorial selects BO with:

.. code-block:: yaml

    Calibration:
        bo_config:
            kernel_type: RBF
            target: 0
            tolerance: 3
            random_points: 4000
            max_evaluations: 100
            total_run: 1
            seed: 812

The candidate pool must contain at least ``max_evaluations`` points. Per-stage
overrides can be passed with ``update_turn_inflow_algo`` and
``update_behavior_algo``. See :doc:`realtwin_generation` for objectives,
routes, budgets, and result locations.

SUMO/Aimsun behavior ranges use the shared physical parameters
``min_gap``, ``acceleration``, ``deceleration``, ``sigma``, ``tau``,
and ``emergencyDecel``. Vissim requires native Wiedemann ranges and explicit
driving-behavior IDs. Put them under ``Calibration.behavior.vissim`` when
sharing a YAML across simulators. The tutorial uses ``W74ax`` and
``W74bxAdd`` for behavior ID 1 and a simulated westbound travel-time reference.
Replace that reference with field observations for empirical calibration.

.. _How To Prepare Elevation Map In TIFF Format:

Optional elevation data
-----------------------

``Network.ElevationMap`` points to an elevation raster. Without one, network
generation treats the area as flat. The download helper is exposed through
``realtwin.util_lib``:

.. code-block:: python

    from realtwin.util_lib import download_elevation_tif_by_bbox

    bbox = (-85.1583, 35.0401, -85.1497, 35.0435)
    download_elevation_tif_by_bbox(bbox, "elevation_map.tif")

The bounding box order is ``(lon_min, lat_min, lon_max, lat_max)``.
This example calls the external elevation service; it is optional and is not
part of the local tutorial checks.

Optional environment helpers
----------------------------

The installation page provides standard environment creation and activation
commands. Real-Twin also exposes:

.. code-block:: python

    import realtwin as rt

    rt.venv_create(venv_name="venv_rt", venv_dir="")

Activate that environment before installing dependencies or running tutorials.
