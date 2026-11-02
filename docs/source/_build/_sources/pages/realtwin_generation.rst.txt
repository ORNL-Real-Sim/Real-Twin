=============================
Scenario Generation
=============================

The current Real-Twin package supports Windows only. Use the class for your
simulator and complete :doc:`realtwin_prepare` first.
The repository scripts share the
:download:`tutorial YAML <../../../tutorials/realtwin_config.yaml>`.
It uses ``datasets/example1`` for observations and
``datasets/example1/output`` for generated files. The SUMO and Vissim scripts
separately import ``datasets/example2/updated.net.xml``.

Calibration supports genetic algorithm (GA), simulated annealing (SA), tabu
search (TS), and Bayesian optimization (BO). The current scripts select BO
for enabled stages.

.. contents:: On this page
   :depth: 2
   :local:

SUMO workflow
-------------

Run from the repository root:

.. code-block:: text

    python tutorials/tutorial_realtwin_SUMO.py

For interactive inspection, use
``python -m jupyter notebook tutorials/tutorial_realtwin_SUMO.ipynb``.
The script's main stages are:

.. code-block:: python

    import realtwin as rt

    twin = rt.RealTwinSUMO(
        input_config_file="./tutorials/realtwin_config.yaml",
        verbose=True,
    )
    twin.env_setup(sel_sim=["SUMO"])
    twin.generate_inputs(incl_sumo_net="./datasets/example2/updated.net.xml")

    # Fill datasets/example1/MatchupTable.xlsx at the preparation prompt.
    twin.generate_abstract_scenario()
    twin.generate_concrete_scenario()
    twin.prepare_simulation()

    sel_behavior_routes = {
        "route_1": {
            "time": 100,
            "route_list": ["-90", "-96", "-103", "-105", "-106", "-99", "-94"],
        },
        "route_2": {
            "time": 150,
            "route_list": ["-97", "-95", "-100", "-107", "-104", "-102", "-93"],
        },
    }
    twin.calibrate(
        sel_algo={"turn_inflow": "BO", "behavior": "BO"},
        sel_behavior_routes=sel_behavior_routes,
    )

Put demand workbooks in ``<input_dir>/Traffic`` and optional Synchro signal
CSVs in ``<input_dir>/Control``. Complete the generated matchup table and check
the observation movements against the network bearings before continuing.
Replace the example routes and travel times (seconds) with your observations.
The tutorial YAML enables behavior and disables turn/inflow calibration, so
only behavior runs despite selecting algorithms for both stages.

SUMO's no-argument ``prepare_simulation()`` prepares 08:00-10:00. To match the
tutorial's 08:00-09:00 calibration window, pass
``start_time=28800, end_time=32400``. Generated SUMO preparation files are
under ``<output_dir>/SUMO/``. The script's ``post_process()`` and
``visualize()`` calls remain ongoing hooks.

Set ``SUMO.is_control_data: false`` in the YAML configuration to skip external
SUMO signal loading. Control CSVs are optional in this mode; traffic data and
``MatchupTable.xlsx`` remain required. The source network's existing signal
logic is preserved. Omitting the flag keeps signal loading enabled.
``generate_inputs()`` sets the flag to false when the Control folder is missing
or empty. If control data is added afterwards, set
``twin.input_config["SUMO"]["is_control_data"] = True`` before generating the
abstract scenario to enable loading again.

SUMO and OpenDrive network identities
-------------------------------------

Input generation retains the source SUMO ``.net.xml`` and exports a separate
OpenDrive ``.xodr``. It does not convert OpenDrive back over the source network.
SUMO simulation preparation copies the source network into its output folder.
New matchup tables keep exact source road IDs, including leading ``-`` and
underscores. Legacy tables that omitted ``-`` use a fallback only when the exact
ID is absent from the network.

For explicit conversions into separate output files::

    from realtwin.rt_sumo.xodr_sumo import xodr_from_net_xml, xodr_to_net_xml

    odr_map = xodr_from_net_xml("source.net.xml", "export.xodr")
    sumo_net = xodr_to_net_xml("export.xodr", "converted.net.xml")

The exporter restores source road and junction IDs using netconvert metadata.
Original SUMO lane IDs and signal programs are saved in OpenDrive ``userData``;
OpenDrive lane IDs remain local integers. The reverse helper restores identities
where roads and lanes correspond one to one, and restores signal programs and
controlled connection indices. Native conversion may change geometry and internal
connections; split roads keep generated segment IDs. Missing saved signal
connections cause a conversion error. OpenDrive files without this metadata keep
native generated SUMO IDs with their OpenDrive origins recorded as parameters.
The helpers require ``netconvert`` on ``PATH`` or an explicit executable path,
and importable ``sumolib``.

Aimsun preparation
------------------

Before running Aimsun, replace the machine-specific ``AIMSUN.model_fname``,
``model_xdor``, ``site_packages``, and ``aimsun_exe_path`` settings in the YAML.
Use a copy of the model because preparation saves changes to it. The Aimsun
runtime's Python packages must match its Python version; ``sel_dir`` alone does
not replace the configured console path.

The checked-in ``tutorials/tutorial_realtwin_Aimsun.py`` currently checks the
environment and calibrates an already prepared model. Its generation and
simulation-preparation calls are commented out. For a new model, enable those
calls or run:

.. code-block:: python

    import realtwin as rt

    twin = rt.RealTwinAimsun(
        input_config_file="./tutorials/realtwin_config.yaml",
        verbose=True,
    )
    twin.env_setup(sel_sim=["AIMSUN"])
    twin.generate_inputs()
    # Complete MatchupTable.xlsx beside the model at the preparation prompt.
    twin.generate_abstract_scenario()
    twin.generate_concrete_scenario()
    twin.prepare_simulation()
    twin.calibrate(
        sel_algo={"turn_inflow": "BO", "behavior": "BO"},
        sel_behavior_routes=[("Subpath1", 2071, 2092, 60.0)],
    )

Replace the section IDs and the 60-second example target with field
observations. Calibration requires a prepared model with a replication and
SQLite output; ``prepare_simulation()`` creates the scenario, experiment, and
replication and configures that output. A behavior-only run uses demand already
in the model.

Set ``AIMSUN.is_control_data: false`` in the YAML configuration to skip
external control loading. Aimsun reads ``Control`` and ``Traffic`` beside
``AIMSUN.model_fname``. Input, abstract and concrete generation also set the
flag to false when that Control folder is missing or has no signal CSV files.
The flag defaults to true when omitted and data is available.

Disabled control skips Synchro reads, signal detector generation, signal import
and RealTwin control-plan creation. Demand generation continues, and simulation
preparation creates a scenario without requiring or assigning the RealTwin
master control plan. Existing model control objects remain unchanged;
calibration uses the same ``.ang`` model. Traffic data and the matchup table
remain required. After adding signal CSVs, set
``twin.input_config["AIMSUN"]["is_control_data"] = True`` and regenerate the
abstract and concrete scenarios.

Calibration algorithms and independent stages
---------------------------------------------

All three scenario classes support GA (default), SA, TS, and BO and accept
``sel_algo={"turn_inflow": "BO", "behavior": "BO"}`` at the calibration step.
Install the optional dependencies with ``python -m pip install -e ".[bo]"``
from the checkout, in the Python environment used by the tutorial. SUMO and
``jtrrouter`` must be on ``PATH`` and ``SUMO_HOME`` must be configured.

The packaged and tutorial YAML files include ``Calibration.bo_config``.
Settings are ``kernel_type`` (default ``RBF``), ``target`` (0), ``tolerance`` (3),
``random_points`` (4000), ``max_evaluations`` (100), ``total_run`` (1), and
``seed`` (812). Supported kernels are RBF, Matern, RationalQuadratic,
ExpSineSquared, and Combined. ExpSineSquared uses a fixed period equal to each
parameter's bound span. The candidate pool must contain at least
``max_evaluations`` points. Per-stage overrides such as
``update_behavior_algo={"bo_config": {"max_evaluations": 30}}`` merge with
the shared settings.

The evaluation budget includes initial samples. BO stops when the best fitness
is at most ``target + tolerance``: mean GEH for turn/inflow, and travel-time
RMSE in seconds for SUMO behavior, or MAE in seconds for Aimsun behavior. Each run uses a distinct reproducible seed.
Each stage applies the best candidate in one additional simulation outside
the search budget, then exports fitness/parameter/best-per-run CSVs and
convergence PNGs to ``turn_inflow_bo_result`` or ``behavior_bo_result`` under
that stage's SUMO directory, or beside the Aimsun model with CSV filenames
prefixed by ``aimsun_bayesopt``. GA remains the default.

Enable ``Calibration.turn_inflow.is_calibration`` and
``Calibration.behavior.is_calibration`` independently. The packaged starter
disables both; the repository tutorial enables behavior only. Disabled stages
can be omitted from ``sel_algo``. Algorithm selection does not enable a stage,
and both flags false make ``calibrate()`` return ``False`` without optimizing.
SUMO behavior-only calibration can
use a prepared scenario without prior turn/inflow calibration. Aimsun uses
demand already in its model. Aimsun behavior routes are tuples of
``(name, start_section_id, end_section_id, observed_seconds)``; they
are required only when behavior is enabled and can be passed to calibrate()
or stored under ``Calibration.behavior.sel_behavior_routes``.

For a prepared Vissim instance with behavior enabled and routes configured in
the tutorial YAML, reduce the BO search budget with:

.. code-block:: python

    twin.calibrate(
        sel_algo={"behavior": "BO"},
        update_behavior_algo={"bo_config": {"max_evaluations": 30}},
    )

SUMO and Aimsun accept the same override. Retain their
``sel_behavior_routes`` argument when supplying observed routes in Python.

Aimsun BO CSV parameters are normalized to [0, 1]. The objectives map
these values to turning percentages, inflows, or physical behavior ranges.
Both enabled stages receive a final application of their best candidate;
an enabled stage without its required inputs reports a failure.

Vissim preparation
------------------

Install ``pywin32`` in the environment running Real-Twin, make SUMO's
``netconvert`` available, and verify your Vissim COM license. Run
``python tutorials/tutorial_realtwin_Vissim.py`` from the repository root, or
follow the stages interactively:

.. code-block:: python

    import realtwin as rt

    twin = rt.RealTwinVISSIM(
        input_config_file="./tutorials/realtwin_config.yaml",
        verbose=True,
    )
    twin.env_setup(sel_sim=["VISSIM"])
    twin.generate_inputs(incl_sumo_net="./datasets/example2/updated.net.xml")
    # Complete <output_dir>/VISSIM/MatchupTable.xlsx at the prompt.
    twin.generate_abstract_scenario()
    twin.generate_concrete_scenario()
    twin.prepare_simulation()
    twin.calibrate(sel_algo={"turn_inflow": "BO", "behavior": "BO"})

``RealTwinVISSIM.generate_inputs(incl_sumo_net=...)`` imports the supplied
SUMO network before exporting its native Vissim movement and link tables.
All preparation stages read ``Traffic`` and ``Control`` under the configured
``input_dir``. Edit the generated ``<output_dir>/VISSIM/MatchupTable.xlsx``;
keep its native link numbers and internal paths. A SUMO workbook cannot replace
this table, even if its column headings are renamed.

For this non-demo workflow, ``incl_sumo_net`` takes precedence over
``VISSIM.sumo_xml_file``. If the argument is omitted, set that configuration
path to a valid SUMO network; its checked-in value is machine-specific. The
import exports native link, movement, and junction CSVs alongside the Vissim
network. Preserve ``FromLinkNo_Vissim``, ``ToLinkNo_Vissim``, and
``InternalLinks_Vissim`` in the workbook and enter ``File_GridSmart`` for each
surveyed junction, plus ``File_Synchro`` and ``IntersectionID_Synchro`` when
control is enabled.

Control data is optional. Set ``VISSIM.is_control_data: false`` in the YAML
configuration to skip Synchro reads and external signal loading. Input,
abstract and concrete generation also set this flag to false when
``<input_dir>/Control`` is missing or contains no signal CSV files. Demand
generation continues; simulation and calibration use
``<NetworkName>_demand.inpx`` instead of ``<NetworkName>_demand_signals.inpx``.
Signals already present in the source network are retained. When control data
is available, omitting the flag keeps loading enabled. After adding signal
CSVs, set ``twin.input_config["VISSIM"]["is_control_data"] = True`` and regenerate
the abstract and concrete scenarios. Standalone stage 2 accepts ``--no-control``
to skip deriving signal columns while retaining existing workbook references.

To reuse observations from a SUMO workbook, run stage 2 with ``--seed-from``;
its ``<network>_junctions.csv`` translates SUMO junction IDs into the imported
OpenDrive IDs. For example::

    python -m realtwin.rt_vissim.02_matchup_table --movements <output_dir>/VISSIM/<network>_movements.csv --outdir <output_dir>/VISSIM --traffic-dir <input_dir>/Traffic --control-dir <input_dir>/Control --seed-from <SUMO_MatchupTable.xlsx>

An existing table with missing link numbers or movement IDs from another
network is rejected before its cells are updated. Back it up before adding
``--regenerate`` to rebuild from the current movement table, then seed or enter
its observations again. ``generate_inputs()`` stops on import or table errors
so subsequent stages cannot silently reuse an invalid network/table pair.

Concrete generation creates ``<NetworkName>_demand.inpx`` and, when external
control is enabled, ``<NetworkName>_demand_signals.inpx``. ``prepare_simulation()``
runs the appropriate network for 900 seconds. This short run is separate from
the configured calibration window. Standalone stages use
``python -m realtwin.rt_vissim.<stage>`` so package imports resolve correctly.

Vissim calibration
------------------

``RealTwinVISSIM.calibrate()`` supports GA (default), SA, TS and BO for unmeasured inflows
and turn splits at junctions marked ``Need calibration? = Y`` in the Vissim
matchup table. Prepare the demand/signals network and observed GridSmart data
first. Enable ``Calibration.turn_inflow.is_calibration`` for this stage.
``Calibration.behavior.is_calibration`` independently enables native driving
behavior calibration against observed route travel times in seconds. When both
are enabled, behavior uses the demand stage's saved network; when the demand
stage has no variables, it explicitly passes its input network forward.

``realtwin.rt_vissim.calibrate_vissim.cali_vissim`` imports separate adapters
from ``cali_turn_inflow`` and ``cali_behavior`` and dispatches their
run_GA/run_SA/run_TS/run_BO methods through the shared
``run_calibration_algorithm`` function, following the Aimsun package pattern.
BO reuses the existing BayesianOptimization and shared bo_config on normalized
parameter bounds. Select ``sel_algo={"turn_inflow": "BO"}`` and update bo_config
through ``update_turn_inflow_algo``. Its results use the ``vissim_bayesopt``
CSV prefix, with convergence plots in ``<calibrated_stem>_bo_result``.

Per-call ``update_turn_inflow_algo`` and ``update_behavior_algo`` merge algorithm,
``scenario_config`` and stage-specific blocks into independent effective configurations.
The subprocess receives these live settings, the selected algorithm, the
output network/tables and the input Traffic directory. The shared Calibration
mapping remains reusable. Simulation start/end are seconds after midnight;
``calibration_time_step`` is seconds per step and must correspond to 1-20
Vissim steps per second. GEH and GEHPercent control the reported target.
Observed count intervals must cover the whole demand window without gaps;
partial observations cannot be compared with a longer full simulation.

Optional turn_inflow settings include ``objective`` (approach or movement),
``budget``, ``seed`` (Vissim), ``optimizer_seed`` and ``out`` (a separate
calibrated model). Every candidate resets the simulation with the fixed seed.
The best model, history/approach CSVs, summary JSON and charts are written
beside the prepared model by default; the source is preserved. A failed
subprocess raises an exception. A successful or no-variable run returns True.
GA/SA/TS honor stage ``max_epoch``, ``max_fe``, ``max_time`` (seconds) and
``max_early_stop``, with scenario_config as fallback and the algorithm's epoch
as a cap. BO retains the shared bo_config evaluation and target stop settings.

Behavior requires explicit native ``Calibration.behavior.params_ranges`` and
``driving_behavior_ids``. Supported continuous attributes are ``W74ax``,
``W74bxAdd``, ``W74bxMult``, ``W99cc0`` and ``W99cc2`` through ``W99cc9``.
Selected driving behaviors must use the corresponding Wiedemann model;
``W99cc1Distr`` is a distribution reference and is not optimized as a number.
Replace the shared SUMO parameter names when configuring Vissim behavior.
For shared configurations, put native settings under
``Calibration.behavior.vissim``; these replace the corresponding shared
behavior values for Vissim, including the entire parameter-range mapping.
Per-call ``update_behavior_algo`` settings take precedence over that block.
The shared configuration remains unchanged. Enabled behavior inputs are
validated before either stage starts; missing settings raise a descriptive
``ValueError`` in the caller instead of a subprocess traceback.
Stage ``calibration_time_step`` values override the shared scenario step. The
tutorial uses 0.1 seconds to match its signal controller's 10 updates per second.
For example, use ``params_ranges={"W74ax": [1.0, 3.0]}`` and
``driving_behavior_ids=[1]`` for a model using Wiedemann 74 behavior 1.

Pass ``sel_behavior_routes=[["corridor", start_link, end_link, observed_seconds]]``
to calibrate, or set ``Calibration.behavior.sel_behavior_routes`` or
``Calibration.behavior.vissim.sel_behavior_routes``. The method
argument takes precedence. Measurements span position zero on the start link
to the end of the end link and aggregate completed vehicles across all classes.
Override these markers with ``behavior.route_positions``, for example
``{"corridor": {"start": 0.0, "end": 0.5}}`` in metres along the respective links.
Match the field measurement locations and choose markers vehicles cross;
connectors can divert vehicles before they reach the full link length. Actual
marker positions are recorded in the summary.
An optional ``reference_source`` mapping is also recorded, to distinguish
simulated tutorial references from field observations.
The Vissim tutorial uses westbound links 9 to 15, with markers at 0.0 and 0.5
metres. Its 39.198670-second reference is the simulated mean of 162 completed
vehicles during 08:00-09:00 with seed 42. The checksummed reference record is
saved in the generated VISSIM ``behavior_reference`` folder. This demonstrates
calibration execution; replace the target and ``reference_source`` metadata
with field observations for real calibration.
The objective is route travel-time MAE in seconds. Missing measurements receive
``missing_route_penalty + observed_seconds`` (default penalty: 10,000 seconds).
A best candidate with missing routes raises an error before saving the model.
Behavior writes a separate ``<source_stem>_behavior_<algo>.inpx``, history/routes
CSVs and summary JSON, plus BO exports when selected. Its ``seed``,
``optimizer_seed``, ``budget`` and ``out`` settings follow the demand stage's
format and can be overridden independently.

Use ``python -m realtwin.rt_vissim.06_calibrate_turn_inflow --help`` for the
standalone stage. Explicit network, matchup, link, movement and Traffic paths
replace project-specific defaults. ``--config`` reads YAML or JSON settings;
``--dry-run`` validates settings and lists variables without starting Vissim.
Use ``--stage behavior --inpx <model> --config <config>`` for behavior calibration;
it does not require Traffic or matchup tables.
Real calibration requires Windows, pywin32 and a Vissim COM licence.
