Supported Simulators
====================

The current Real-Twin package supports Windows only for all supported
simulators. Choose the scenario class for the simulator you intend to run. Each class
checks its own simulator environment and uses native input and output files.

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Simulator
     - Scenario class
     - Requirements and workflow
   * - SUMO
     - :class:`realtwin.RealTwinSUMO`
     - Windows and SUMO tools on ``PATH``. Import a ``.net.xml`` network or generate one
       from the configured polygon, then prepare demand and optional signals.
   * - PTV Vissim
     - :class:`realtwin.RealTwinVISSIM`
     - Windows, ``pywin32``, a Vissim COM license, and SUMO ``netconvert``.
       Import the selected SUMO network through OpenDRIVE and use native
       Vissim link/movement tables.
   * - Aimsun Next
     - :class:`realtwin.RealTwinAimsun`
     - Windows, a licensed Aimsun installation, an ``.ang`` model, a configured
       ``aconsole.exe`` path, and compatible Python packages.

For example, after constructing a SUMO instance:

.. code-block:: python

    twin.env_setup(sel_sim=["SUMO"])

Use ``["VISSIM"]`` with ``RealTwinVISSIM`` and ``["AIMSUN"]`` with
``RealTwinAimsun``. Calling ``env_setup()`` without an argument defaults to
that class's simulator. A SUMO instance does not prepare a native Vissim or
Aimsun scenario.

SUMO ``env_setup()`` can check/select a version and search additional executable
directories. Commercial simulator installation and licensing must be completed
before running their tutorials. Aimsun's version check and scenario stages use
``AIMSUN.aimsun_exe_path``, even when ``sel_dir`` is used for discovery.

All three scenario workflows support optional external signal control and
independent turn/inflow and behavior calibration with genetic algorithm (GA),
simulated annealing (SA), tabu search (TS), or Bayesian optimization (BO).
Disabling external control preserves existing source/model control objects;
it does not make a signalized network unsignalized. See
:doc:`realtwin_prepare` for the input locations and
:doc:`realtwin_generation` for executable examples.

The :class:`realtwin.SimAV` workflow is SUMO-specific and is described in
:doc:`autonomous_vehicles_simulation`.

Simulator information:

* `SUMO <https://sumo.dlr.de/docs/index.html>`_
* `Aimsun Next <https://www.aimsun.com/>`_
* `PTV Vissim <https://www.ptvgroup.com/en/solutions/ptv-vissim/>`_
