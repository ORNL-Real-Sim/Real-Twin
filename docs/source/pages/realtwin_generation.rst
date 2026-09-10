=============================
Scenario Generation
=============================

.. code-block:: python
    :linenos:
    :emphasize-lines: 9

    import realtwin as rt

    # Please refer to the official documentation for more details on RealTwin preparation before running the simulation
    # # https://real-twin.readthedocs.io/en/latest/index.html

    if __name__ == '__main__':

        # Step 1: Prepare your configuration file (in YAML format)
        CONFIG_FILE = "./realtwin_config.yaml"

        # Step 2: initialize the realtwin object
        twin = rt.RealTwinSUMO(input_config_file=CONFIG_FILE, verbose=True)

        # Step 3: check simulator env: if SUMO, VISSIM, Aimsun, etc... are installed
        twin.env_setup(sel_sim=["SUMO", "VISSIM"])

        # Step 4: Create Matchup Table from SUMO network
        updated_sumo_net = r"./datasets/example2/chatt.net.xml"
        twin.generate_inputs(incl_sumo_net=updated_sumo_net)

        # BEFORE step 5, there are three steps to be performed:
        # 1. Prepare Traffic Demand and save it to Traffic Folder in input directory
        # 2. Prepare Control Data (Signal) and save it to Control Folder in input directory
        # 3. Manually fill in the Matchup Table in the input directory

        # Step 5: generate abstract scenario
        twin.generate_abstract_scenario()

        # AFTER step 5, Double-check the Matchup Table in the input directory to ensure it is correct.

        # Step 6: generate scenarios
        twin.generate_concrete_scenario()

        # Step 7: simulate the scenario
        twin.prepare_simulation()

        # Step 8: perform calibration, Available algorithms: GA: Genetic Algorithm, SA: Simulated Annealing, TS: Tabu Search, BO: Bayesian Optimization
        twin.calibrate(sel_algo={"turn_inflow": "GA", "behavior": "GA"})

        # Step 9 (ongoing): post-process the simulation results
        twin.post_process()  # keyword arguments can be passed to specify the post-processing options

        # Step 10 (ongoing): visualize the simulation results
        twin.visualize()  # keyword arguments can be passed to specify the visualization options

Bayesian optimization for SUMO and Aimsun
-----------------------------------------

``RealTwinSUMO`` and ``RealTwinAimsun`` accept
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
``Calibration.behavior.is_calibration`` independently. Disabled stages
can be omitted from ``sel_algo``. SUMO behavior-only calibration can
use a prepared scenario without prior turn/inflow calibration. Aimsun uses
demand already in its model. Aimsun behavior routes are tuples of
``(name, start_section_id, end_section_id, observed_seconds)``; they
are required only when behavior is enabled and can be passed to calibrate()
or stored under ``Calibration.behavior.sel_behavior_routes``.

Aimsun BO CSV parameters are normalized to [0, 1]. The objectives map
these values to turning percentages, inflows, or physical behavior ranges.
Both enabled stages receive a final application of their best candidate;
an enabled stage without its required inputs reports a failure.
