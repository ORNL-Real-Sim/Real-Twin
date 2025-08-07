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
        twin = rt.RealTwin(input_config_file=CONFIG_FILE, verbose=True)

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

        # Step 8: perform calibration, Available algorithms: GA: Genetic Algorithm, SA: Simulated Annealing, TS: Tabu Search
        twin.calibrate(sel_algo={"turn_inflow": "GA", "behavior": "GA"})

        # Step 9 (ongoing): post-process the simulation results
        twin.post_process()  # keyword arguments can be passed to specify the post-processing options

        # Step 10 (ongoing): visualize the simulation results
        twin.visualize()  # keyword arguments can be passed to specify the visualization options
