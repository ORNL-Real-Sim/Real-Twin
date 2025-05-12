##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a __TBD__           #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors (Add you name below to acknowledge your contribution):        #
# Xiangyong Roy Luo                                                          #
##############################################################################


import realtwin as rt

# Please refer to the official documentation for more details on RealTwin preparation before running the simulation
#

if __name__ == '__main__':

    # Step 1: Prepare your configuration file (in YAML format)
    CONFIG_FILE = "./realtwin_config.yaml"

    # Step 2: initialize the realtwin object
    twin = rt.RealTwin(input_config_file=CONFIG_FILE, verbose=True)

    # Step 3: check simulator env: if SUMO, VISSIM, Aimsun, etc... are installed
    twin.env_setup(sel_sim=["SUMO", "VISSIM"])

    # Step 4: Create Matchup Table and Update the Table
    twin.generate_inputs()

    # Step 5: generate abstract scenario
    twin.generate_abstract_scenario()

    # Step 6: generate scenarios
    twin.generate_concrete_scenario()

    # Step 7: simulate the scenario
    twin.prepare_simulation()

    # Step 8: perform calibration, Available algorithms: GA: Genetic Algorithm, SA: Simulated Annealing, TS: Tabu Search
    twin.calibrate(sel_algo={"turn_inflow": "GA", "behavior": "GA"})

    # Step 9 (ongoing): post-process the simulation results
    twin.post_process()  # keywords arguments can be passed to specify the post-processing options

    # Step 10 (ongoing): visualize the simulation results
    twin.visualize()  # keywords arguments can be passed to specify the visualization options
