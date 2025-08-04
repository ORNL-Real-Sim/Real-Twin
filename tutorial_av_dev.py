##############################################################################
# Copyright (c) 2024-, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################


import realtwin as rt

# Please refer to the official documentation for more details on RealTwin preparation before running the simulation
# # https://real-twin.readthedocs.io/en/latest/index.html

if __name__ == '__main__':

    # Step 1: Prepare/generate configuration file (in YAML format)
    rt.prepare_av_configs()
    CONFIG_FILE = "path-to-generated-config-file"

    # Step 2: Update the configuration file
    # Manually update the configuration file from User.

    # Step 3: initialize the SimAV object
    sim = rt.SimAV(input_config_file=CONFIG_FILE, verbose=True)

    # Step 4: Simulation generation
    sim.run_simulation()
