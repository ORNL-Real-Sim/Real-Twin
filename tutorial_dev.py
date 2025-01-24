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

'''
##############################################################
# Created Date: Wednesday, December 18th 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


import realtwin as rt


if __name__ == '__main__':

    CONFIG_FILE = "./public_configs.yaml"

    # initialize the realtwin object
    twin = rt.REALTWIN(input_config_file=CONFIG_FILE, verbose=True)

    # NOTE optional: crate or delete a Python virtual environment for the simulation
    # twin.venv_create(venv_name=twin._venv_name, venv_dir=twin.input_config["output_dir"])
    # twin.venv_delete(venv_name=twin._venv_name, venv_dir=twin.input_config["output_dir"])

    # check simulator env: if SUMO, VISSIM, Aimsun, etc... are installed
    twin.env_setup(sel_sim=["SUMO", "VISSIM"])

    # NOTE optional: check simulator including additional selection of directories
    # change the new_dir to your own directory where the SUMO is installed (multiple versions)
    # new_dir = [r"C:\Users\xh8\ornl_workspace\github_workspace\Real-Twin\SUMO\sumo-1.20.0\bin"]
    # twin.env_setup(sel_sim=["SUMO", "VISSIM"], sel_dir=new_dir)

    # NOTE optional: strict simulator check, if the version is not matched, install the required version
    # twin.env_setup(sel_sim=["SUMO", "VISSIM"], sel_dir=new_dir, sumo_version="1.21.0", strict_sumo_version=True)

    # generate abstract scenario
    twin.generate_abstract_scenario()

    # generate scenarios
    twin.generate_concrete_scenario()

    # simulate the scenario
    twin.prepare_simulation()

    # perform calibration
    # keyword arguments can be passed to specify the calibration options
    # or change from internal and external configuration files
    twin.calibrate()

    # post-process the simulation results
    twin.post_process()  # keywords arguments can be passed to specify the post-processing options

    # visualize the simulation results
    twin.visualize()  # keywords arguments can be passed to specify the visualization options
