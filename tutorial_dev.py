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

    INPUT_DIR = "./datasets/scenario_dummy/"

    # initialize the realtwin object
    twin = rt.REALTWIN(input_config_file="")

    # environment setup
    # Check if SUMO, VISSIM, AIMSUN, etc... are installed

    # redundant check
    twin.env_setup(sel_sim=["SUMO", "VISSIM"], create_venv=False)

    # NOTE redundant check including selection of directories
    new_dir = [r"C:\Users\xh8\ornl_workspace\github_workspace\Real-Twin\SUMO\sumo-1.20.0\bin"]
    # change the new_dir to your own directory where the SUMO is installed (multiple versions)
    twin.env_setup(sel_sim=["SUMO", "VISSIM"], create_venv=False, sel_dir=new_dir)

    # NOTE Strict version check
    twin.env_setup(sel_sim=["SUMO", "VISSIM"], create_venv=False, strict_sumo_version=True, sel_dir=new_dir)

    # twin.venv_delete(venv_name=twin._venv_name,
    #                  venv_dir=twin._output_dir)

    # load the dataset
    # print out the general information,
    # such as # of nodes, # of edges, # signalized intersections, etc.
    twin.load_inputs()

    # generate scenarios
    twin.generate_concrete_scenario()  # keywords arguments can be passed to specify the scenario generation options

    # simulate the scenario
    twin.simulate()  # keywords arguments can be passed to specify the simulation options

    # perform calibration
    # keyword arguments can be passed to specify the calibration options
    # or change from internal and external configuration files
    twin.calibrate()

    # post-process the simulation results
    twin.post_process()  # keywords arguments can be passed to specify the post-processing options

    # visualize the simulation results
    twin.visualize()  # keywords arguments can be passed to specify the visualization options
