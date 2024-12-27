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
    twin = rt.REALTWIN(input_dir=INPUT_DIR)

    # environment setup
    # Check if SUMO, VISSIM, AIMSUN, etc... are installed
    twin.env_setup(sel_sim=["SUMO", "VISSIM"], create_venv=False)

    # twin.venv_delete(venv_name=twin._venv_name,
    #                  venv_dir=twin._output_dir)

    # load the dataset
    # print out the general informations,
    # such as # of nodes, # of edges, # signalized intersections, etc.
    twin.load_inputs()

    # generate scenarios
    twin.generate_concrete_scenario()  # keywards arguments can be passed to specify the scenario generation options

    # simulate the scenario
    twin.simulate()  # keywards arguments can be passed to specify the simulation options

    # post-process the simulation results
    twin.post_process()  # keywards arguments can be passed to specify the post-processing options

    # visualize the simulation results
    twin.visualize()  # keywards arguments can be passed to specify the visualization options