'''
##############################################################
# Created Date: Tuesday, March 4th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
import random
from functools import partial
from datetime import datetime
import shutil

from mealpy import FloatVar, SA, Tuner
from util_cali_behavior import (get_travel_time_from_EdgeData_xml,
                                update_flow_xml_from_solution,
                                run_jtrrouter_to_create_rou_xml,
                                result_analysis_on_EdgeData)
import pandas as pd
import numpy as np
import pyufunc as pf

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    sys.path = list(set(sys.path))  # remove duplicates

else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci

rng = np.random.default_rng(seed=812)


scenario_config = {
    "input_dir": r"C:\Users\xh8\ornl_work\github_workspace\Real-Twin-Dev\mealpy_tes\input_dir_dummy",
    "network_name": "chatt",
    "sim_name": "chatt.sumocfg",
    "sim_start_time": 28800,
    "sim_end_time": 32400,
    "path_turn": "chatt.turn.xml",
    "path_inflow": "chatt.inflow.xml",
    "path_summary": "summary.xlsx",
    "path_edge": 'EdgeData.xml',
    "calibration_target": {'GEH': 5, 'GEHPercent': 0.85},
    "calibration_interval": 60,
    "demand_interval": 15,
    "EB_tt": 240,
    "WB_tt": 180,
    "EB_edge_list": ["-312", "-293", "-297", "-288", "-286",
                     "-302", "-3221", "-322", "-313", "-284",
                     "-328", "-304"],
    "WB_edge_list": ["-2801", "-280", "-307", "-327", "-281",
                     "-315", "-321", "-300", "-2851", "-285",
                     "-290", "-298", "-295"]
}


def run_SUMO_create_EdgeData(sim_name: str, sim_end_time: float) -> bool:
    """run SUMO simulation using traci module

    Args:
        sim_name (str): the name of the simulation, it should be the .sumocfg file
        sim_end_time (float): the end time of the simulation

    Returns:
        bool: True if the simulation is successful
    """
    sim_label = f"sim_{rng.random()}"
    print(f"  :sim_label: {sim_label}")
    traci.start(["sumo", "-c", sim_name], label=sim_label)

    while traci.simulation.getTime() < sim_end_time:
        traci.simulationStep()
    traci.close()
    return True


def fitness_func(solution: list | np.ndarray, scenario_config: dict = None, error_func: str = "rmse", remove_temp_dir: bool = False) -> float:
    """ Evaluate the fitness of a given solution for SUMO calibration."""
    print(f"  :solution: {solution}")

    global temp_dir_list

    # Set up SUMO command with car-following parameters
    if error_func not in ["rmse", "mae"]:
        raise ValueError("error_func must be either 'rmse' or 'mae'")

    if solution[5] >= 9.3:  # emergencyDecel
        solution[5] = 9.3
    if solution[5] < solution[2]:  # emergencyDecel < deceleration
        solution[5] = solution[2] + random.randrange(1, 5)
    # print("after emergencydecel update", solution)

    # get path from scenario_config
    network_name = scenario_config.get("network_name")
    sim_input_dir = Path(scenario_config.get("input_dir"))

    # change the working directory to the input directory for SUMO
    # os.chdir(sim_input_dir)

    path_net = pf.path2linux(sim_input_dir / f"{network_name}.net.xml")
    path_flow = pf.path2linux(sim_input_dir / f"{network_name}.flow.xml")
    path_turn = pf.path2linux(sim_input_dir / f"{network_name}.turn.xml")
    path_rou = pf.path2linux(sim_input_dir / f"{network_name}.rou.xml")
    path_EdgeData = pf.path2linux(sim_input_dir / "EdgeData.xml")
    EB_tt = scenario_config.get("EB_tt")
    WB_tt = scenario_config.get("WB_tt")
    EB_edge_list = scenario_config.get("EB_edge_list")
    WB_edge_list = scenario_config.get("WB_edge_list")

    sim_name = scenario_config.get("sim_name")
    path_sim_name = pf.path2linux(sim_input_dir / f"{scenario_config.get("sim_name")}")

    # create a template fold for the simulation
    temp_dir = sim_input_dir / f"temp_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")}"

    os.makedirs(temp_dir, exist_ok=True)
    temp_dir_list.append(temp_dir)

    shutil.copy(path_sim_name, temp_dir)
    shutil.copy(path_net, temp_dir)
    shutil.copy(path_flow, temp_dir)
    shutil.copy(path_turn, temp_dir)
    shutil.copy(path_rou, temp_dir)
    shutil.copy(path_EdgeData, temp_dir)
    os.chdir(temp_dir)

    # sim_name_new = pf.path2linux(temp_dir / f"{scenario_config.get('sim_name')}")
    path_net_new = pf.path2linux(temp_dir / f"{network_name}.net.xml")
    path_flow_new = pf.path2linux(temp_dir / f"{network_name}.flow.xml")
    path_turn_new = pf.path2linux(temp_dir / f"{network_name}.turn.xml")
    path_rou_new = pf.path2linux(temp_dir / f"{network_name}.rou.xml")
    path_EdgeData_new = pf.path2linux(temp_dir / "EdgeData.xml")

    update_flow_xml_from_solution(path_flow_new, solution)

    run_jtrrouter_to_create_rou_xml(network_name, path_net_new, path_flow_new, path_turn_new, path_rou_new)

    # Define the command to run SUMO
    # sumo_command = f"sumo -c \"{sim_name}\""
    # sumoProcess = subprocess.Popen(sumo_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # sumoProcess.wait()
    run_SUMO_create_EdgeData(sim_name, scenario_config.get("sim_end_time"))

    # Read output file or TraCI to evaluate the fitness
    # Example: calculate average travel time, lower is better
    # Logic to read and calculate travel time from SUMO output
    travel_time_EB = get_travel_time_from_EdgeData_xml(path_EdgeData_new, EB_edge_list)
    travel_time_WB = get_travel_time_from_EdgeData_xml(path_EdgeData_new, WB_edge_list)

    if error_func == "rmse":
        fitness_err = -np.sqrt(0.5 * ((EB_tt - travel_time_EB)**2 + (WB_tt - travel_time_WB)**2))
    elif error_func == "mae":
        fitness_err = -((abs(EB_tt - travel_time_EB) + abs(WB_tt - travel_time_WB)) / 2)
    else:
        raise ValueError("error_func must be either 'rmse' or 'mae'")

    # remove the temporary directory
    if remove_temp_dir:
        for t_dir in temp_dir_list:
            try:
                shutil.rmtree(t_dir)
            except Exception:
                pass

    print(f"  :fitness_err: {fitness_err}")
    return fitness_err


# Write your own function, remember the starting positions has to be: list of N vectors or 2D matrix of position vectors
def create_starting_solutions(n_dims=None, pop_size=None, num=1):
    return np.ones((pop_size, n_dims)) * num + rng.uniform(-1, 1)

temp_dir_list = []

problem_dict = {
    "obj_func": partial(fitness_func, scenario_config=scenario_config, error_func="rmse", remove_temp_dir=True),
    "bounds": FloatVar(lb=[1.0, 2.5, 4, 0.0, 0.25, 5.0], ub=[3.0, 3.0, 5.3, 1.0, 1.25, 9.3],),
    "minmax": "max",  # maximize or minimize
}

params_bbo_grid = {
    "epoch": [1000],  # max iterations
    "temp_init": [100, 150, 90, 80],  # initial temperature
    # "scale": [0.1, 0.01, 0.05, 0.2, 0.3, 0.4],  # scale factor
    "step_size": [0.0109, 0.01, 0.05, 0.1, 0.2],  # step size
}

term_dict = {
    "max_epoch": 500,  # max iterations
    "max_fe": 10000,  # max function evaluations
    # "max_time": 3600,  # max time in seconds
    "max_early_stop": 20,
}

init_vals = create_starting_solutions(n_dims=6, pop_size=2, num=1)
init_vals = np.array([2.5, 2.6, 4.5, 0.5, 1.0, 9.0] * 2).reshape(2, 6)

# model = SA.GaussianSA(epoch=1000, pop_size=50, temp_init=100, step_size=0.0109, scale=0.1, verbose=True)
# model = SA.OriginalSA(epoch=1000, pop_size=2, temp_init=100, step_size=0.0109,)
# model = SA.SwarmSA(epoch=1000, pop_size=50, max_sub_iter=5, t0=500, t1=1,
#                    move_count=5, mutation_rate=0.1, mutation_step_size=0.1, mutation_step_size_damp=0.99)
# g_best = model.solve(problem_dict, termination=term_dict, mode="single")
# g_best = model.solve(problem_dict, termination=term_dict, starting_solutions=init_vals)

# print(g_best.solution)
# print(g_best.target.fitness)


########
model = SA.OriginalSA()
tuner = Tuner(model, params_bbo_grid)
tuner.execute(problem=problem_dict, termination=term_dict, n_trials=10, n_jobs=2, verbose=True)

# print(tuner.best_row)
# print(tuner.best_score)
# print(tuner.best_params)
# print(type(tuner.best_params))
#
# print(tuner.best_algorithm)
# # Better to save the tuning results to CSV for later usage
# tuner.export_results()
# tuner.export_figures()
#
# # Now we can even re-train the algorithm with the best parameter by calling resolve() function
# # Resolve() function will call the solve() function in algorithm with default problem parameter is removed.
# # other parameters of solve() function is keeped and can be used.
# g_best = tuner.resolve(mode="thread", n_workers=4, termination=term_dict)
#
# # Print out the best score of the best parameter
# print(g_best.solution, g_best.target.fitness)
#
# print(tuner.algorithm.problem.get_name())
#
# # Print out the algorithm with the best parameter
# print(tuner.best_algorithm.get_name())
