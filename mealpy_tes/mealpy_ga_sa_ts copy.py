'''
##############################################################
# Created Date: Thursday, March 6th 2025
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

from mealpy import FloatVar, SA, GA, TS
from util_cali_behavior import (get_travel_time_from_EdgeData_xml,
                                update_flow_xml_from_solution,
                                run_jtrrouter_to_create_rou_xml,
                                fitness_func,
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


# Write your own function, remember the starting positions has to be: list of N vectors or 2D matrix of position vectors
def create_starting_solutions(n_dims=None, pop_size=None, num=1):
    return np.ones((pop_size, n_dims)) * num + rng.uniform(-1, 1)

temp_dir_list = []

problem_dict = {
    "obj_func": partial(fitness_func, scenario_config=scenario_config, error_func="rmse"),
    "bounds": FloatVar(lb=[1.0, 2.5, 4, 0.0, 0.25, 5.0], ub=[3.0, 3.0, 5.3, 1.0, 1.25, 9.3],),
    "minmax": "max",  # maximize or minimize
    "log_to": "console",
    # "log_to": "file",
    # "log_file": "result.log",
    "save_population": True,              # Default = False
}

term_dict = {
    "max_epoch": 500,  # max iterations
    "max_fe": 10000,  # max function evaluations
    # "max_time": 3600,  # max time in seconds
    "max_early_stop": 20,
}

# init_vals = create_starting_solutions(n_dims=6, pop_size=50, num=1)
init_vals = np.array([2.5, 2.6, 4.5, 0.5, 1.0, 9.0] * 2).reshape(2, 6)

# MODELS
# Simulated Annealing
# pop_size have no effect on the SA algorithm
# model = SA.GaussianSA(epoch=1000, pop_size=50, temp_init=100, scale=0.1)
model_sa = SA.OriginalSA(epoch=1000, pop_size=2, temp_init=100, step_size=0.1,)
# model = SA.SwarmSA(epoch=1000, pop_size=50, max_sub_iter=5, t0=500, t1=1,
#                    move_count=5, mutation_rate=0.1, mutation_step_size=0.1, mutation_step_size_damp=0.99)

# Genetic Algorithm
model_ga = GA.BaseGA(epoch=1000,
                     pop_size=50,
                     pc=0.95,  # crossover probability
                     pm=0.025,  # mutation probability
                     selection="tournament",  # "roulette", "tournament", "random"
                     k_way=0.2,  # k-way for tournament selection
                     crossover="uniform",  # one_point, multi-point, uniform, arithmetic
                     mutation_multipoints=True,
                     )  # mutation, flip, swap, scramble, inversion.
# Elite Single GA: elite single-point mutation
# models_gs = GA.EliteSingleGA(epoch=1000,
#                              pop_size=50,
#                              pc=0.95,  # crossover probability
#                              pm=0.025,  # mutation probability
#                              selection="tournament",  # "roulette", "tournament", "random"
#                              crossover="uniform",  # one_point, multi-point, uniform, arithmetic
#                              mutation="flip",  # flip, swap, scramble, inversion.
#                              k_way=0.2,  # k-way for tournament selection
#                              elite_best=0.1,  # percentage of the best in elite group, or int, the number of best elite
#                              elite_worst=0.3,  # percentage of the worst in elite group, or int, the number of worst elite
#                              )
# EliteMultiGA: elite multipoints-mutation
# MultiGA: multipoints-mutation
# SingleGA: single-point-mutation

# Tabu Search
model_ts = TS.OriginalTS(epoch=1000,
                         pop_size=2,  # This parameter has no effect on the TS algorithm
                         tabu_size=10,  # maximum size of tabu list
                         neighbour_size=10,  # size of the neighborhood for generating candidate solutions
                         perturbation_scale=0.05  # scale of perturbation for generating candidate solutions
                         )

# g_best = model_sa.solve(problem_dict, termination=term_dict, mode="single")
# g_best = model_sa.solve(problem_dict, termination=term_dict, starting_solutions=init_vals)

# g_best = model_ga.solve(problem_dict, termination=term_dict, mode="single")

g_best = model_ts.solve(problem_dict, termination=term_dict, mode="single")
print(g_best.solution)
print(g_best.target.fitness)
