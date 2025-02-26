'''
##############################################################
# Created Date: Tuesday, February 25th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

import os
import sys
from pathlib import Path
import random
import numpy as np
import pyufunc as pf
import matplotlib.pyplot as plt
import pygad
import subprocess
from functools import partial

from utils_cali import (fitness_func,
                        get_travel_time_from_EdgeData_xml,
                        update_flow_xml_from_solution,
                        run_jtrrouter_to_create_rou_xml,
                        result_analysis_on_EdgeData,)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import time
rng = np.random.default_rng(812)


def on_generation(ga_instance):
    current_generation = ga_instance.generations_completed
    print(f"  :Generation {current_generation}:")
    print("  :Fitness of the best solution :", ga_instance.best_solution())
    print(f"  :population size: {ga_instance.population}")

    # population = ga_instance.population
    # print("Population:", population)

    # Here you can also modify GA parameters if needed
    # For example, adjust mutation rate based on some condition:
    if ga_instance.best_solution()[1] > -1000:
        ga_instance.mutation_probability = 0.01


def fitness_func_tt(ept_, solution: list | np.ndarray, solution_idx: int, scenario_config: dict):
    # Set up SUMO command with car-following parameters
    if solution[5] >= 9.3:  # emergencyDecel
        solution[5] = 9.3
    if solution[5] < solution[2]:  # emergencyDecel < deceleration
        solution[5] = solution[2] + random.randrange(1, 5)
    # print("after emergencydecel update", solution)

    # get path from scenario_config
    network_name = scenario_config.get("network_name")
    path_net = pf.path2linux(Path(scenario_config.get("input_dir")) / f"{network_name}.net.xml")
    path_flow = pf.path2linux(Path(scenario_config.get("input_dir")) / f"{network_name}.flow.xml")
    path_turn = pf.path2linux(Path(scenario_config.get("input_dir")) / f"{network_name}.turn.xml")
    path_rou = pf.path2linux(Path(scenario_config.get("input_dir")) / f"{network_name}.rou.xml")
    path_EdgeData = pf.path2linux(Path(scenario_config.get("input_dir")) / "EdgeData.xml")
    EB_tt = scenario_config.get("EB_tt")
    WB_tt = scenario_config.get("WB_tt")
    EB_edge_list = scenario_config.get("EB_edge_list")
    WB_edge_list = scenario_config.get("WB_edge_list")

    sim_name = scenario_config.get("sim_name")

    update_flow_xml_from_solution(path_flow, solution)

    run_jtrrouter_to_create_rou_xml(network_name, path_net, path_flow, path_turn, path_rou)

    # Define the command to run SUMO
    sumo_command = f"sumo -c \"{sim_name}\""
    sumoProcess = subprocess.Popen(sumo_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sumoProcess.wait()

    # Read output file or TraCI to evaluate the fitness
    # Example: calculate average travel time, lower is better
    # Logic to read and calculate travel time from SUMO output
    travel_time_EB = get_travel_time_from_EdgeData_xml(path_EdgeData, EB_edge_list)
    travel_time_WB = get_travel_time_from_EdgeData_xml(path_EdgeData, WB_edge_list)

    fitness_err = -np.sqrt(0.5 * ((EB_tt - travel_time_EB)**2 + (WB_tt - travel_time_WB)**2))
    return fitness_err


class GeneticAlgorithm:

    def __init__(self, ga_config: dict, scenario_config: dict, verbose: bool = True):
        """Input parameters are dictionaries containing configurations for Genetic Algorithm and scenario results.
        """
        self.ga_config = ga_config
        self.scenario_config = scenario_config
        self.verbose = verbose

        # get input dir and output dir from the scenario config
        self.input_dir = pf.path2linux(os.path.abspath(scenario_config.get("input_dir", os.getcwd())))
        self.output_dir = pf.path2linux(os.path.abspath(os.path.join(self.input_dir, "genetic_algorithm_result")))
        os.makedirs(self.output_dir, exist_ok=True)

        # change the current working directory to the input dir
        self.__current_dir = os.getcwd()
        os.chdir(self.input_dir)

        # initialize dataframes from the scenario config
        if path_edge := self.scenario_config.get("path_edge"):
            self.path_edge_abs = pf.path2linux(Path(self.input_dir) / path_edge)

    def run_GA(self) -> bool:

        # convert the initial parameters and ranges to numpy arrays
        initial_parameters = self.ga_config.get("initial_parameters")
        param_ranges = self.ga_config.get("param_ranges")
        if isinstance(initial_parameters, dict):
            initial_parameters = np.array(list(initial_parameters.values()))
        if isinstance(param_ranges, dict):
            param_ranges = np.array(list(param_ranges.values()))

        network_name = self.scenario_config.get("network_name")
        path_net = pf.path2linux(Path(self.input_dir) / f"{network_name}.net.xml")
        path_flow = pf.path2linux(Path(self.input_dir) / f"{network_name}.flow.xml")
        path_turn = pf.path2linux(Path(self.input_dir) / self.scenario_config.get("path_turn"))
        path_rou = pf.path2linux(Path(self.input_dir) / f"{network_name}.rou.xml")

        EB_edge_list = self.scenario_config.get("EB_edge_list")
        WB_edge_list = self.scenario_config.get("WB_edge_list")
        EB_tt = self.scenario_config.get("EB_tt")
        WB_tt = self.scenario_config.get("WB_tt")
        calibration_target = self.scenario_config.get("calibration_target")
        sim_start_time = self.scenario_config.get("sim_start_time")
        sim_end_time = self.scenario_config.get("sim_end_time")

        path_summary = pf.path2linux(
            Path(self.input_dir) / self.scenario_config.get("path_summary"))
        path_EdgeData = pf.path2linux(
            Path(self.input_dir) / self.scenario_config.get("path_edge"))

        # update the rou xml file
        run_jtrrouter_to_create_rou_xml(network_name, path_net, path_flow, path_turn, path_rou)

        # print out current travel time values and original fitness value
        travel_time_EB_orig = get_travel_time_from_EdgeData_xml(self.path_edge_abs, EB_edge_list)
        travel_time_WB_orig = get_travel_time_from_EdgeData_xml(self.path_edge_abs, WB_edge_list)
        print(" :Current travel time values: ", travel_time_EB_orig, travel_time_WB_orig)

        fitness_rmse = np.sqrt(0.5 * ((EB_tt - travel_time_EB_orig)**2 + (WB_tt - travel_time_WB_orig)**2))
        print("  :Original fitness value: ", fitness_rmse)

        # print out the initial Mean GEH and GEH percent
        fitness_func(self.scenario_config, initial_parameters)
        _, mean_geh, geh_percent = result_analysis_on_EdgeData(path_summary,
                                                               path_EdgeData,
                                                               calibration_target,
                                                               sim_start_time,
                                                               sim_end_time)
        print("  :Original Mean GEH: ", mean_geh)
        print("  :Original GEH Percent: ", geh_percent)

        # Perform the genetic algorithm
        # Genetic Algorithm configuration
        # Create a partial function where param1 and param2 are preset.
        ga_instance = pygad.GA(num_generations=50,
                               num_parents_mating=3,
                               fitness_func=lambda _, sol, idx: fitness_func_tt(_, sol, idx, self.scenario_config),
                               sol_per_pop=20,
                               num_genes=18,
                               crossover_type="single_point",
                               mutation_type="random",
                               gene_space=[{'low': 1.00, 'high': 3.00},
                                           {'low': 2.50, 'high': 3.00},
                                           {'low': 4.00, 'high': 5.30},
                                           {'low': 0.00, 'high': 1.00},
                                           {'low': 0.25, 'high': 1.25},
                                           {'low': 5.0, 'high': 9.30}],  # Adjust ranges as needed
                               initial_population=[initial_parameters] * 20,
                               mutation_percent_genes=80,
                               mutation_probability=0.1,
                               keep_parents=1,
                               mutation_by_replacement=True,
                               parent_selection_type="tournament",
                               on_generation=on_generation)
        # Run the GA
        ga_instance.run()
        ga_instance.plot_fitness()
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)

        # Fetch the best solution
        solution, solution_fitness, _ = ga_instance.best_solution()

        print(f"Best Solution: minGap: {solution[0]}, accel: {solution[1]}, "
              f"decel: {solution[2]}, sigma: {solution[3]}, tau: {solution[4]}, "
              f"emergencyDecel: {solution[5]}, Fitness: {solution_fitness}")

        best_fit_parameters = solution
        fitness_func(best_fit_parameters)
        _, mean_geh, geh_percent = result_analysis_on_EdgeData(path_summary,
                                                               path_EdgeData,
                                                               calibration_target,
                                                               sim_start_time,
                                                               sim_end_time)
        print("  :Optimized Mean GEH: ", mean_geh)
        print("  :Optimized GEH Percent: ", geh_percent)

    def run_vis(self):
        pass


if __name__ == "__main__":

    ga_config = {"initial_parameters": {"min_gap": 2.5,        # minimum gap in meters
                                        "acceleration": 2.6,  # max acceleration in m/s^2
                                        "deceleration": 4.5,  # max deceleration in m/s^2
                                        "sigma": 0.5,          # driver imperfection
                                        "tau": 1.00,            # desired headway
                                        "emergencyDecel": 9.0},   # emergency deceleration
                 "param_ranges": {"min_gap": (1.0, 3.0),
                                  "acceleration": (2.5, 3),
                                  "deceleration": (4, 5.3),
                                  "sigma": (0, 1),
                                  "tau": (0.25, 1.25),
                                  "emergencyDecel": (5.0, 9.3)},
                 "num_generation": 50,
                 "num_variables": 16,
                 "num_turning_ratio": 12,  # remaining should be inflow
                 "ubc": 200,  # inflow upper bound constant
                 "population_size": 2,  # must be even
                 "crossover_rate": 0.75,
                 "mutation_rate": 0.1,
                 "elitism_size": 1,  # Number of elite individuals to carry over
                 "best_fitness_value": float('inf'),
                 "max_no_improvement": 5,  # Stop if no improvement in 5 iterations
                 }

    scenario_config = {
        "input_dir": r"C:\Users\xh8\ornl_work\gitlab_workspace\realtwintool\tools\SUMO\Calibration\xl_behavior\input_dir",
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
        "lower_bound": 0,   # For Tabu search only
        "upper_bound": 1,
        "EB_tt": 240,
        "WB_tt": 180,
        "EB_edge_list": ["-312", "-293", "-297", "-288", "-286",
                         "-302", "-3221", "-322", "-313", "-284",
                         "-328", "-304"],
        "WB_edge_list": ["-2801", "-280", "-307", "-327", "-281",
                         "-315", "-321", "-300", "-2851", "-285",
                         "-290", "-298", "-295"]
    }

    ga = GeneticAlgorithm(ga_config, scenario_config)
    ga.run_GA()