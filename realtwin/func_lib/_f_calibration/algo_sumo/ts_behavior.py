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

from realtwin.func_lib._f_calibration.algo_sumo.util_cali_behavior import (
    fitness_func,
    get_travel_time_from_EdgeData_xml,
    update_flow_xml_from_solution,
    run_jtrrouter_to_create_rou_xml,
    result_analysis_on_EdgeData,)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    sys.path = list(set(sys.path))  # remove duplicates

else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import time
rng = np.random.default_rng(812)


def generate_neighbors(solution: list | np.ndarray, param_ranges: dict, tabu_list: list, decimal_places: float) -> list:

    neighbors = []
    perturbVal = round(random.uniform(0, 0.5), decimal_places)
    delta_values = [-perturbVal, perturbVal]
    for i in range(len(solution)):
        for delta in delta_values:
            neighbor = solution.copy()

            neighbor[i] = round(neighbor[i] + delta, decimal_places)
            # print (ranges[i][0], neighbor[i], ranges[i][1] )

            # Ensure the neighbor is within the specified range
            # if ranges[i][0] <= neighbor[i] <= ranges[i][1] and neighbor not in tabu_list:
            if (param_ranges[i][0] <= neighbor[i] <= param_ranges[i][1] and neighbor.tolist() not in [item.tolist() for item in tabu_list]):
                neighbors.append(neighbor)
    return neighbors


class TabuSearchForBehavioral:
    """Tabu search algorithm for calibration"""

    def __init__(self, scenario_config: dict, ts_config: dict, verbose: bool = True):
        self.ts_config = ts_config
        self.scenario_config = scenario_config
        self.verbose = verbose

        # get input dir and output dir from the scenario config
        self.input_dir = pf.path2linux(os.path.abspath(scenario_config.get("input_dir", os.getcwd())))
        self.output_dir = pf.path2linux(os.path.abspath(os.path.join(self.input_dir, "tabu_search_result")))
        os.makedirs(self.output_dir, exist_ok=True)

        # change the current working directory to the input dir
        self.__current_dir = os.getcwd()
        os.chdir(self.input_dir)

        if path_edge := self.scenario_config.get("path_edge", "EdgeData.xml"):
            self.path_edge_abs = pf.path2linux(Path(self.input_dir) / path_edge)

    def run_calibration(self):

        if self.verbose:
            print("\n  :Tabu Search is running...")

        max_iteration = self.ts_config.get("max_iteration")

        # convert the initial parameters and ranges to numpy arrays
        initial_parameters = self.ga_config.get("initial_params")
        param_ranges = self.ga_config.get("params_ranges")
        if isinstance(initial_parameters, dict):
            initial_parameters = np.array(list(initial_parameters.values()))
        if isinstance(param_ranges, dict):
            param_ranges = np.array(list(param_ranges.values()))

        decimal_places = self.ts_config.get("decimal_places")
        tabu_list_size = self.ts_config.get("tabu_list_size")

        EB_edge_list = self.ts_config.get("EB_edge_list")
        WB_edge_list = self.ts_config.get("WB_edge_list")
        EB_tt = self.ts_config.get("EB_tt")
        WB_tt = self.ts_config.get("WB_tt")

        calibration_target = self.scenario_config.get("calibration_target")
        sim_start_time = self.scenario_config.get("sim_start_time")
        sim_end_time = self.scenario_config.get("sim_end_time")

        path_summary = pf.path2linux(
            Path(self.input_dir) / self.scenario_config.get("path_summary"))
        path_EdgeData = pf.path2linux(
            Path(self.input_dir) / self.scenario_config.get("path_edge"))

        # print out current travel time values and original fitness value
        travel_time_EB_orig = get_travel_time_from_EdgeData_xml(self.path_edge_abs, EB_edge_list)
        travel_time_WB_orig = get_travel_time_from_EdgeData_xml(self.path_edge_abs, WB_edge_list)
        print(" :Current travel time values: ", travel_time_EB_orig, travel_time_WB_orig)

        fitness_rmse = np.sqrt(0.5 * ((EB_tt - travel_time_EB_orig)**2 + (WB_tt - travel_time_WB_orig)**2))
        print("  :Original fitness value: ", fitness_rmse)

        # print out the initial Mean GEH and GEH percent
        fitness_func(self.scenario_config, initial_parameters, error_func="mae")
        _, mean_geh, geh_percent = result_analysis_on_EdgeData(path_summary,
                                                               path_EdgeData,
                                                               calibration_target,
                                                               sim_start_time,
                                                               sim_end_time)
        print("  :Original Mean GEH: ", mean_geh)
        print("  :Original GEH Percent: ", geh_percent)

        # Perform the simulated annealing algorithm
        current_solution = initial_parameters
        best_solution = current_solution.copy()
        best_cost = fitness_func(self.scenario_config, best_solution)
        self.history = []
        tabu_list = []

        for iteration in range(max_iteration):
            neighbors = generate_neighbors(current_solution, param_ranges, tabu_list, decimal_places)
            # print("  neighbors: ", neighbors)
            # Evaluate the neighbors and select the best one
            best_neighbor = min(neighbors, key=lambda neighbor: fitness_func(self.scenario_config, neighbor))
            best_neighbor_cost = fitness_func(self.scenario_config, best_neighbor)

            # Update tabu list
            tabu_list.append(best_neighbor)
            if len(tabu_list) > tabu_list_size:
                tabu_list.pop(0)

            # Update the solution if the neighbor is better
            if best_neighbor_cost < best_cost:
                current_solution = best_neighbor
                best_cost = best_neighbor_cost
                best_solution = best_neighbor

            self.history.append(best_cost)
            print(f"  :Iteration {iteration + 1}: Best Cost = {best_cost}")

        # result analysis
        fitness_func(self.scenario_config, best_solution, error_func="mae")
        _, mean_geh, geh_percent = result_analysis_on_EdgeData(path_summary,
                                                               path_EdgeData,
                                                               calibration_target,
                                                               sim_start_time,
                                                               sim_end_time)
        print("  :Optimized Mean GEH: ", mean_geh)
        print("  :Optimized GEH Percent: ", geh_percent)

        # change the current working directory back to the original directory
        os.chdir(self.__current_dir)

    def run_vis(self):

        if not hasattr(self, 'history'):
            print("Please run the algorithm first!")
            return
        plt.plot(self.history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost vs. Iteration in Tabu Search')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    ts_config = {
        "initial_parameters": {"min_gap": 2.5,        # minimum gap in meters
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
        "max_iteration": 50,
        "num_neighbors": 5,
        "tabu_list_size": 10,
        "decimal_places": 5,
        "EB_tt": 240,
        "WB_tt": 180,
        "EB_edge_list": ["-312", "-293", "-297", "-288", "-286",
                         "-302", "-3221", "-322", "-313", "-284",
                         "-328", "-304"],
        "WB_edge_list": ["-2801", "-280", "-307", "-327", "-281",
                         "-315", "-321", "-300", "-2851", "-285",
                         "-290", "-298", "-295"]
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

    }

    ts = TabuSearchForBehavioral(ts_config, scenario_config)
    ts.run_calibration()
    ts.run_vis()
