'''
##############################################################
# Created Date: Tuesday, February 25th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

import os
import sys
# import xml.etree.ElementTree as ET
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import pyufunc as pf

from realtwin.func_lib._f_calibration.algo_sumo.util_cali_behavior import (
    fitness_func,
    get_travel_time_from_EdgeData_xml,
    # update_flow_xml_from_solution,
    # run_jtrrouter_to_create_rou_xml,
    result_analysis_on_EdgeData,)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    sys.path = list(set(sys.path))  # remove duplicates

else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class SimulatedAnnealingForBehavioral:
    """ Simulated Annealing algorithm for running the simulator """

    def __init__(self, scenario_config: dict, sa_config: dict, verbose: bool = False):
        self.sa_config = sa_config
        self.scenario_config = scenario_config
        self.verbose = verbose

        # get input dir and output dir from the scenario config
        self.input_dir = pf.path2linux(os.path.abspath(scenario_config.get("input_dir", os.getcwd())))
        self.output_dir = pf.path2linux(Path(self.input_dir) / "simulated_annealing_result")
        os.makedirs(self.output_dir, exist_ok=True)

        # change the current working directory to the input dir
        self.__current_dir = os.getcwd()
        os.chdir(self.input_dir)

        if path_edge := self.scenario_config.get("path_edge", "EdgeData.xml"):
            self.path_edge_abs = pf.path2linux(Path(self.input_dir) / path_edge)

    def run_calibration(self) -> bool:

        if self.verbose:
            print("\n  :Simulated Annealing algorithm is running...")

        init_temperature = self.sa_config.get("init_temperature", 100)
        max_iteration = self.sa_config.get("max_iteration")

        # convert the initial parameters and ranges to numpy arrays
        initial_parameters = self.sa_config.get("initial_params")
        param_ranges = self.sa_config.get("params_ranges")
        if isinstance(initial_parameters, dict):
            initial_parameters = np.array(list(initial_parameters.values()))
        if isinstance(param_ranges, dict):
            param_ranges = np.array(list(param_ranges.values()))

        cooling_rate = self.sa_config.get("cooling_rate")
        EB_edge_list = self.sa_config.get("EB_edge_list")
        WB_edge_list = self.sa_config.get("WB_edge_list")
        EB_tt = self.sa_config.get("EB_tt")
        WB_tt = self.sa_config.get("WB_tt")

        calibration_target = self.scenario_config.get("calibration_target")
        sim_start_time = self.scenario_config.get("sim_start_time")
        sim_end_time = self.scenario_config.get("sim_end_time")

        path_summary = pf.path2linux(Path(self.input_dir) / self.scenario_config.get("path_summary"))
        path_EdgeData = pf.path2linux(Path(self.input_dir) / self.scenario_config.get("path_edge", "EdgeData.xml"))

        # print out current travel time values and original fitness value
        travel_time_EB_orig = get_travel_time_from_EdgeData_xml(self.path_edge_abs, EB_edge_list)
        travel_time_WB_orig = get_travel_time_from_EdgeData_xml(self.path_edge_abs, WB_edge_list)
        print("  :Current travel time values: ", travel_time_EB_orig, travel_time_WB_orig)

        fitness_rmse = np.sqrt(0.5 * ((EB_tt - travel_time_EB_orig)**2 + (WB_tt - travel_time_WB_orig)**2))
        print("  :fitness value: ", fitness_rmse)

        # print out the initial Mean GEH and GEH percent
        # add the EB_tt, WB_tt, EB_edge_list, WB_edge_list to the scenario_config
        self.scenario_config["EB_tt"] = EB_tt
        self.scenario_config["WB_tt"] = WB_tt
        self.scenario_config["EB_edge_list"] = EB_edge_list
        self.scenario_config["WB_edge_list"] = WB_edge_list
        fitness_func(self.scenario_config, initial_parameters)
        _, mean_geh, geh_percent = result_analysis_on_EdgeData(path_summary,
                                                               path_EdgeData,
                                                               calibration_target,
                                                               sim_start_time,
                                                               sim_end_time)
        print("  :Mean GEH: ", mean_geh)
        print("  :GEH Percent: ", geh_percent)

        # Perform the simulated annealing algorithm
        current_params = initial_parameters
        current_cost = -fitness_func(self.scenario_config, current_params)
        best_solution = current_params
        best_cost = current_cost
        self.history = [(current_params, current_cost)]

        iter_count = 0
        temperature = init_temperature

        while temperature > 1e-3 and iter_count < max_iteration:
            print(f"  :Iteration {iter_count}, Temperature: {temperature:.2f}, Current Cost: {current_cost:.2f}")

            # Generate a new solution by perturbing the current solution
            new_params = current_params.copy()
            # Randomly alter parameters within their defined ranges
            for param_index in range(len(param_ranges)):
                range_min, range_max = param_ranges[param_index]
                new_params[param_index] = random.uniform(range_min, range_max)
            # print("new_params", new_params)
            new_cost = -fitness_func(self.scenario_config, new_params)
            cost_diff = new_cost - current_cost

            # Calculate the acceptance probability
            if cost_diff < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp((current_cost - new_cost) / temperature)

            if accept_prob > random.random():
                current_params = new_params
                current_cost = new_cost

                if new_cost < best_cost:
                    best_solution = new_params
                    best_cost = new_cost
                    self.history.append((best_solution, best_cost))

            # Cool down the temperature
            temperature *= cooling_rate
            iter_count += 1

        print(f"  :Optimized Parameters: {best_solution}")
        print(f"  :Optimized Cost: {best_cost}")

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
        """ Run the visualization from the result """

        if not hasattr(self, "history"):
            raise ValueError("No history found, please run the algorithm first.")

        costs = [cost for _, cost in self.history]
        plt.figure(figsize=(10, 5))
        plt.plot(costs, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost vs. Iteration in Simulated Annealing')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":

    sa_config = {
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
        "initial_temperature": 100,
        "max_iteration": 150,
        "cooling_rate": 0.9891,
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
    }

    sa = SimulatedAnnealingForBehavioral(sa_config, scenario_config)
    sa.run_calibration()
    sa.run_vis()
