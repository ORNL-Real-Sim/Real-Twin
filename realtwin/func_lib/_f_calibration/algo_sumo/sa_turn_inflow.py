'''
##############################################################
# Created Date: Monday, February 10th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pyufunc as pf
import shutil
import matplotlib.pyplot as plt

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    sys.path = list(set(sys.path))  # remove duplicates

else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import time
calibration_start_time = time.time()
rng = np.random.default_rng(812)

from realtwin.func_lib._f_calibration.algo_sumo.util_cali_turn_inflow import (
    update_turn_flow_from_solution,  # step 1: update turning ratios and inflow counts
    create_rou_turn_flow_xml,  # step 2: create rou.xml file
    run_SUMO_create_EdgeData,  # step 3: run SUMO to create EdgeData.xml
    result_analysis_on_EdgeData)  # step 4: analyze EdgeData.xml to get best solution

# from util_cali_turn_inflow import (
#     update_turn_flow_from_solution,  # step 1: update turning ratios and inflow counts
#     create_rou_turn_flow_xml,  # step 2: create rou.xml file
#     run_SUMO_create_EdgeData,  # step 3: run SUMO to create EdgeData.xml
#     result_analysis_on_EdgeData)  # step 4: analyze EdgeData.xml to get best solution


class SimulatedAnnealingForTurnFlow:
    """ Simulated Annealing algorithm for running the simulator """

    def __init__(self, scenario_config: dict, turn_inflow_config: dict, verbose: bool = False):
        """ Initialize the Simulated Annealing algorithm with the given scenario and SA configurations """
        self.turn_inflow_cfg = turn_inflow_config
        self.scenario_config = scenario_config
        self.verbose = verbose

        # get input dir and output dir from the scenario config
        self.input_dir = pf.path2linux(os.path.abspath(scenario_config.get("input_dir", os.getcwd())))
        self.output_dir = pf.path2linux(Path(self.input_dir) / "simulated_annealing_result")
        os.makedirs(self.output_dir, exist_ok=True)

        # change the current working directory to the input dir
        os.chdir(self.input_dir)

        # initialize dataframes from the scenario config
        if path_turn := self.scenario_config.get("path_turn"):
            self.df_turn = pd.read_excel(pf.path2linux(Path(self.input_dir) / path_turn))

        if path_inflow := self.scenario_config.get("path_inflow"):
            self.df_inflow = pd.read_excel(pf.path2linux(Path(self.input_dir) / path_inflow))

        if path_summary := self.scenario_config.get("path_summary"):
            self.df_summary = pd.read_excel(pf.path2linux(Path(self.input_dir) / path_summary))

        if path_edge := self.scenario_config.get("path_edge", 'EdgeData.xml'):
            self.path_edge_abs = pf.path2linux(Path(self.input_dir) / path_edge)

    def run_single_calibration(self, initial_solution: np.array, ical: str, remove_old_files: bool = True) -> tuple:
        """ Run a single calibration iteration to get the best solution """

        # update turn and flow
        df_turn, df_inflow = update_turn_flow_from_solution(self.df_turn,
                                                            self.df_inflow,
                                                            initial_solution,
                                                            self.scenario_config["calibration_interval"],
                                                            self.scenario_config["demand_interval"])

        # update rou.xml from updated turn and flow
        create_rou_turn_flow_xml(self.scenario_config.get("network_name"),
                                 self.scenario_config.get("sim_start_time"),
                                 self.scenario_config.get("sim_end_time"),
                                 df_turn,
                                 df_inflow,
                                 ical,
                                 self.input_dir,
                                 self.output_dir,
                                 remove_old_files=remove_old_files)

        # run SUMO to get EdgeData.xml
        run_SUMO_create_EdgeData(self.scenario_config.get("sim_name"),
                                 self.scenario_config.get("sim_end_time"))

        # analyze EdgeData.xml to get best solution
        best_flag, best_value, best_percent = result_analysis_on_EdgeData(self.df_summary,
                                                                          self.path_edge_abs,
                                                                          self.scenario_config["calibration_target"],
                                                                          self.scenario_config["sim_start_time"],
                                                                          self.scenario_config["sim_end_time"])
        return (best_flag, best_value, best_percent)

    def generate_neighbor(self, current_params, step_size=0.05):
        """Generate a neighbor solution by perturbing the current solution"""

        perturbation = rng.uniform(-step_size, step_size, size=current_params.shape)
        neighbor = current_params + perturbation
        neighbor = np.clip(neighbor, 0, 1)
        return neighbor

    def acceptance_probability(self, current_cost, neighbor_cost, temperature):
        """ Calculate the acceptance probability for the neighbor solution """
        if neighbor_cost < current_cost:
            return 1.0

        return np.exp(-(neighbor_cost - current_cost) / temperature)

    @pf.func_running_time
    def run_calibration(self, *, init_solution: np.array = None, remove_old_files: bool = True) -> bool:
        """ Run the Simulated Annealing algorithm for finding the best solution from the given scenario """

        if self.verbose:
            print("\n  :Simulated Annealing algorithm is running...")

        # get parameters from config
        if (sa_cfg := self.turn_inflow_cfg.get("sa_config")) is None:
            raise ValueError("Simulated Annealing configuration is missing.")

        if not init_solution:
            # get initial solution from config
            initial_params = self.turn_inflow_cfg.get("initial_params")
            if not init_solution:
                initial_params = np.array([0.5] * sa_cfg.get("num_variables"))  # medium starting value

        num_turning_ratio = sa_cfg.get("num_turning_ratio")
        ubc = sa_cfg.get("ubc")
        # cost_difference = self.sa_config.get("cost_difference")
        # accept_prob = self.sa_config.get("accept_prob")
        initial_temperature = sa_cfg.get("initial_temperature")
        # initial_temperature = -cost_difference/(math.log(accept_prob))  #2.885
        cooling_rate = sa_cfg.get("cooling_rate")
        stopping_temperature = sa_cfg.get("stopping_temperature")
        # max_iteration = self.sa_config.get("max_iteration")
        # lower_bound = self.sa_config.get("lower_bound")
        # upper_bound = self.sa_config.get("upper_bound")

        network_name = self.scenario_config.get("network_name")
        ical = 1

        current_params = initial_params
        init_solution = current_params.copy()
        init_solution[num_turning_ratio:] = init_solution[num_turning_ratio:] * ubc
        current_cost = self.run_single_calibration(init_solution,
                                                   ical,
                                                   remove_old_files=remove_old_files)[1]
        best_value = current_cost
        temperature = initial_temperature

        self.best_values_over_time = []

        # Simulated Annealing Loop
        # while ical <= max_iteration:
        while temperature > stopping_temperature:
            neighbor = self.generate_neighbor(current_params, step_size=0.25)

            # current_params will be update in the each iteration
            init_solution = current_params.copy()
            init_solution[num_turning_ratio:] = init_solution[num_turning_ratio:] * ubc
            neighbor_cost = self.run_single_calibration(init_solution,
                                                        ical,
                                                        remove_old_files=False)[1]

            # get temp route path for rou, flow, turn
            path_temp = pf.path2linux(Path(self.output_dir) / "temp_route")
            temp_rou = pf.path2linux(Path(path_temp) / f"{network_name}{ical}.rou.xml")
            temp_flow = pf.path2linux(Path(path_temp) / f"{network_name}{ical}.flow.xml")
            temp_turn = pf.path2linux(Path(path_temp) / f"{network_name}{ical}.turn.xml")

            if neighbor_cost < current_cost:
                # update current_params if best solution is found
                current_params = neighbor

                # update current_cost if best solution is found
                current_cost = neighbor_cost
                if neighbor_cost < best_value:
                    # update best_solution if best solution is found
                    _, best_value = neighbor, neighbor_cost

                    # Save best solution to temp files
                    temp_rou_best = pf.path2linux(Path(path_temp) / f"{network_name}_best.rou.xml")
                    temp_flow_best = pf.path2linux(Path(path_temp) / f"{network_name}_best.flow.xml")
                    temp_turn_best = pf.path2linux(Path(path_temp) / f"{network_name}_best.turn.xml")

                    shutil.copy(temp_rou, temp_rou_best)
                    shutil.copy(temp_flow, temp_flow_best)
                    shutil.copy(temp_turn, temp_turn_best)
            else:
                if np.exp(-(neighbor_cost - current_cost) / temperature) > rng.random():
                    current_params = neighbor
                    current_cost = neighbor_cost

            # delete temp files for each iteration
            os.remove(temp_rou)
            os.remove(temp_flow)
            os.remove(temp_turn)

            # if acceptance_probability(current_cost, neighbor_cost, temperature) > np.random.rand():
            #     current_params = neighbor
            #     current_cost = neighbor_cost

            temperature *= cooling_rate
            print(f'  :Calibrate iteration {ical}:')
            print(f'  :current best GEH is {current_cost}, '
                  f'best GEH to date is {best_value}')
            print(f"  :Current calibration time is {time.time() - calibration_start_time} sec.")
            self.best_values_over_time.append(best_value)
            ical += 1

        path_best_solution = pf.path2linux(os.path.join(self.output_dir, "GEH_best_solution.txt"))
        np.savetxt(path_best_solution, self.best_values_over_time, fmt='%.4f')

        # copy best rou.xml, flow.xml, turn.xml to output dir
        shutil.copy(temp_rou_best, pf.path2linux(Path(self.output_dir) / f"{network_name}.rou.xml"))
        shutil.copy(temp_flow_best, pf.path2linux(Path(self.output_dir) / f"{network_name}.flow.xml"))
        shutil.copy(temp_turn_best, pf.path2linux(Path(self.output_dir) / f"{network_name}.turn.xml"))
        shutil.copy(temp_rou_best, pf.path2linux(Path(self.input_dir) / f"{network_name}.rou.xml"))
        shutil.copy(temp_flow_best, pf.path2linux(Path(self.input_dir) / f"{network_name}.flow.xml"))
        shutil.copy(temp_turn_best, pf.path2linux(Path(self.input_dir) / f"{network_name}.turn.xml"))

        if remove_old_files:
            # delete temp route folder
            path_temp_route = pf.path2linux(Path(self.output_dir) / 'temp_route')
            shutil.rmtree(path_temp_route)

        print("  :Simulated Annealing algorithm is done!")
        return True

    def run_vis(self):
        """ Run the visualization of the results """
        if not hasattr(self, "best_values_over_time"):
            raise AttributeError("Run the SA algorithm first!")

        plt.plot(self.best_values_over_time, '-', color='red')  # markersize=5
        plt.xlabel('Iteration')
        plt.ylabel('Mean GEH')
        plt.title('Mean GEH over iterations using SA')
        # plt.legend()
        # plt.xticks(ticks=range(0,21))
        # plt.xticks(range(0,3001,500))
        # plt.ylim(0, 10)
        # plt.xlim(0, 3000)
        plt.show()


if __name__ == "__main__":

    turn_inflow_config = {
        "initial_params": [0.5, 0.5, 0.5, 0.5, 0.5,
                           0.5, 0.5, 0.5, 0.5, 0.5,
                           0.5, 0.5, 100, 100, 100, 100],
        "params_ranges": [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                          [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                          [0, 1], [0, 1], [50, 200], [50, 200], [50, 200], [50, 200]],
        "ga_config": {"num_variables": 16,
                      "num_turning_ratio": 12,  # remaining should be inflow
                      "ubc": 200,  # inflow upper bound constant
                      "population_size": 2,  # must be even
                      "num_generations": 5,
                      "crossover_rate": 0.75,
                      "mutation_rate": 0.1,
                      "elitism_size": 1,  # Number of elite individuals to carry over
                      "best_fitness_value": float('inf'),
                      "max_no_improvement": 5,  # Stop if no improvement in 5 iterations
                      },
        "sa_config": {"num_variables": 16,
                      "num_turning_ratio": 12,
                      "ubc": 200,
                      "cost_difference": 2,
                      "initial_temperature": 100,
                      "cooling_rate": 0.99,
                      "stopping_temperature": 1e-3},
        "ts_config": {"iterations": 3,
                      "tabu_size": 120,
                      "neighborhood_size": 32,
                      "move_range": 0.5,  # Initial move range
                      "lower_bound": 0,
                      "upper_bound": 1,
                      "lbc": 0,  # lower bound for inflow counts
                      "ubc": 200,  # upper bound for inflow counts
                      "num_turning_ratio": 12,
                      "max_no_improvement_local": 5,
                      "max_no_improvement_global": 30, },
    }

    scenario_config = {
        "input_dir": r"C:\Users\xh8\ornl_work\gitlab_workspace\realtwintool\tools\SUMO\Calibration\xl_turn_and_flow\input_dir",
        "network_name": "chatt",
        "sim_name": "chatt.sumocfg",
        "sim_start_time": 28800,
        "sim_end_time": 32400,
        "path_turn": "turn.xlsx",
        "path_inflow": "inflow.xlsx",
        "path_summary": "summary.xlsx",
        "path_edge": "EdgeData.xml",
        "calibration_target": {"GEH": 5, "GEHPercent": 0.85},
        "calibration_interval": 60,
        "demand_interval": 15,
        # Add more configurations as needed
    }

    sa = SimulatedAnnealingForTurnFlow(scenario_config, turn_inflow_config, verbose=True)
    sa.run_calibration()
    sa.run_vis()
