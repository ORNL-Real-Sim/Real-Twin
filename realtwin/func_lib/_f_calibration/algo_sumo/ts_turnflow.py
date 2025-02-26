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
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import time
calibration_start_time = time.time()
rng = np.random.default_rng(812)

from util_cali import (update_turn_flow_from_solution,  # step 1: update turning ratios and inflow counts
                       create_rou_turn_flow_xml,  # step 2: create rou.xml file
                       run_SUMO_create_EdgeData,  # step 3: run SUMO to create EdgeData.xml
                       result_analysis_on_EdgeData)  # step 4: analyze EdgeData.xml to get best solution


class TabuSearch:
    """Tabu search algorithm for calibration"""

    def __init__(self, ts_config: dict, scenario_config: dict, verbose: bool = True):
        self.ts_config = ts_config
        self.scenario_config = scenario_config
        self.verbose = verbose

        # get input dir and output dir from the scenario config
        self.input_dir = pf.path2linux(os.path.abspath(scenario_config.get("input_dir", os.getcwd())))
        self.output_dir = pf.path2linux(os.path.abspath(os.path.join(self.input_dir, "tabu_search_result")))
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

        if path_edge := self.scenario_config.get("path_edge"):
            self.path_edge_abs = pf.path2linux(Path(self.input_dir) / path_edge)

    def is_close_to_tabu(self, solution, tabu_list: list, Np, threshold_p: float, threshold_c: float) -> int:
        flag_p = 0
        flag_c = 0
        for tabu_solution in tabu_list:
            if np.linalg.norm(solution[0:Np] - tabu_solution[0:Np]) < threshold_p:
                flag_p = 1
                # break the loop if flag_p is 1
                break

        # Early exit if flag_p is 0
        if flag_p == 0:
            return 0

        for tabu_solution in tabu_list:
            if np.linalg.norm(solution[Np:] - tabu_solution[Np:]) < threshold_c:
                flag_c = 1
                # break the loop if flag_c is 1
                break

        return flag_p * flag_c

    def run_single_calibration(self, initial_solution: np.array, ical: str, remove_old_files: bool = True) -> tuple:
        """ Run a single calibration iteration to get the best solution """

        # update turn and flow
        self.df_turn, self.df_inflow = update_turn_flow_from_solution(self.df_turn,
                                                                      self.df_inflow,
                                                                      initial_solution,
                                                                      self.scenario_config["calibration_interval"],
                                                                      self.scenario_config["demand_interval"])

        # update rou.xml from updated turn and flow
        create_rou_turn_flow_xml(self.scenario_config.get("network_name"),
                                 self.scenario_config.get("sim_start_time"),
                                 self.scenario_config.get("sim_end_time"),
                                 self.df_turn,
                                 self.df_inflow,
                                 ical,
                                 self.input_dir,
                                 self.output_dir,
                                 remove_old_files=remove_old_files)

        # run SUMO to get EdgeData.xml
        run_SUMO_create_EdgeData(self.scenario_config.get("sim_name"),
                                 self.scenario_config.get("sim_end_time"))

        # analyze EdgeData.xml to get best solution
        best_flag, best_value, best_percent = result_analysis_on_EdgeData(self.df_summary,
                                                                          self.scenario_config["calibration_target"],
                                                                          self.scenario_config["sim_start_time"],
                                                                          self.scenario_config["sim_end_time"],
                                                                          self.path_edge_abs)
        return (best_flag, best_value, best_percent)

    def run_TS(self, *, initial_solution: np.array = None, remove_old_files: bool = True) -> bool:
        """ Run the Tabu Search algorithm for calibration """

        if self.verbose:
            print("\n  :Tabu Search is running...")

        # Fot temporary testing in beta version
        if not initial_solution:
            initial_solution = np.array([0.5, 0.5, 0.5, 0.5, 0.5,
                                         0.5, 0.5, 0.5, 0.5, 0.5,
                                         0.5, 0.5, 100, 100, 100,
                                         100])

        # get configurations from both ts_config and scenario_config
        iterations = self.ts_config.get("iterations", 30)
        tabu_size = self.ts_config.get("tabu_size", 120)
        neighborhood_size = self.ts_config.get("neighborhood_size", 32)
        move_range = self.ts_config.get("move_range", 0.5)
        lower_bound = self.ts_config.get("lower_bound", 0)
        upper_bound = self.ts_config.get("upper_bound", 1)
        lbc = self.ts_config.get("lbc", 0)  # lower bound for inflow counts
        ubc = self.ts_config.get("ubc", 200)  # upper bound for inflow counts
        num_turning_ratio = self.ts_config.get("num_turning_ratio", 12)

        max_no_improvement_local = self.ts_config.get("max_no_improvement_local", 5)
        max_no_improvement_global = self.ts_config.get("max_no_improvement_global", 30)

        network_name = self.scenario_config.get("network_name")

        # get the best solution from the first run
        best_flag, best_value, best_percent = self.run_single_calibration(initial_solution, "_init", True)
        print('Initial mean GEH is {}.'.format(best_value))

        # collect the best solution
        tabu_list = [initial_solution.tolist()]

        self.GEH_summary = []
        fail_count = 0

        # current best solution, will be updated in each iteration
        best_solution = initial_solution.copy()

        for each_iter in range(iterations):
            print(f'Calibration iteration {each_iter}:')
            flag_move = 0
            iloc = 0
            while (flag_move == 0 and iloc < max_no_improvement_local):
                iloc += 1
                print(f'  :Calibration iteration {each_iter}, local search trail {iloc}:')
                GEH_iter = []
                for neighbor in range(neighborhood_size):
                    # initial neighborhood value
                    neighborhood = best_solution.copy()

                    # update neighborhood: continue generate if close to tabu list
                    while True:
                        for i in range(len(neighborhood)):
                            while True:
                                real_minus_range = move_range
                                real_plus_range = move_range
                                move = rng.uniform(-real_minus_range, real_plus_range)
                                if i <= num_turning_ratio - 1:
                                    if lower_bound <= neighborhood[i] + move <= upper_bound:
                                        neighborhood[i] += move
                                        break
                                else:
                                    if lbc <= neighborhood[i] + move * ubc <= ubc:
                                        neighborhood[i] += move
                                        break
                        if self.is_close_to_tabu(neighborhood, tabu_list, num_turning_ratio, 0.05, 5) == 0:
                            break
                        # if any(neighborhood.tolist() == tabu_sol for tabu_sol in tabu_list):
                        #     break

                    ical = f'{each_iter}_{neighbor}'
                    neighbor_flag, neighbor_value, neighbor_pct = self.run_single_calibration(neighborhood,
                                                                                              ical,
                                                                                              remove_old_files=False)

                    print(f'  :Neighbor {neighbor}: mean GEH is {neighbor_value}.')

                    GEH_iter.append(neighbor_value)

                    # get temp route path for rou, flow, turn
                    path_temp = pf.path2linux(Path(self.output_dir) / "temp_route")
                    temp_rou = pf.path2linux(Path(path_temp) / f"{network_name}{ical}.rou.xml")
                    temp_flow = pf.path2linux(Path(path_temp) / f"{network_name}{ical}.flow.xml")
                    temp_turn = pf.path2linux(Path(path_temp) / f"{network_name}{ical}.turn.xml")

                    # save best solution generated in this iteration
                    if neighbor_value < best_value:
                        flag_move = 1

                        # update best solution and best value for next iteration
                        best_solution = neighborhood.copy()
                        best_value = neighbor_value.copy()

                        # Save best solution to file
                        temp_rou_best = pf.path2linux(Path(path_temp) / f"{network_name}_best.rou.xml")
                        temp_flow_best = pf.path2linux(Path(path_temp) / f"{network_name}_best.flow.xml")
                        temp_turn_best = pf.path2linux(Path(path_temp) / f"{network_name}_best.turn.xml")

                        shutil.copy(temp_rou, temp_rou_best)
                        shutil.copy(temp_flow, temp_flow_best)
                        shutil.copy(temp_turn, temp_turn_best)

                    # delete temp files for each iteration
                    os.remove(temp_rou)
                    os.remove(temp_flow)
                    os.remove(temp_turn)

            if flag_move == 1:
                print(f'  :Successfully make an improvement at iteration {each_iter}.')
                # move_range -=0.05
                move_range *= 0.9
                tabu_list.append(best_solution.tolist())
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)
                fail_count = 0
            else:
                # move_range +=0.05
                move_range *= 1.1
                fail_count += 1
                print(f'  No improvement at iteration {each_iter}. Have failed to improve for {fail_count} iterations')

            self.GEH_summary.append(GEH_iter)
            print(f'  :Current best mean GEH is {best_value}.')
            print(f"  :Current calibration time is {time.time() - calibration_start_time} sec.")
            if fail_count >= max_no_improvement_global:
                print(f"  :No improvement is made in the last {max_no_improvement_global} "
                      "iterations. Calibration will be terminated.")
                break

        # # Using best rou file to generate final results
        try:
            # copy the best rou.xml to the input directory
            shutil.copy(temp_rou_best, pf.path2linux(Path(self.input_dir) / f"{network_name}.rou.xml"))

            # generate EdgeData.xml based on best rou.xml
            run_SUMO_create_EdgeData(self.scenario_config.get("sim_name"),
                                    self.scenario_config.get("sim_end_time"))

            # analyze EdgeData.xml to get best solution
            flag, meanGEH, GEH_percent = result_analysis_on_EdgeData(
                self.df_summary,
                self.scenario_config["calibration_target"],
                self.scenario_config["sim_start_time"],
                self.scenario_config["sim_end_time"],
                self.path_edge_abs)
            print(f"  :In final results, {int(GEH_percent * 10000) / 100} percent GEH is lower than 5.")

            if flag:
                print("  :All traffic volume requirements are met.")
            else:
                print("  :Not all traffic volume requirements are met.")
        except Exception as e:
            print(f"  :Error in generating final results: {e}")

        path_best_solution = pf.path2linux(Path(self.output_dir) / 'GEH_best_solution.txt')
        np.savetxt(path_best_solution, best_solution, fmt='%.4f')

        # copy best rou.xml, flow.xml, turn.xml to output dir
        shutil.copy(temp_rou_best, pf.path2linux(Path(self.output_dir) / f"{network_name}.rou.xml"))
        shutil.copy(temp_flow_best, pf.path2linux(Path(self.output_dir) / f"{network_name}.flow.xml"))
        shutil.copy(temp_turn_best, pf.path2linux(Path(self.output_dir) / f"{network_name}.turn.xml"))
        if remove_old_files:
            # delete temp route folder
            path_temp_route = pf.path2linux(Path(self.output_dir) / 'temp_route')
            shutil.rmtree(path_temp_route)
        return True

    def run_vis(self) -> bool:
        if not hasattr(self, 'GEH_summary'):
            print("No GEH summary found. Please run the calibration first.")
            return

        df = pd.DataFrame(self.GEH_summary).T
        df.columns = np.arange(1, len(self.GEH_summary) + 1).tolist()

        # Plotting the minimum values of each column connected by a red line
        min_values = [df[col].min() for col in df.columns]
        plt.plot(df.columns, min_values, 'o-', color='red', markersize=10, label='Best GEH in each iteration')

        plt.xlabel('Iteration')
        plt.ylabel('Mean GEH')
        plt.title('Mean GEH over iterations using TS')
        # plt.xticks(ticks=df.columns, labels=df.columns)  # Set X-axis ticks to match column numbers
        # plt.legend()

        num_iterations = self.ts_config.get("iterations", 30)
        plt.xticks(range(0, num_iterations + 1, 5))
        plt.ylim(0, 12)
        plt.show()
        return True


if __name__ == "__main__":
    ts_config = {
        "iterations": 3,
        "tabu_size": 120,
        "neighborhood_size": 32,
        "move_range": 0.5,  # Initial move range
        "lower_bound": 0,
        "upper_bound": 1,
        "lbc": 0,  # lower bound for inflow counts
        "ubc": 200,  # upper bound for inflow counts
        "num_turning_ratio": 12,
        "max_no_improvement_local": 5,
        "max_no_improvement_global": 30,
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

    ts = TabuSearch(ts_config, scenario_config, verbose=True)
    ts.run_TS(remove_old_files=False)
    ts.run_vis()
