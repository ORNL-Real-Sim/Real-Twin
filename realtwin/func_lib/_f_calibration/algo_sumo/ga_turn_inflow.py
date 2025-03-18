import os
import sys
import time
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyufunc as pf

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    sys.path = list(set(sys.path))  # remove duplicates
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

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


class GeneticAlgorithmForTurnFlow:
    """Genetic Algorithm for optimization for running simulators"""

    def __init__(self, scenario_config: dict, turn_inflow_config: dict, verbose: bool = True):
        """Input parameters are dictionaries containing configurations for Genetic Algorithm and scenario results.

        Args:
            ga_config: dict
                ga_config = {
                    num_variables: 16,
                    num_turning_ratio: 12,  # remaining should be inflow
                    ubc: 200,
                    population_size: 50,  # must be even
                    num_generations: 30,
                    crossover_rate: 0.75,
                    mutation_rate: 0.1,
                    elitism_size: 1,  # Number of elite individuals to carry over
                    best_fitness_value: float('inf'),
                    max_no_improvement: 5  # Stop if no improvement in 5 iterations}
            scenario_config: dict
                scenario_config = {
                    input_dir: '',
                    output_dir: '',
                    path_turn: 'turn.xlsx',
                    path_inflow: 'inflow.xlsx',
                    path_summary: 'summary.xlsx',
                    path_edge: 'EdgeData.xml',
                    network_name: "chatt",
                    sim_start_time: 28800,
                    sim_end_time: 32400,
                    lower_bound: 0,
                    upper_bound: 1,
                    sim_name: "chatt.sumocfg",
                    calibration_target: {'GEH': 5, 'GEHPercent': 0.85},}
        """
        self.turn_inflow_cfg = turn_inflow_config
        self.scenario_config = scenario_config
        self.verbose = verbose

        # get input dir and output dir from the scenario config
        self.input_dir = pf.path2linux(os.path.abspath(scenario_config.get("input_dir", os.getcwd())))
        self.output_dir = pf.path2linux(os.path.abspath(os.path.join(self.input_dir, "genetic_algorithm_result")))
        os.makedirs(self.output_dir, exist_ok=True)

        # change the current working directory to the input dir
        # this will allow sumocfg and net, rou files to be found by SUMO using Traci
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

    @pf.func_time
    def run_calibration(self, *, remove_old_files: bool = True) -> bool:
        """ Run the Genetic Algorithm for finding the best solution for the given scenario. """

        if self.verbose:
            print("\n  :Running Genetic Algorithm...")

        if (ga_cfg := self.turn_inflow_cfg.get("ga_config")) is None:
            raise ValueError("  :Please provide a valid ga_config in the input configuration file.")

        # get parameters from the config
        population_size = ga_cfg.get("population_size")
        num_variables = ga_cfg.get("num_variables")
        num_generations = ga_cfg.get("num_generations")
        num_turning_ratio = ga_cfg.get("num_turning_ratio")
        ubc = ga_cfg.get("ubc")
        crossover_rate = ga_cfg.get("crossover_rate", 0.75)
        mutation_rate = ga_cfg.get("mutation_rate", 0.1)
        elitism_size = ga_cfg.get("elitism_size", 1)  # Number of elite individuals to carry over
        best_fitness_value = ga_cfg.get("best_fitness_value", float('inf'))
        max_no_improvement = ga_cfg.get("max_no_improvement", 5)  # Stop if no improvement in 5 iterations

        network_name = self.scenario_config.get("network_name")

        # crate the object level instance
        self.minGEH_set = []

        # Initialize population
        rng = np.random.default_rng(seed=812)
        population = rng.random((population_size, num_variables))
        # iterations_without_improvement = 0
        # Evolution loop
        calibration_start_time = time.time()
        for generation in range(num_generations):
            print(f'  :Calibrate generation {generation}')
            # Evaluate fitness
            fitness = np.zeros(population_size)
            for i in range(population_size):
                ical = f'{generation}_{i}'

                # prepare initial solution for each iteration
                ini_solution = population[i].copy()
                ini_solution[num_turning_ratio:] = ini_solution[num_turning_ratio:] * ubc
                fitness[i] = self.run_single_calibration(ini_solution, ical, remove_old_files=remove_old_files)[1]

            # Check for improvement
            current_best_fitness = np.min(fitness)
            self.minGEH_set.append(current_best_fitness)
            print(f'  :minimum mean GEH in this iteration is {current_best_fitness}.')

            if current_best_fitness < best_fitness_value:
                best_fitness_value = current_best_fitness
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            # Check if stopping criteria met
            if iterations_without_improvement >= max_no_improvement:
                print(f"No improvement in the last {max_no_improvement} iterations. Stopping early.")
                break

            # Elitism
            elite_indices = np.argsort(fitness)[:elitism_size]
            elite_individuals = population[elite_indices]

            # Selection (tournament)
            max_fitness = np.max(fitness)
            inverted_fitness = max_fitness + 1 - fitness
            selected_indices = rng.choice(range(population_size),
                                          size=population_size - elitism_size,
                                          replace=True,
                                          p=inverted_fitness / np.sum(inverted_fitness))
            selected_population = population[selected_indices]

            # Crossover
            offspring = []
            num_offspring = population_size - elitism_size
            for i in range(0, num_offspring, 2):
                if rng.random() < crossover_rate and i + 1 < num_offspring:
                    crossover_point = rng.integers(1, num_variables)
                    offspring1 = np.concatenate((selected_population[i][:crossover_point],
                                                 selected_population[i + 1][crossover_point:]))
                    offspring2 = np.concatenate((selected_population[i + 1][:crossover_point],
                                                 selected_population[i][crossover_point:]))
                    offspring.append(offspring1)
                    offspring.append(offspring2)
                else:
                    offspring.append(selected_population[i])
                    if i + 1 < num_offspring:
                        offspring.append(selected_population[i + 1])
            offspring = np.array(offspring)

            # Mutation
            mutation_indices = rng.random((population_size - elitism_size, num_variables)) < mutation_rate
            offspring[mutation_indices] = rng.random(np.sum(mutation_indices))

            # Update population
            if generation < num_generations - 1:  # avoid change population after final iteration
                population = np.vstack((elite_individuals, offspring))
            print(f"  :Current calibration time is {time.time() - calibration_start_time} sec.")
            # print(population)

        # Find the best solution
        best_fitness_index = np.argmin(fitness)
        best_solution = population[best_fitness_index]
        print("  :Best solution:", best_solution)
        best_solution[num_turning_ratio:] = best_solution[num_turning_ratio:] * ubc
        ical = 'final'

        # set the turn and flow df to the original df
        self.df_turn = pd.read_excel(pf.path2linux(Path(self.input_dir) / self.scenario_config.get("path_turn")))
        self.df_inflow = pd.read_excel(pf.path2linux(Path(self.input_dir) / self.scenario_config.get("path_inflow")))
        best_flag, best_value, best_percent = self.run_single_calibration(best_solution,
                                                                          ical,
                                                                          remove_old_files=False)
        print("  :Mean GEH:", best_value)
        print(f"  :In final results, {int(best_percent * 10000) / 100} percent GEH is lower than 5.")
        if best_flag:
            print("  :All traffic volume requirements are met.")
        else:
            print("  :Not all traffic volume requirements are met.")
        best_solution_final = best_solution
        best_solution_final[num_turning_ratio:] = best_solution_final[num_turning_ratio:] * ubc

        path_best_solution = pf.path2linux(Path(self.output_dir) / 'GEH_best_solution.txt')
        np.savetxt(path_best_solution, best_solution_final, fmt='%.4f')

        # copy the temp files to the input dir for future calibration\
        path_temp_route = pf.path2linux(Path(self.output_dir) / 'temp_route')
        temp_rou = pf.path2linux(Path(path_temp_route) / f"{network_name}{ical}.rou.xml")
        temp_flow = pf.path2linux(Path(path_temp_route) / f"{network_name}{ical}.flow.xml")
        temp_turn = pf.path2linux(Path(path_temp_route) / f"{network_name}{ical}.turn.xml")

        shutil.copy(temp_rou, pf.path2linux(Path(self.input_dir) / f"{network_name}.rou.xml"))
        shutil.copy(temp_flow, pf.path2linux(Path(self.input_dir) / f"{network_name}.flow.xml"))
        shutil.copy(temp_turn, pf.path2linux(Path(self.input_dir) / f"{network_name}.turn.xml"))
        shutil.copy(temp_rou, pf.path2linux(Path(self.output_dir) / f"{network_name}.rou.xml"))
        shutil.copy(temp_flow, pf.path2linux(Path(self.output_dir) / f"{network_name}.flow.xml"))
        shutil.copy(temp_turn, pf.path2linux(Path(self.output_dir) / f"{network_name}.turn.xml"))

        if remove_old_files:
            # delete the temp route folder
            path_temp_route = pf.path2linux(os.path.join(self.input_dir, 'genetic_algorithm_result/temp_route'))
            shutil.rmtree(path_temp_route)

        print("  :Genetic Algorithm finished.")
        return True

    def run_vis(self):
        """Visualize the results of the Genetic Algorithm."""

        if not hasattr(self, 'minGEH_set'):
            raise AttributeError("Please run the GA first.")

        # plot the min GEH set
        plt.plot(self.minGEH_set, marker='o', color="red")
        plt.xlabel("Iteration")
        plt.ylabel("Mean GEH")
        plt.title("Mean GEH over Iterations using Genetic Algorithm")
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
                      "accept_prob": 0.5,
                      "initial_temperature": 100,
                      "cooling_rate": 0.99,
                      "stopping_temperature": 1e-3,
                      "max_iteration": 3,
                      "lower_bound": 0,
                      "upper_bound": 1, },
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
        "input_dir": r"C:\Users\xh8\ornl_work\github_workspace\Real-Twin-Dev\datasets\input_dir_dummy",
        "network_name": "chatt",
        "sim_name": "chatt.sumocfg",
        "sim_start_time": 28800,
        "sim_end_time": 32400,
        "path_turn": "turn.xlsx",
        "path_inflow": "inflow.xlsx",
        "path_summary": "summary.xlsx",
        "path_edge": 'EdgeData.xml',
        "calibration_target": {'GEH': 5, 'GEHPercent': 0.85},  # GEH:
        "calibration_interval": 60,
        "demand_interval": 15,
        # "lower_bound": 0,   # For Tabu search only
        # "upper_bound": 1,
    }

    ga = GeneticAlgorithmForTurnFlow(scenario_config, turn_inflow_config)
    ga.run_calibration(remove_old_files=True)
    ga.run_vis()
