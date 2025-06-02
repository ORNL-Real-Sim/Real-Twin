##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a MIT               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################

import os
import sys
from functools import partial
import random
from pathlib import Path
import subprocess
import pandas as pd
import pyufunc as pf

from mealpy import FloatVar, SA, GA, TS
try:
    from realtwin.func_lib._f_calibration.algo_sumo.util_cali_behavior import (
        get_travel_time_from_EdgeData_xml,
        update_flow_xml_from_solution,
        run_jtrrouter_to_create_rou_xml,
        result_analysis_on_EdgeData,
    )
except ImportError:
    from util_cali_behavior import (
        get_travel_time_from_EdgeData_xml,
        update_flow_xml_from_solution,
        run_jtrrouter_to_create_rou_xml,
        result_analysis_on_EdgeData,
    )
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    sys.path = list(set(sys.path))  # remove duplicates

else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci

rng = np.random.default_rng(seed=812)


def fitness_func(solution: list | np.ndarray, scenario_config: dict = None, error_func: str = "rmse") -> float:
    """ Evaluate the fitness of a given solution for SUMO calibration."""

    # Set up SUMO command with car-following parameters
    if error_func not in ["rmse", "mae"]:
        raise ValueError("error_func must be either 'rmse' or 'mae'")

    if solution[5] >= 9.3:  # emergencyDecel
        solution[5] = 9.3
    if solution[5] < solution[2]:  # emergencyDecel < deceleration
        solution[5] = solution[2] + random.randrange(1, 5)

    # get path from scenario_config
    network_name = scenario_config.get("network_name")
    sel_behavior_routes = scenario_config.get("sel_behavior_routes")
    sim_input_dir = Path(scenario_config.get("dir_behavior"))
    path_net = pf.path2linux(sim_input_dir / f"{network_name}.net.xml")
    path_flow = pf.path2linux(sim_input_dir / f"{network_name}.flow.xml")
    path_turn = pf.path2linux(sim_input_dir / f"{network_name}.turn.xml")
    path_rou = pf.path2linux(sim_input_dir / f"{network_name}.rou.xml")
    path_EdgeData = pf.path2linux(sim_input_dir / "EdgeData.xml")

    sim_name = scenario_config.get("sim_name")
    sim_end_time = scenario_config.get("sim_end_time")

    # change the working directory to the input directory for SUMO
    os.chdir(sim_input_dir)

    update_flow_xml_from_solution(path_flow, solution)

    # Create the route XML file using jtrrouter 
    run_jtrrouter_to_create_rou_xml(network_name, path_net, path_flow, path_turn, path_rou)

    # Define the command to run SUMO
    sumo_command = f"sumo -c \"{sim_name}\""
    sumoProcess = subprocess.Popen(sumo_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sumoProcess.wait()

    # Read output file or TraCI to evaluate the fitness
    travel_time_diff_list = []
    for route in sel_behavior_routes:
        route_time = sel_behavior_routes[route]["time"]
        route_edge_list = sel_behavior_routes[route]["route_list"]
        travel_time = get_travel_time_from_EdgeData_xml(path_EdgeData, route_edge_list)
        travel_time_diff = abs(route_time - travel_time)
        travel_time_diff_list.append(travel_time_diff)

    if error_func == "rmse":
        fitness_err = np.sqrt(sum([x ** 2 for x in travel_time_diff_list]) / len(travel_time_diff_list))
    elif error_func == "mae":
        fitness_err = sum(travel_time_diff_list) / len(travel_time_diff_list)
    else:
        raise ValueError("error_func must be either 'rmse' or 'mae'")

    # Calculate GEH from updated results
    # path_summary = pf.path2linux(sim_input_dir / "summary.xlsx")
    # calibration_target = scenario_config.get("calibration_target")
    # sim_start_time = scenario_config.get("sim_start_time")
    # sim_end_time = scenario_config.get("sim_end_time")
    # _, mean_geh, geh_percent = result_analysis_on_EdgeData(path_summary,
    #                                                        path_EdgeData,
    #                                                        calibration_target,
    #                                                        sim_start_time,
    #                                                        sim_end_time)
    # print(f"  :GEH: Mean Percentage: {mean_geh:.6f}, {geh_percent:.6f}, Travel time error: {fitness_err:.6f}")
    print(f"  :Travel time error: {fitness_err:.6f}")

    return fitness_err


class BehaviorCali:
    """ Behavior Optimization class for SUMO calibration

    Args:
        scenario_config (dict): the configuration for the scenario.
        behavior_config (dict): the configuration for the behavior.
        verbose (bool): whether to print the log. Defaults to True.

    Notes:
        We use the mealpy library for optimization. mealpy is a Python library for optimization algorithms.
            https://mealpy.readthedocs.io/en/latest/index.html

        1. The `init_solution` parameter is used to provide initial solutions for the population. None by default.
        2. Behavior includes: min_gap(meters), acceleration(m/s^2), deceleration(m/s^2), sigma, tau, and emergencyDecel.

    See Also:
        Problem_dict: https://mealpy.readthedocs.io/en/latest/pages/general/simple_guide.html
        termination_dict: https://mealpy.readthedocs.io/en/latest/pages/general/advance_guide.html#stopping-condition-termination

    Examples:
        >>> from realtwin import BehaviorOpt
        >>> prob_dict = {"obj_func": partial(fitness_func, scenario_config=scenario_config, error_func="rmse"),
                        "bounds": FloatVar(lb=[1.0, 2.5, 4, 0.0, 0.25, 5.0], ub=[3.0, 3.0, 5.3, 1.0, 1.25, 9.3],),
                        "minmax": "max",  # maximize or minimize
                        "log_to": "console",
                        "save_population": True}
        >>> init_solution = [2.5, 2.6, 4.5, 0.5, 1.0, 9.0]
        >>> term_dict = {"max_epoch": 500, "max_fe": 10000, "max_time": 3600, "max_early_stop": 20}
        >>> opt = BehaviorOpt(problem_dict=prob_dict, init_solution=init_solution, term_dict=term_dict)
        >>> g_best, model_opt = opt.run_GA(epoch=1000, pop_size=30, pc=0.95, pm=0.1, sel_model="BaseGA")

        Save result figures to output_dir
        >>> opt.run_vis(output_dir="output_dir", model=model_opt)
        >>> print(g_best.solution)
        >>> print(g_best.target.fitness)
    """

    def __init__(self, scenario_config: dict = None, behavior_config: dict = None, verbose: bool = True):

        self.scenario_config = scenario_config
        self.behavior_cfg = behavior_config
        self.verbose = verbose

        # prepare termination criteria from scenario config
        self.term_dict = {
            "max_epoch": self.scenario_config.get("max_epoch", 1000),
            "max_fe": self.scenario_config.get("max_fe", 10000),
            "max_time": self.scenario_config.get("max_time", None),
            "max_early_stop": self.scenario_config.get("max_early_stop", 80),
        }

        init_params = self.behavior_cfg.get("initial_params", None)
        if isinstance(init_params, dict):
            self.init_solution = list(init_params.values())
        elif isinstance(init_params, list):
            self.init_solution = init_params
        elif isinstance(init_params, np.ndarray):
            self.init_solution = init_params.tolist()
        else:
            self.init_solution = None

        params_ranges = self.behavior_cfg.get("params_ranges").values()
        params_lb = [val[0] for val in params_ranges]
        params_ub = [val[1] for val in params_ranges]
        self.problem_dict = {
            "obj_func": partial(fitness_func, scenario_config=self.scenario_config, error_func="rmse"),
            "bounds": FloatVar(lb=params_lb, ub=params_ub,),
            "minmax": "min",  # maximize or minimize
            "log_to": "console",
            # "log_to": "file",
            # "log_file": "result.log",
            "save_population": True,              # Default = False
        }

    def _generate_initial_solutions(self, init_vals: list, pop_size: int) -> np.array:
        """Generate initial solutions for inputs.

        Args:
            init_vals (list | np.array): initial values for the solutions.
            pop_size (int): population size.

        Returns:
            np.array: array of initial solutions.
        """

        # TDD
        if not isinstance(init_vals, (list, np.ndarray)):
            print("Error: init_vals must be a list, numpy array, or None.")
            return None

        if init_vals is not None:
            return np.array(list(init_vals) * pop_size).reshape(pop_size, len(init_vals))
        return None

    def run_vis(self, output_dir: str, model) -> bool:
        """Save the results of the optimization.

        See Also:
            https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.utils.html#module-mealpy.utils.history

        Args:
            output_dir (str): the directory to save the results.
            model: the optimized model object.
        """

        # save the best solution
        try:
            pass
            # model.history.save_global_objectives_chart(filename=f"{output_dir}/global_objectives")
            # model.history.save_local_objectives_chart(filename=f"{output_dir}/local_objectives")
            # model.history.save_global_best_fitness_chart(filename=f"{output_dir}/global_best_fitness")
            # model.history.save_local_best_fitness_chart(filename=f"{output_dir}/local_best_fitness")
            # model.history.save_runtime_chart(filename=f"{output_dir}/runtime")
            # model.history.save_exploration_exploitation_chart(filename=f"{output_dir}/exploration_exploitation")
            # model.history.save_diversity_chart(filename=f"{output_dir}/diversity")
            # model.history.save_trajectory_chart(filename=f"{output_dir}/trajectory")
        except Exception as e:
            print(f"  :Error in saving vis: {e}")
            return False
        return True

    def run_GA(self, **kwargs):
        """Run Genetic Algorithm (GA) for behavior optimization.

        Note:
            1. The `model_selection` parameter allows you to choose different types of GA models. Default is "BaseGA".
                Options include "BaseGA", "EliteSingleGA", "EliteMultiGA", "MultiGA", and "SingleGA".
            2. Additional keyword arguments (`**kwargs`) can be passed for specific GA models (See Also).
            3. Please check original GA model documentation for more kwargs in details: https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.evolutionary_based.html#module-mealpy.evolutionary_based.GA

        Warning:
            You can change the input parameters only from input_config.yaml file.

        See Also:
            https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.evolutionary_based.html#module-mealpy.evolutionary_based.GA

        Args:
            epoch (int): the iterations. Defaults to 1000.
            pop_size (int): population size. Defaults to 50.
            pc (float): crossover probability. Defaults to 0.95.
            pm (float): mutation probability. Defaults to 0.025.
            model_selection (str): the type of GA model to use. Defaults to "BaseGA".
                options: "BaseGA", "EliteSingleGA", "EliteMultiGA", "MultiGA", "SingleGA".
            kwargs: additional keyword arguments for specific GA models (Navigate to See Also).
        """
        if (ga_config := self.behavior_cfg.get("ga_config")) is None:
            raise ValueError("ga_config is not provided in Calibration/turn_inflow setting in input_config.yaml file.")

        epoch = ga_config.get("epoch", 1000)  # max iterations
        pop_size = ga_config.get("pop_size", 50)  # population size
        if pop_size < 10:  # minimum population size for GA in mealpy is 10
            pop_size = 10
        pc = ga_config.get("pc", 0.75)  # crossover probability
        pm = ga_config.get("pm", 0.1)  # mutation probability

        selection = ga_config.get("selection", "roulette")  # selection method
        k_way = ga_config.get("k_way", 0.2)  # k-way for tournament selection
        crossover = ga_config.get("crossover", "uniform")  # crossover method
        mutation = ga_config.get("mutation", "swap")  # mutation method

        # percentage of the best in elite group, or int, the number of best elite
        elite_best = ga_config.get("elite_best", 0.1)

        # percentage of the worst in elite group, or int, the number of worst elite
        elite_worst = ga_config.get("elite_worst", 0.3)

        # "BaseGA", "EliteSingleGA", "EliteMultiGA", "MultiGA", "SingleGA"
        sel_model = ga_config.get("model_selection", "BaseGA")

        # Generate initial solution for inputs
        init_vals = self._generate_initial_solutions(self.init_solution, pop_size)

        if sel_model not in ["BaseGA", "EliteSingleGA", "EliteMultiGA", "MultiGA", "SingleGA"]:
            print("Error: sel_model must be one of the following: "
                  "'BaseGA', 'EliteSingleGA', 'EliteMultiGA', 'MultiGA', 'SingleGA'.")
            print("Defaulting to 'BaseGA'.")
            sel_model = "BaseGA"

        if sel_model == "BaseGA":
            model_ga = GA.BaseGA(epoch=epoch, pop_size=pop_size, pc=pc, pm=pm, **kwargs)
        elif sel_model == "EliteSingleGA":
            model_ga = GA.EliteSingleGA(epoch=epoch, pop_size=pop_size, pc=pc, pm=pm,
                                        selection=selection,
                                        k_way=k_way,
                                        crossover=crossover,
                                        mutation=mutation,
                                        elite_best=elite_best,
                                        elite_worst=elite_worst, **kwargs)
        elif sel_model == "EliteMultiGA":
            model_ga = GA.EliteMultiGA(epoch=epoch, pop_size=pop_size, pc=pc, pm=pm,
                                       selection=selection,
                                       k_way=k_way,
                                       crossover=crossover,
                                       mutation=mutation,
                                       elite_best=elite_best,
                                       elite_worst=elite_worst, **kwargs)
        elif sel_model == "MultiGA":
            model_ga = GA.MultiGA(epoch=epoch, pop_size=pop_size, pc=pc, pm=pm,
                                  selection=selection,
                                  k_way=k_way,
                                  crossover=crossover,
                                  mutation=mutation, **kwargs)
        elif sel_model == "SingleGA":
            model_ga = GA.SingleGA(epoch=epoch, pop_size=pop_size, pc=pc, pm=pm,
                                   selection=selection,
                                   k_way=k_way,
                                   crossover=crossover,
                                   mutation=mutation, **kwargs)

        # solve the problem
        g_best = model_ga.solve(self.problem_dict, termination=self.term_dict, starting_solutions=init_vals)

        # update files with the best solution
        fitness_func(g_best.solution, scenario_config=self.scenario_config, error_func="rmse")

        return (g_best, model_ga)

    def run_SA(self, **kwargs):
        """Run Simulated Annealing (SA) for behavior optimization.

        Note:
            1. The `model_selection` parameter allows you to choose different types of SA models. Default is "OriginalSA".
                Options include "OriginalSA", "GaussianSA", "SwarmSA".

        See Also:
            https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.physics_based.html#module-mealpy.physics_based.SA

        Warning:
            You can change the input parameters only from input_config.yaml file.

        Args:
            epoch (int): iterations. Defaults to 1000.
            pop_size (int): population size. Defaults to 2.
            temp_init (float): initial temperature. Defaults to 100.
            cooling_rate (float): Defaults to 0.99.
            scale (float): the change scale of initialization. Defaults to 0.1.
            model_selection (str): select diff. Defaults to "OriginalSA". Options: "OriginalSA", "GaussianSA", "SwarmSA".
            kwargs: additional keyword arguments for specific SA models (Navigate to See Also).
        """
        if (sa_config := self.behavior_cfg.get("sa_config")) is None:
            raise ValueError("sa_config is not provided in Calibration/turn_inflow setting in yaml file.")

        epoch = sa_config.get("epoch", 1000)  # max iterations
        pop_size = sa_config.get("pop_size", 2)  # population size
        temp_init = sa_config.get("temp_init", 100)  # initial temperature
        cooling_rate = sa_config.get("cooling_rate", 0.891)  # cooling rate
        scale = sa_config.get("scale", 0.1)  # scale of the change
        sel_model = sa_config.get("model_selection", "OriginalSA")  # "OriginalSA", "GaussianSA", "SwarmSA"

        # Generate initial solution for inputs
        init_vals = self._generate_initial_solutions(self.init_solution, pop_size)

        if sel_model not in ["OriginalSA", "GaussianSA", "SwarmSA"]:
            print("Error: sel_model must be one of the following: "
                  "'OriginalSA', 'GaussianSA', 'SwarmSA'.")
            print("Defaulting to 'OriginalSA'.")
            sel_model = "OriginalSA"

        if sel_model == "OriginalSA":
            model_sa = SA.OriginalSA(epoch=epoch,
                                     pop_size=pop_size,
                                     temp_init=temp_init,
                                     cooling_rate=cooling_rate,
                                     step_size=scale,
                                     **kwargs)
        elif sel_model == "GaussianSA":
            model_sa = SA.GaussianSA(epoch=epoch,
                                     pop_size=pop_size,
                                     temp_init=temp_init,
                                     cooling_rate=cooling_rate,
                                     scale=scale,
                                     **kwargs)
        elif sel_model == "SwarmSA":
            model_sa = SA.SwarmSA(epoch=epoch,
                                  pop_size=pop_size,
                                  max_sub_iter=5,
                                  t0=temp_init,
                                  t1=1,
                                  move_count=5,
                                  mutation_rate=0.1,
                                  mutation_step_size=0.1,
                                  mutation_step_size_damp=cooling_rate,
                                  **kwargs)

        g_best = model_sa.solve(self.problem_dict, termination=self.term_dict, starting_solutions=init_vals)

        # update files with the best solution
        fitness_func(g_best.solution, scenario_config=self.scenario_config, error_func="rmse")

        return (g_best, model_sa)

    def run_TS(self, **kwargs):
        """Run Tabu Search (TS) for behavior optimization.

        See Also:
            https://github.com/thieu1995/mealpy/blob/master/mealpy/math_based/TS.py

        Warning:
            You can change the input parameters only from input_config.yaml file.

        Args:
            epoch (int): max iterations. Defaults to 1000.
            pop_size (int): population size. Defaults to 2.
            tabu_size (int): maximum size of tabu list. Defaults to 10.
            neighbour_size (int): size of the neighborhood for generating candidate solutions. Defaults to 10.
            perturbation_scale (float): scale of perturbation for generating candidate solutions. Defaults to 0.05.
            kwargs: additional keyword arguments for specific TS models (Navigate to See Also).
        """
        if (ts_config := self.behavior_cfg.get("ts_config")) is None:
            raise ValueError("ts_config is not provided in Calibration/turn_inflow setting in yaml file.")

        epoch = ts_config.get("epoch", 1000)  # max iterations
        pop_size = ts_config.get("pop_size", 2)  # population size
        tabu_size = ts_config.get("tabu_size", 10)  # maximum size of tabu list

        # size of the neighborhood for generating candidate solutions
        neighbour_size = ts_config.get("neighbour_size", 10)

        # scale of perturbation for generating candidate solutions
        perturbation_scale = ts_config.get("perturbation_scale", 0.05)

        # Generate initial solution for inputs
        init_vals = self._generate_initial_solutions(self.init_solution, pop_size)

        model_ts = TS.OriginalTS(epoch=epoch,
                                 pop_size=pop_size,
                                 tabu_size=tabu_size,
                                 neighbour_size=neighbour_size,
                                 perturbation_scale=perturbation_scale,
                                 **kwargs)
        # not print out log to console
        self.problem_dict["log_to"] = "None"
        g_best = model_ts.solve(self.problem_dict, termination=self.term_dict, starting_solutions=init_vals)

        # update files with the best solution
        fitness_func(g_best.solution, scenario_config=self.scenario_config, error_func="rmse")

        return (g_best, model_ts)


if __name__ == "__main__":

    scenario_config = {
        "input_dir": r"C:\Users\xh8\ornl_work\github_workspace\Real-Twin-Dev\realtwin\func_lib\_f_calibration\algo_sumo_\input_dir_dummy",
        "network_name": "chatt",
        "sim_name": "chatt.sumocfg",
        "sim_start_time": 28800,
        "sim_end_time": 32400,
        "path_turn": "chatt.turn.xml",
        "path_inflow": "chatt.flow.xml",
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

    behavior_config = {"initial_params": {"min_gap": 2.5,        # minimum gap in meters
                                          "acceleration": 2.6,  # max acceleration in m/s^2
                                          "deceleration": 4.5,  # max deceleration in m/s^2
                                          "sigma": 0.5,          # driver imperfection
                                          "tau": 1.00,            # desired headway
                                          "emergencyDecel": 9.0},   # emergency deceleration
                       "params_ranges": {"min_gap": (1.0, 3.0),
                                         "acceleration": (2.5, 3),
                                         "deceleration": (4, 5.3),
                                         "sigma": (0, 1),
                                         "tau": (0.25, 1.25),
                                         "emergencyDecel": (5.0, 9.3)},
                       "max_epoch": 500,
                       "max_fe": 10000,
                       "max_time": 3600,
                       "max_early_stop": 20,

                       "ga_config": {"epoch": 1000,
                                     "pop_size": 30,
                                     "pc": 0.95,
                                     "pm": 0.1,
                                     "selection": "roulette",
                                     "k_way": 0.2,
                                     "crossover": "uniform",
                                     "mutation": "swap",
                                     "elite_best": 0.1,
                                     "elite_worst": 0.3,
                                     "model_selection": "BaseGA",
                                     },
                       "sa_config": {"epoch": 1000,
                                     "temp_init": 100,
                                     "cooling_rate": 0.891,
                                     "scale": 0.1,
                                     "model_selection": "OriginalSA",
                                     },
                       "ts_config": {"epoch": 1000,
                                     "tabu_size": 10,
                                     "neighbour_size": 10,
                                     "perturbation_scale": 0.05,
                                     },
                       }

    opt = BehaviorCali(scenario_config=scenario_config,
                       behavior_config=behavior_config, verbose=True)

    # Run Genetic Algorithm
    # g_best = opt.run_GA()

    # Run Simulated Annealing
    # g_best = opt.run_SA()

    # Run Tabu Search
    g_best = opt.run_TS()
