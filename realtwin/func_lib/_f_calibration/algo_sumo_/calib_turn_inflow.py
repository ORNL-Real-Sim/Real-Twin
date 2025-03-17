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

from mealpy import FloatVar, SA, GA, TS
from util_cali_behavior import (get_travel_time_from_EdgeData_xml,
                                update_flow_xml_from_solution,
                                run_jtrrouter_to_create_rou_xml,
                                result_analysis_on_EdgeData,
                                run_SUMO_create_EdgeData,
                                update_turn_flow_from_solution,
                                create_rou_turn_flow_xml)
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


def fitness_func_turn_flow(solution: list | np.ndarray, scenario_config: dict = None, **kwargs) -> float:
    """ Objective function for SUMO calibration."""
    """ Run a single calibration iteration to get the best solution """

    path_turn = pf.path2linux(Path(scenario_config.get("input_dir")) / scenario_config.get("path_turn"))
    path_inflow = pf.path2linux(Path(scenario_config.get("input_dir")) / scenario_config.get("path_inflow"))
    path_summary = pf.path2linux(Path(scenario_config.get("input_dir")) / scenario_config.get("path_summary"))
    path_edge = pf.path2linux(Path(scenario_config.get("input_dir")) / scenario_config.get("path_edge"))

    # TODO will remove in the future iteration
    os.chdir(scenario_config.get("input_dir"))

    # update turn and flow
    df_turn, df_inflow = update_turn_flow_from_solution(path_turn,
                                                        path_inflow,
                                                        solution,
                                                        scenario_config["calibration_interval"],
                                                        scenario_config["demand_interval"])

    # update rou.xml from updated turn and flow
    create_rou_turn_flow_xml(scenario_config.get("network_name"),
                             scenario_config.get("sim_start_time"),
                             scenario_config.get("sim_end_time"),
                             df_turn,
                             df_inflow,
                             scenario_config.get("input_dir"))

    # run SUMO to get EdgeData.xml
    run_SUMO_create_EdgeData(scenario_config.get("sim_name"),
                             scenario_config.get("sim_end_time"))

    # analyze EdgeData.xml to get best solution
    best_flag, mean_GEH, GEH_percent = result_analysis_on_EdgeData(path_summary,
                                                                   path_edge,
                                                                   scenario_config["calibration_target"],
                                                                   scenario_config["sim_start_time"],
                                                                   scenario_config["sim_end_time"])
    print(f"  :GEH: Mean Percentage: {mean_GEH}, {GEH_percent}")
    return mean_GEH


class TurnInflowCalib:
    """ Behavior Optimization class for SUMO calibration.

    See Also:
        Problem_dict: https://mealpy.readthedocs.io/en/latest/pages/general/simple_guide.html
        termination_dict: https://mealpy.readthedocs.io/en/latest/pages/general/advance_guide.html#stopping-condition-termination

    Args:
        problem_dict (dict): dictionary containing the problem definition.
            e.g., problem_dict = {"obj_func": partial(fitness_func, scenario_config=scenario_config, error_func="rmse"),
                                    "bounds": FloatVar(lb=[1.0, 2.5, 4, 0.0, 0.25, 5.0], ub=[3.0, 3.0, 5.3, 1.0, 1.25, 9.3],),
                                    "minmax": "max",  # maximize or minimize
                                    "log_to": "console",
                                    or
                                    "log_to": "file",
                                    "log_file": "result.log",
                                    "save_population": True,              # Default = False}
        init_solution (list): initial solution for the optimization.
            e.g. init_solution = [2.5, 2.6, 4.5, 0.5, 1.0, 9.0]
        term_dict (dict): dictionary containing the termination criteria.
            e.g., term_dict = {"max_epoch": 500,  # max iterations
                               "max_fe": 10000,  # max function evaluations
                               "max_time": 3600,  # max time in seconds
                               "max_early_stop": 20,}

    """

    def __init__(self, scenario_config: dict = None, algo_config: dict = None, verbose: bool = True):

        self.scenario_config = scenario_config
        self.algo_config = algo_config
        self.verbose = verbose

        # prepare termination criteria from scenario config
        self.term_dict = {
            "max_epoch": self.algo_config.get("max_epoch", 1000),
            "max_fe": self.algo_config.get("max_fe", 10000),
            "max_time": self.algo_config.get("max_time", 3600),
            "max_early_stop": self.algo_config.get("max_early_stop", 20),
        }

        # prepare problem dict from algo config
        init_params = self.algo_config.get("initial_params", None)
        if isinstance(init_params, dict):
            self.init_solution = init_params.values()
        elif isinstance(init_params, list):
            self.init_solution = init_params
        elif isinstance(init_params, np.ndarray):
            self.init_solution = init_params.tolist()
        else:
            if self.verbose:
                print("  :Info: initial parameters are not provided, using None.")
            self.init_solution = None

        params_ranges = self.algo_config.get("params_ranges", None)
        if isinstance(params_ranges, dict):
            params_lb = [val[0] for val in params_ranges.values()]
            params_ub = [val[1] for val in params_ranges.values()]
        elif isinstance(params_ranges, list):  # list of tuples [(min, max), ...]
            params_lb = [val[0] for val in params_ranges]
            params_ub = [val[1] for val in params_ranges]
        else:
            raise ValueError("  :Error: params_ranges in configuration file must be a dict or list of tuples.")

        self.problem_dict = {
            "obj_func": partial(fitness_func_turn_flow, scenario_config=scenario_config),
            "bounds": FloatVar(lb=params_lb, ub=params_ub,),
            "minmax": "max",  # maximize or minimize
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
        if not isinstance(init_vals, (list, np.ndarray, type(None))):
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
        model.history.save_global_objectives_chart(filename=Path(output_dir) / "global_objectives")
        model.history.save_local_objectives_chart(filename=Path(output_dir) / "local_objectives")
        model.history.save_global_best_fitness_chart(filename=Path(output_dir) / "global_best_fitness")
        model.history.save_local_best_fitness_chart(filename=Path(output_dir) / "local_best_fitness")
        model.history.save_runtime_chart(filename=Path(output_dir) / "runtime")
        model.history.save_exploration_exploitation_chart(filename=Path(output_dir) / "exploration_exploitation")
        model.history.save_diversity_chart(filename=Path(output_dir) / "diversity")
        model.history.save_trajectory_chart(filename=Path(output_dir) / "trajectory")
        return True

    def run_GA(self, **kwargs):
        """Run Genetic Algorithm (GA) for behavior optimization.

        Note:
            1. The `init_solution` parameter is used to provide initial solutions for the population. None by default.
            2. The `ga_model` parameter allows you to choose different types of GA models. Default is "BaseGA". Options include "BaseGA", "EliteSingleGA", "EliteMultiGA", "MultiGA", and "SingleGA".
            3. Additional keyword arguments (`**kwargs`) can be passed for specific GA models.
            4. Please check original GA model documentation for more kwargs in details: https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.evolutionary_based.html#module-mealpy.evolutionary_based.GA

        See Also:
            https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.evolutionary_based.html#module-mealpy.evolutionary_based.GA

        Warning:
            You can change the input parameters only from input_config.yaml file.

        Args:
            epoch (int): the iterations. Defaults to 1000.
            pop_size (int): population size. Defaults to 50.
            pc (float): crossover probability. Defaults to 0.95.
            pm (float): mutation probability. Defaults to 0.025.
            ga_model (str): the type of GA model to use. Defaults to "BaseGA".
                options: "BaseGA", "EliteSingleGA", "EliteMultiGA", "MultiGA", "SingleGA".
            **kwargs: additional keyword arguments for specific GA models.
        """

        epoch = self.algo_config.get("epoch", 1000)  # max iterations
        pop_size = self.algo_config.get("pop_size", 50)  # population size
        pc = self.algo_config.get("pc", 0.75)  # crossover probability
        pm = self.algo_config.get("pm", 0.1)  # mutation probability

        selection = self.algo_config.get("selection", "roulette")  # selection method
        k_way = self.algo_config.get("k_way", 0.2)  # k-way for tournament selection
        crossover = self.algo_config.get("crossover", "uniform")  # crossover method
        mutation = self.algo_config.get("mutation", "swap")  # mutation method

        # percentage of the best in elite group, or int, the number of best elite
        elite_best = self.algo_config.get("elite_best", 0.1)

        # percentage of the worst in elite group, or int, the number of worst elite
        elite_worst = self.algo_config.get("elite_worst", 0.3)

        # "BaseGA", "EliteSingleGA", "EliteMultiGA", "MultiGA", "SingleGA"
        sel_model = self.algo_config.get("model_selection", "BaseGA")

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
        fitness_func_turn_flow(g_best.solution, scenario_config=scenario_config)

        return (g_best, model_ga)

    def run_SA(self, **kwargs):
        """Run Simulated Annealing (SA) for behavior optimization.

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
            sel_model (str): select diff. Defaults to "OriginalSA".
        """

        epoch = self.algo_config.get("epoch", 1000)  # max iterations
        pop_size = self.algo_config.get("pop_size", 2)  # population size
        temp_init = self.algo_config.get("temp_init", 100)  # initial temperature
        cooling_rate = self.algo_config.get("cooling_rate", 0.891)  # cooling rate
        scale = self.algo_config.get("scale", 0.1)  # scale of the change
        sel_model = self.algo_config.get("model_selection", "OriginalSA")  # "OriginalSA", "GaussianSA", "SwarmSA"

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
                                     scale=scale,
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
        fitness_func_turn_flow(g_best.solution, scenario_config=scenario_config)

        return (g_best, model_sa)

    def run_TS(self, *,
               epoch: int = 1000,  # max iterations
               pop_size: int = 2,  # This parameter has no effect on the TS algorithm, only for compatibility
               tabu_size: int = 10,  # maximum size of tabu list
               neighbour_size: int = 10,  # size of the neighborhood for generating candidate solutions
               perturbation_scale: float = 0.05,  # scale of perturbation for generating candidate solutions
               **kwargs):
        """Run Tabu Search (TS) for behavior optimization.

        See Also:
            https://github.com/thieu1995/mealpy/blob/master/mealpy/math_based/TS.py

        Args:
            epoch (int): max iterations. Defaults to 1000.
            pop_size (int): population size. Defaults to 2.
            tabu_size (int): maximum size of tabu list. Defaults to 10.
            neighbour_size (int): size of the neighborhood for generating candidate solutions. Defaults to 10.
            perturbation_scale (float): scale of perturbation for generating candidate solutions. Defaults to 0.05.

        """

        epoch = self.algo_config.get("epoch", 1000)  # max iterations
        pop_size = self.algo_config.get("pop_size", 2)  # population size
        tabu_size = self.algo_config.get("tabu_size", 10)  # maximum size of tabu list

        # size of the neighborhood for generating candidate solutions
        neighbour_size = self.algo_config.get("neighbour_size", 10)

        # scale of perturbation for generating candidate solutions
        perturbation_scale = self.algo_config.get("perturbation_scale", 0.05)

        # Generate initial solution for inputs
        init_vals = self._generate_initial_solutions(self.init_solution, pop_size)

        model_ts = TS.OriginalTS(epoch=epoch,
                                 pop_size=pop_size,
                                 tabu_size=tabu_size,
                                 neighbour_size=neighbour_size,
                                 perturbation_scale=perturbation_scale,
                                 **kwargs)
        g_best = model_ts.solve(self.problem_dict, termination=self.term_dict, starting_solutions=init_vals)

        # update files with the best solution
        fitness_func_turn_flow(g_best.solution, scenario_config=scenario_config)

        return (g_best, model_ts)


if __name__ == "__main__":

    scenario_config = {
        "input_dir": r"C:\Users\xh8\ornl_work\github_workspace\Real-Twin-Dev\datasets\chattanooga",
        "network_name": "chatt",
        "sim_name": "chatt.sumocfg",
        "sim_start_time": 28800,
        "sim_end_time": 32400,
        "path_turn": "turn.xlsx",
        "path_inflow": "inflow.xlsx",
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

    ga_config = {"initial_parameters": {"min_gap": 2.5,        # minimum gap in meters
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
                 "num_generation": 50,

                 "EB_tt": 240,
                 "WB_tt": 180,
                 "EB_edge_list": ["-312", "-293", "-297", "-288", "-286",
                                  "-302", "-3221", "-322", "-313", "-284",
                                  "-328", "-304"],
                 "WB_edge_list": ["-2801", "-280", "-307", "-327", "-281",
                                  "-315", "-321", "-300", "-2851", "-285",
                                  "-290", "-298", "-295"],
                 "epoch": 1000,
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
                 "max_epoch": 500,
                 "max_fe": 10000,
                 "max_time": 3600,
                 "max_early_stop": 20,
                 }

    problem_dict = {
        "obj_func": partial(fitness_func_turn_flow, scenario_config=scenario_config),
        "bounds": FloatVar(lb=[0] * 12 + [50] * 4, ub=[1] * 12 + [200] * 4),
        "minmax": "min",  # maximize or minimize
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

    init_vals = [0.5] * 12 + [100] * 4  # initial values for the optimization

    opt = TurnInflowCalib(scenario_config=scenario_config, algo_config=ga_config, verbose=True)

    # Run Genetic Algorithm
    g_best = opt.run_GA()

    # Run Simulated Annealing
    # g_best = opt.run_SA(epoch=1000, pop_size=2, temp_init=100, cooling_rate=0.98, scale=0.1, sel_model="OriginalSA")

    # Run Tabu Search
    # g_best = opt.run_TS(epoch=1000, pop_size=2, tabu_size=10, neighbour_size=10, perturbation_scale=0.1)
