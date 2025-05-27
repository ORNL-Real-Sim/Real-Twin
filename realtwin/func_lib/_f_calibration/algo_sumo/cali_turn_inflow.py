##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a __TBD__           #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################

import os
import sys
from pathlib import Path
from functools import partial
import shutil
from mealpy import FloatVar, SA, GA, TS
try:
    from realtwin.func_lib._f_calibration.algo_sumo.util_cali_turn_inflow import (
        update_turn_flow_from_solution,
        run_SUMO_create_EdgeData,
        run_jtrrouter_to_create_rou_xml,
        result_analysis_on_EdgeData)
except ImportError:
    from util_cali_turn_inflow import (
        update_turn_flow_from_solution,
        run_SUMO_create_EdgeData,
        run_jtrrouter_to_create_rou_xml,
        result_analysis_on_EdgeData,
        read_MatchupTable,
        generate_turn_demand_cali,
        generate_inflow,
        generate_turn_summary,)


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

    TurnDf_Calibration = scenario_config.get("TurnDf_Calibration")
    TurnToCalibrate = scenario_config.get("TurnToCalibrate")
    InflowDf_Calibration = scenario_config.get("InflowDf_Calibration")
    InflowEdgeToCalibrate = scenario_config.get("InflowEdgeToCalibrate")
    RealSummary_Calibration = scenario_config.get("RealSummary_Calibration")
    calibration_interval = scenario_config.get("calibration_interval", 60)
    demand_interval = scenario_config.get("demand_interval", 15)

    network_name = scenario_config.get("network_name")
    sim_start_time = scenario_config.get("sim_start_time")
    sim_end_time = scenario_config.get("sim_end_time")
    path_net = scenario_config.get("path_net")
    path_rou = pf.path2linux(Path(scenario_config.get("dir_turn_inflow")) / "route" / f"{network_name}.rou.xml")
    sim_name = scenario_config.get("sim_name")
    path_edge = pf.path2linux(Path(scenario_config.get("dir_turn_inflow")) / "EdgeData.xml")
    calibration_target = scenario_config.get("calibration_target")

    # TODO will remove in the future iteration - change current working dir at beginning of the calibration
    os.chdir(scenario_config.get("dir_turn_inflow"))

    # update turn and flow
    df_turn, df_inflow = update_turn_flow_from_solution(solution,
                                                        TurnDf_Calibration,
                                                        TurnToCalibrate,
                                                        InflowDf_Calibration,
                                                        InflowEdgeToCalibrate,
                                                        calibration_interval,
                                                        demand_interval)

    # update rou.xml from updated turn and flow in route and turn_flow folders
    run_jtrrouter_to_create_rou_xml(network_name,
                                    path_net,
                                    df_turn,
                                    df_inflow,
                                    path_rou,
                                    sim_start_time,
                                    sim_end_time)

    # run SUMO to get EdgeData.xml
    run_SUMO_create_EdgeData(sim_name, sim_end_time)

    # analyze EdgeData.xml to get best solution
    _, mean_GEH, GEH_percent = result_analysis_on_EdgeData(RealSummary_Calibration,
                                                           path_edge,
                                                           calibration_target,
                                                           sim_start_time,
                                                           sim_end_time)
    print(f"  :GEH: Mean Percentage: {mean_GEH}, {GEH_percent}")

    # minimize the negative percentage of GEH and the mean GEH
    # return [mean_GEH, -GEH_percent]
    return [mean_GEH]


class TurnInflowCali:
    """ Turn and Inflow Optimization class for SUMO calibration.

    See Also:
        Problem_dict: https://mealpy.readthedocs.io/en/latest/pages/general/simple_guide.html
        termination_dict: https://mealpy.readthedocs.io/en/latest/pages/general/advance_guide.html#stopping-condition-termination

    Args:
        scenario_config (dict): the configuration for the scenario.
        turn_inflow_config (dict): the configuration for the turn and inflow.
        verbose (bool): whether to print the information. Defaults to True.

    Note:
        We use the mealpy library for optimization. mealpy is a Python library for optimization algorithms.
            https://mealpy.readthedocs.io/en/latest/index.html

        1. The `scenario_config` parameter is used and can be modified from the `input_config.yaml` file.

        2. The `turn_inflow_config` parameter is used and can be modified from the `input_config.yaml` file.

    """

    def __init__(self, scenario_config: dict = None, turn_inflow_config: dict = None, verbose: bool = True):
        """Initialize the TurnInflowCalib class with scenario and turn inflow configurations."""

        self.scenario_config = scenario_config
        self.turn_inflow_cfg = turn_inflow_config
        self.verbose = verbose

        # prepare termination criteria from scenario config
        self.term_dict = {
            "max_epoch": self.scenario_config.get("max_epoch", 1000),
            "max_fe": self.scenario_config.get("max_fe", 10000),
            "max_time": self.scenario_config.get("max_time", None),
            "max_early_stop": self.scenario_config.get("max_early_stop", 80),
        }

        # prepare problem dict from algo config
        init_params = self.turn_inflow_cfg.get("initial_params", None)
        if isinstance(init_params, dict):
            self.init_solution = list(init_params.values())
        elif isinstance(init_params, list):
            self.init_solution = init_params
        elif isinstance(init_params, np.ndarray):
            self.init_solution = init_params.tolist()
        else:
            if self.verbose:
                print("  :Info: initial parameters are not provided, using None.")
            self.init_solution = None

        params_ranges = self.turn_inflow_cfg.get("params_ranges", None)
        if isinstance(params_ranges, dict):
            params_lb = [val[0] for val in params_ranges.values()]
            params_ub = [val[1] for val in params_ranges.values()]
        elif isinstance(params_ranges, list):  # list of tuples [(min, max), ...]
            params_lb = [val[0] for val in params_ranges]
            params_ub = [val[1] for val in params_ranges]
        else:
            raise ValueError("  :Error: params_ranges in configuration file must be a dict or list of tuples.")

        self.problem_dict = {
            "obj_func": partial(fitness_func_turn_flow, scenario_config=self.scenario_config),
            "bounds": FloatVar(lb=params_lb, ub=params_ub,),
            "minmax": "min",  # maximize or minimize
            "log_to": "console",
            # "log_to": "file",
            # "log_file": "result.log",
            "save_population": True,              # Default = False
            # "obj_weights": [0.7, 0.3],  # weights for multi-objective optimization
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

        # check if output_dir exists
        os.makedirs(output_dir, exist_ok=True)

        # save the best solution
        try:
            model.history.save_global_objectives_chart(filename=f"{output_dir}/global_objectives")
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

    def run_GA(self, **kwargs) -> tuple:
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
        if (ga_config := self.turn_inflow_cfg.get("ga_config")) is None:
            raise ValueError("  :Error: ga_config is not provided in the configuration file.")

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
        if epoch > self.term_dict["max_epoch"]:
            self.term_dict["max_epoch"] = epoch
        g_best = model_ga.solve(self.problem_dict, termination=self.term_dict, starting_solutions=init_vals)

        # update files with the best solution
        fitness_func_turn_flow(g_best.solution, scenario_config=self.scenario_config)

        return (g_best, model_ga)

    def run_SA(self, **kwargs) -> tuple:
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
            kwargs: additional keyword arguments for specific SA models. Navigate to See Also for more details.
        """
        if (sa_config := self.turn_inflow_cfg.get("sa_config")) is None:
            raise ValueError("  :Error: sa_config is not provided in the configuration file.")

        epoch = sa_config.get("epoch", 1000)  # max iterations
        pop_size = sa_config.get("pop_size", 2)  # population size
        temp_init = sa_config.get("temp_init", 100)  # initial temperature
        cooling_rate = sa_config.get("cooling_rate", 0.891)  # cooling rate
        step_size = sa_config.get("step_size", 0.1)  # step size for the change
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
                                     step_size=step_size,
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
        fitness_func_turn_flow(g_best.solution, scenario_config=self.scenario_config)

        return (g_best, model_sa)

    def run_TS(self, **kwargs) -> tuple:
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
            kwargs: additional keyword arguments for specific TS models. Navigate to See Also for more details.
        """
        if (ts_config := self.turn_inflow_cfg.get("ts_config")) is None:
            raise ValueError("  :Error: ts_config is not provided in the configuration file.")

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
        g_best = model_ts.solve(self.problem_dict, termination=self.term_dict, starting_solutions=init_vals)

        # update files with the best solution
        fitness_func_turn_flow(g_best.solution, scenario_config=self.scenario_config)

        return (g_best, model_ts)

    def _clean_up(self):
        """Clean up the temporary files generated during the calibration process."""
        network_name = self.scenario_config.get("network_name")
        turn_inflow_dir = self.scenario_config.get("dir_turn_inflow")
        route_dir = os.path.join(turn_inflow_dir, "route")
        flow_file = Path(route_dir) / f"{network_name}.flow.xml"
        turn_file = Path(route_dir) / f"{network_name}.turn.xml"
        shutil.copy(flow_file, turn_inflow_dir)
        shutil.copy(turn_file, turn_inflow_dir)
        # remove the route folder
        shutil.rmtree(route_dir)


if __name__ == "__main__":

    # create turn and inflow and summary df
    path_matchup_table = r"C:\Users\xh8\ornl_work\github_workspace\Real-Twin-Dev\datasets\tss\MatchupTable.xlsx"
    traffic_dir = r"C:\Users\xh8\ornl_work\github_workspace\Real-Twin-Dev\datasets\tss\Traffic"
    path_net_turn_inflow = r"C:\Users\xh8\ornl_work\github_workspace\Real-Twin-Dev\datasets\tss\output\SUMO\turn_inflow\chatt.net.xml"
    MatchupTable_UserInput = read_MatchupTable(path_matchup_table=path_matchup_table)
    TurnDf, IDRef = generate_turn_demand_cali(path_matchup_table=path_matchup_table, traffic_dir=traffic_dir)

    InflowDf_Calibration, InflowEdgeToCalibrate, N_InflowVariable = generate_inflow(path_net_turn_inflow,
                                                                                    MatchupTable_UserInput,
                                                                                    TurnDf,
                                                                                    IDRef)

    (TurnToCalibrate, TurnDf_Calibration,
     RealSummary_Calibration,
     N_Variable, N_TurnVariable) = generate_turn_summary(TurnDf,
                                                         MatchupTable_UserInput,
                                                         N_InflowVariable)

    scenario_config = {
        "input_dir": r"C:\Users\xh8\ornl_work\github_workspace\Real-Twin-Dev\datasets\tss\output\SUMO\turn_inflow",
        "network_name": "chatt",
        "sim_name": "chatt.sumocfg",
        "sim_start_time": 28800,
        "sim_end_time": 32400,
        "calibration_target": {'GEH': 5, 'GEHPercent': 0.85},
        "calibration_interval": 60,
        "demand_interval": 15,
    }

    scenario_config["TurnToCalibrate"] = TurnToCalibrate
    scenario_config["TurnDf_Calibration"] = TurnDf_Calibration
    scenario_config["InflowDf_Calibration"] = InflowDf_Calibration
    scenario_config["InflowEdgeToCalibrate"] = InflowEdgeToCalibrate
    scenario_config["RealSummary_Calibration"] = RealSummary_Calibration
    scenario_config["N_InflowVariable"] = N_InflowVariable
    scenario_config["N_Variable"] = N_Variable
    scenario_config["N_TurnVariable"] = N_TurnVariable
    scenario_config["path_net"] = r"C:\Users\xh8\ornl_work\github_workspace\Real-Twin-Dev\datasets\tss\output\SUMO\turn_inflow\chatt.net.xml"

    turn_inflow_config = {"initial_params": [0.5, 0.5, 0.5, 0.5, 0.5,
                                             0.5, 0.5, 0.5, 0.5, 0.5,
                                             0.5, 0.5, 100, 100, 100, 100],
                          "params_ranges": [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                                            [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                                            [0, 1], [0, 1], [50, 200], [50, 200], [50, 200], [50, 200]],
                          "max_epoch": 1000,
                          "max_fe": 10000,
                          "max_time": 3600,
                          "max_early_stop": 20,

                          "ga_config": {"epoch": 10,
                                        "pop_size": 8,
                                        "pc": 0.75,
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
                                        "scale": 0.1,  # the scale in gaussian random
                                        "model_selection": "OriginalSA",  # "OriginalSA", "GaussianSA", "SwarmSA"
                                        },
                          "ts_config": {"epoch": 1000,
                                        "tabu_size": 10,
                                        "neighbour_size": 10,
                                        "perturbation_scale": 0.05,
                                        },
                          }

    opt = TurnInflowCalib(scenario_config=scenario_config, turn_inflow_config=turn_inflow_config, verbose=True)

    # Run Genetic Algorithm
    # g_best = opt.run_GA()

    # Run Simulated Annealing
    g_best = opt.run_SA()

    # Run Tabu Search
    # g_best = opt.run_TS()
