##############################################################################
# Copyright (c) 2024-, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################

""" Turn and Inflow Calibration Algorithm for SUMO """

import os
import sys
from pathlib import Path
from functools import partial
import shutil
import warnings
from mealpy import FloatVar, GA, SA, TS

try:
    from realtwin.func_lib._f_calibration.algo_sumo.util_cali_turn_inflow import (
        update_turn_inflow_from_solution,
        run_SUMO_create_EdgeData,
        run_jtrrouter_to_create_rou_xml,
        result_analysis_on_EdgeData,
        read_MatchupTable,
        generate_turn_demand_cali,
        generate_inflow,
        generate_turn_summary,)
except ImportError:
    from util_cali_turn_inflow import (
        update_turn_inflow_from_solution,
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
    warnings.warn("Environment variable 'SUMO_HOME' is not set. "
                  "please declare environment variable 'SUMO_HOME'")
    # sys.exit("please declare environment variable 'SUMO_HOME'")

rng = np.random.default_rng(seed=812)


def fitness_func_turn_inflow(solution: list | np.ndarray, scenario_config: dict | None = None, **kwargs) -> float:
    """ Objective function for SUMO calibration, Run a single calibration iteration to get the best solution

    Args:
        solution (list | np.ndarray): the solution to evaluate.
        scenario_config (dict): the configuration for the scenario. Defaults to None.
        **kwargs: additional keyword arguments.
    Returns:
        float: the fitness value.
    """

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
    solution = np.asarray(solution)
    solution = np.clip(solution, 0, None)

    # update turn and flow
    df_turn, df_inflow = update_turn_inflow_from_solution(solution,
                                                        TurnDf_Calibration,
                                                        TurnToCalibrate,
                                                        InflowDf_Calibration,
                                                        InflowEdgeToCalibrate,
                                                        calibration_interval,
                                                        demand_interval)

    # update rou.xml from updated turn and flow in route and turn_inflow folders
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
    print(f"  GEH: {mean_GEH}, Percentage of GEH<5: {GEH_percent*100:.2f}%")

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

    def __init__(self, scenario_config: dict | None = None, turn_inflow_config: dict | None = None, verbose: bool = True, **kwargs):
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
        # init_params = self.turn_inflow_cfg.get("initial_params", None)
        # if isinstance(init_params, dict):
        #     self.init_solution = list(init_params.values())
        # elif isinstance(init_params, list):
        #     self.init_solution = init_params
        # elif isinstance(init_params, np.ndarray):
        #     self.init_solution = init_params.tolist()
        # else:
        #     if self.verbose:
        #         print("  :Info: initial parameters are not provided, using None.")
        #     self.init_solution = None
        self.init_solution = None

        # params_ranges = self.turn_inflow_cfg.get("params_ranges", None)
        # if isinstance(params_ranges, dict):
        #     params_lb = [val[0] for val in params_ranges.values()]
        #     params_ub = [val[1] for val in params_ranges.values()]
        # elif isinstance(params_ranges, list):  # list of tuples [(min, max), ...]
        #     params_lb = [val[0] for val in params_ranges]
        #     params_ub = [val[1] for val in params_ranges]
        # else:
        #     raise ValueError("  :Error: params_ranges in configuration file must be a dict or list of tuples.")

        n_variable = self.scenario_config.get("N_Variable")
        n_inflow_variable = self.scenario_config.get("N_InflowVariable")
        n_turn_variable = self.scenario_config.get("N_TurnVariable")
        max_inflow = self.scenario_config.get("max_inflow", 200)  # max inflow for the inflow variables

        # fitness function for the optimization, default is fitness_func_turn_inflow
        self.fitness_func = fitness_func_turn_inflow if kwargs.get("fitness_func") is None else kwargs.get("fitness_func")

        self.problem_dict = {
            "obj_func": partial(self.fitness_func, scenario_config=self.scenario_config),
            "bounds": FloatVar(lb=[0] * n_variable, ub=[1] * n_turn_variable + [max_inflow] * n_inflow_variable),
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
        """Export BO results or the Mealpy global-objectives chart."""
        os.makedirs(output_dir, exist_ok=True)
        if callable(getattr(model, "run_vis", None)):
            return model.run_vis(output_dir=output_dir)

        try:
            model.history.save_global_objectives_chart(
                filename=f"{output_dir}/global_objectives"
            )
        except Exception as exc:
            # Preserve optional plotting across Mealpy backends and versions.
            print(f"  :Error in saving vis: {exc}")
            return False
        return True

    def run_GA(self, **kwargs) -> tuple:
        """Run GA and apply the best solution to this simulator's inputs.

        Configure BaseGA, EliteSingleGA, EliteMultiGA, MultiGA, or SingleGA
        through ga_config.model_selection. Extra keywords go to the Mealpy
        constructor. Adapter-specific initialization and termination stay here.

        Returns:
            The best agent and fitted optimizer.
        """
        if (ga_config := self.turn_inflow_cfg.get("ga_config")) is None:
            raise ValueError(
                "  :Error: ga_config is not provided in the configuration file."
            )

        parameters = {
            "epoch": ga_config.get("epoch", 1000),
            "pop_size": max(ga_config.get("pop_size", 50), 10),
            "pc": ga_config.get("pc", 0.75),
            "pm": ga_config.get("pm", 0.1),
        }
        model_types = {
            "BaseGA": GA.BaseGA,
            "EliteSingleGA": GA.EliteSingleGA,
            "EliteMultiGA": GA.EliteMultiGA,
            "MultiGA": GA.MultiGA,
            "SingleGA": GA.SingleGA,
        }
        model_name = ga_config.get("model_selection", "BaseGA")
        # Sequence membership also accepts malformed, unhashable selectors.
        if model_name not in tuple(model_types):
            print(
                "Error: sel_model must be one of the following: "
                "'BaseGA', 'EliteSingleGA', 'EliteMultiGA', 'MultiGA', 'SingleGA'."
            )
            print("Defaulting to 'BaseGA'.")
            model_name = "BaseGA"

        if model_name != "BaseGA":
            parameters.update(
                selection=ga_config.get("selection", "roulette"),
                k_way=ga_config.get("k_way", 0.2),
                crossover=ga_config.get("crossover", "uniform"),
                mutation=ga_config.get("mutation", "swap"),
            )
        if model_name in ("EliteSingleGA", "EliteMultiGA"):
            parameters.update(
                elite_best=ga_config.get("elite_best", 0.1),
                elite_worst=ga_config.get("elite_worst", 0.3),
            )
        model_ga = model_types[model_name](**parameters, **kwargs)
        self.term_dict["max_epoch"] = max(
            self.term_dict["max_epoch"], ga_config.get("epoch", 1000)
        )
        g_best = model_ga.solve(
            self.problem_dict,
            termination=self.term_dict,
        )

        fitness_func_turn_inflow(g_best.solution, scenario_config=self.scenario_config)
        return g_best, model_ga

    def run_SA(self, **kwargs) -> tuple:
        """Run Simulated Annealing (SA) for behavior optimization.

        See Also:
            https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.physics_based.html#module-mealpy.physics_based.SA

        Warning:
            You can change the input parameters only from input_config.yaml file.

        Notes:
            SA input parameters:
                epoch (int): iterations. Defaults to 1000.
                pop_size (int): population size. Defaults to 2.
                temp_init (float): initial temperature. Defaults to 100.
                cooling_rate (float): Defaults to 0.99.
                scale (float): the change scale of initialization. Defaults to 0.1.
                sel_model (str): select diff. Defaults to "OriginalSA".

        Args:
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
        fitness_func_turn_inflow(g_best.solution, scenario_config=self.scenario_config)

        return (g_best, model_sa)

    def run_TS(self, **kwargs) -> tuple:
        """Run Tabu Search (TS) for behavior optimization.

        See Also:
            https://github.com/thieu1995/mealpy/blob/master/mealpy/math_based/TS.py

        Warning:
            You can change the input parameters only from input_config.yaml file.

        Notes:
            TS input parameters:
                epoch (int): iterations. Defaults to 1000.
                pop_size (int): population size. Defaults to 2.
                tabu_size (int): maximum size of tabu list. Defaults to 10.
                neighbour_size (int): size of the neighborhood for generating candidate solutions. Defaults to 10.
                perturbation_scale (float): scale of perturbation for generating candidate solutions. Defaults to 0.05.

        Args:
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
        # not print out log to console
        self.problem_dict["log_to"] = "None"
        g_best = model_ts.solve(self.problem_dict, termination=self.term_dict, starting_solutions=init_vals)

        # update files with the best solution
        fitness_func_turn_inflow(g_best.solution, scenario_config=self.scenario_config)

        return (g_best, model_ts)

    def run_BO(self) -> tuple:
        """Optimize with BO, then restore simulation files using the best solution."""
        from ._bayesian_opt import BayesianOptimization

        bounds = self.problem_dict["bounds"]
        model = BayesianOptimization(
            scenario_config=self.scenario_config,
            algo_config=self.turn_inflow_cfg,
            verbose=self.verbose,
            bounds=(bounds.lb, bounds.ub),
        )
        g_best = model.solve(self.fitness_func)
        self.fitness_func(g_best.solution.copy(), scenario_config=self.scenario_config)
        return g_best, model

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
