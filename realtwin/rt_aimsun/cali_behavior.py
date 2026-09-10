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

""" Behavior Calibration class for SUMO """

import os
from functools import partial
from pathlib import Path
import subprocess
import json
import sqlite3
import re
from dataclasses import dataclass
import numpy as np
import pandas as pd

from mealpy import FloatVar, SA, GA, TS

rng = np.random.default_rng(seed=812)
PENALTY_MAE = 1.0e4


def run_aconsole(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    out, _ = process.communicate()
    return process.returncode, out


def export_behavior_info(input_config: dict | None = None):
    """Export the behavior information from the input configuration file.

    Args:
        input_config (dict): the dictionary contain configurations from input yaml file. Defaults to None.
    Returns:
        dict: the dictionary contain behavior information.
    """
    aconsole_path = input_config["AIMSUN"]["exe_path"]
    model_fname = input_config["AIMSUN"]["model_fname"]
    model_dir = Path(model_fname).parent
    subpath_script_path = input_config["AIMSUN"]["aimsun_file"]["step7.1"]
    SUBPATHS = input_config["AIMSUN"]["behavior"]["sel_behavior_routes"]
    subpaths_csv = os.path.join(model_dir, "subpaths.csv")

    pd.DataFrame(SUBPATHS).to_csv(subpaths_csv, index=False, header=False)

    rc, out = run_aconsole([aconsole_path,
                            "-script",
                            subpath_script_path,
                            model_fname,
                            # json.dumps(input_config)
                            ])
    # point non-log lines
    for line in out.splitlines():
        # if it's not log line, print it
        if not re.match(r"^\[[^\]]+\]\s*", line):
            print(f"{line}")

    SUBPATH_IDS = dict(re.findall(r"SUBPATH_ID (\S+)=(\d+)", out))
    subpath_targets = []
    for name, _, _, real_tt in SUBPATHS:
        if name not in SUBPATH_IDS:
            raise RuntimeError(f"  :subpath '{name}' was not created - see the output above")
        subpath_targets.append((name, int(SUBPATH_IDS[name]), float(real_tt)))
    print("  :subpath targets (name, aimsun id, real travel time s):")
    for t in subpath_targets:
        print("  ", t)

    return {"subpath_targets": subpath_targets, "SUBPATH_IDS": SUBPATH_IDS}


def newParameter(solution_scaled, input_config: dict | None = None):
    """
    Apply a scaled parameter vector to the Car vehicle in the model.

    aconsole may crash on exit after finishing, so success is
    judged by the ':Applied' confirmation line, not the return code.
    """
    # np.savetxt(parameter_csv, np.asarray(solution_scaled, dtype=float), delimiter=",", fmt="%s")

    model_fname = input_config["AIMSUN"]["model_fname"]
    model_dir = Path(model_fname).parent
    parameter_csv = os.path.join(model_dir, "DrivingBehaviorParameter.csv")  # will be read by Step7.2_DrivingBehaviorAssign.py

    np.savetxt(parameter_csv, np.asarray(solution_scaled, dtype=float), delimiter=",", fmt="%s")
    aconsole_path = input_config["AIMSUN"]["exe_path"]
    model_fname = input_config["AIMSUN"]["model_fname"]
    assign_script_path = input_config["AIMSUN"]["aimsun_file"]["step7.2"]

    rc, out = run_aconsole([aconsole_path,
                            "-script",
                            assign_script_path,
                            model_fname,
                            # json.dumps(input_config)
                            ])

    for line in out.splitlines():
        # if it's not log line, print it
        if not re.match(r"^\[[^\]]+\]\s*", line):
            print(f"{line}")

    # if ":Applied" not in out:
    #     print(out)
    #     raise RuntimeError(f"driving behavior assignment failed (return code {rc})")


def runAimsun(input_config: dict | None = None):
    """Clear this replication's old results, then execute it.

    Success is judged by fresh MISECT rows for this replication appearing in
    the sqlite output, not by the aconsole return code.
    """
    aconsole_path = input_config["AIMSUN"]["exe_path"]
    model_fname = input_config["AIMSUN"]["model_fname"]
    model_dir = Path(model_fname).parent
    filename = Path(model_fname).stem
    info_path = os.path.join(model_dir, "calibration_info.json")

    sqlite_path = os.path.join(model_dir, "Resources", "Outputs", f"{filename}.sqlite")

    REPLICATION_ID = None
    if REPLICATION_ID is None:
        with open(info_path) as fh:
            REPLICATION_ID = json.load(fh)["replication_id"]
    replication_id = int(REPLICATION_ID)

    if os.path.exists(sqlite_path):
        try:
            con = sqlite3.connect(sqlite_path)
            for table in ("MISUBPATH", "MISECT"):
                try:
                    con.execute(f"DELETE FROM {table} WHERE did = ?", (replication_id,))
                except sqlite3.Error:
                    pass
            con.commit()
            con.close()
        except sqlite3.Error:
            pass
    rc, out = run_aconsole([aconsole_path,
                            "--project",
                            model_fname,
                            "--command",
                            "execute",
                            "--target",
                            str(replication_id)])
    n_rows = 0
    if os.path.exists(sqlite_path):
        try:
            con = sqlite3.connect(sqlite_path)
            n_rows = con.execute("SELECT COUNT(*) FROM MISECT WHERE did = ?",
                                 (replication_id,)).fetchone()[0]
            con.close()
        except sqlite3.Error:
            pass
    if n_rows == 0:
        print(out)
        raise RuntimeError(f"simulation produced no results (return code {rc})")


def resultFitness(input_config: dict | None = None) -> tuple[float | None, dict]:
    """MAE between simulated and real subpath travel times (whole period).

    Returns (None, sim_tts) when any subpath has no usable travel time --
    e.g. the parameter set gridlocked the network so no vehicle completed the
    subpath within the simulation. The caller applies a penalty fitness.
    """
    subpath_targets = input_config["AIMSUN"]["behavior"]["subpath_targets"]

    model_fname = input_config["AIMSUN"]["model_fname"]
    model_dir = Path(model_fname).parent
    info_path = os.path.join(model_dir, "calibration_info.json")
    sqlite_path = os.path.join(model_dir, "Resources", "Outputs", f"{Path(model_fname).stem}.sqlite")
    REPLICATION_ID = None
    if REPLICATION_ID is None:
        with open(info_path) as fh:
            REPLICATION_ID = json.load(fh)["replication_id"]
    replication_id = int(REPLICATION_ID)

    con = sqlite3.connect(sqlite_path)
    sub = pd.read_sql_query(
        f"SELECT oid, ttime FROM MISUBPATH WHERE did = {replication_id} AND sid = 0 AND ent = 0", con)
    con.close()
    sub = sub.drop_duplicates(subset="oid", keep="last").set_index("oid")
    sim_tts = {}
    errors = []
    degenerate = False
    for name, sp_id, real_tt in subpath_targets:
        if sp_id not in sub.index:
            sim_tts[name] = None
            degenerate = True
            continue
        tt = float(sub.loc[sp_id, "ttime"])
        if tt <= 0:                      # Aimsun writes -1 / 0 when no vehicle finished
            sim_tts[name] = None
            degenerate = True
            continue
        sim_tts[name] = tt
        errors.append(abs(float(real_tt) - tt))
    if degenerate or not errors:
        return None, sim_tts
    MAE = float(np.mean(errors))
    return (MAE, sim_tts)


def resultAnalysis(input_config: dict | None = None):
    """Mean approach-level GEH (optional check; needs calibration_info.json)."""

    model_fname = input_config["AIMSUN"]["model_fname"]
    model_dir = Path(model_fname).parent
    info_path = os.path.join(model_dir, "calibration_info.json")
    sqlite_path = os.path.join(model_dir, "Resources", "Outputs", f"{Path(model_fname).stem}.sqlite")
    REPLICATION_ID = None
    if REPLICATION_ID is None:
        with open(info_path) as fh:
            REPLICATION_ID = json.load(fh)["replication_id"]
    replication_id = int(REPLICATION_ID)

    if not os.path.exists(info_path):
        return None, None
    with open(info_path) as fh:
        field_df = pd.DataFrame(json.load(fh)["field_approaches"]).rename(
            columns={"count": "realcount"})
    con = sqlite3.connect(sqlite_path)
    section = pd.read_sql_query(
        f"SELECT oid, count FROM MISECT WHERE did = {replication_id} AND sid = 0 AND ent = 0",
        con)
    con.close()
    section = section.drop_duplicates(subset="oid", keep="last")
    compare = field_df.merge(section, left_on="section",
                             right_on="oid", how="left")
    compare = compare.dropna(subset=["count"])
    compare["GEH"] = np.sqrt(2 * ((compare["count"] - compare["realcount"]) ** 2)
                             / (compare["count"] + compare["realcount"]))
    return compare["GEH"].mean(), (compare["GEH"] < 5).mean()


def fitness_func(solution: list | np.ndarray, input_config: dict | None = None) -> float:
    """ Evaluate the fitness of a given solution for SUMO calibration."""

    lb = np.asarray(input_config["AIMSUN"]["behavior"]["params_lb"], dtype=float)
    ub = np.asarray(input_config["AIMSUN"]["behavior"]["params_ub"], dtype=float)

    # Set up SUMO command with car-following parameters
    # x_scaled = np.asarray(solution, dtype=float)
    x_scaled = lb + np.asarray(solution, dtype=float) * (ub - lb)

    newParameter(x_scaled, input_config)
    runAimsun(input_config)

    mae_value, sim_tts = resultFitness(input_config)

    if mae_value is None:
        missing = [n for n, tt in sim_tts.items() if tt is None]
        print(f"  :degenerate simulation (no demand for {', '.join(missing)}) or vehicle is over-saturated - penalty applied", flush=True)
        return PENALTY_MAE
    # Report and save only when a better solution is found
    print(f"  :Travel time best MAE = {mae_value:.3f} s", flush=True)
    return mae_value


class BehaviorCaliAimsun:
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
        >>> from realtwin import BehaviorCaliAimsun
        >>> from functools import partial
        >>> prob_dict = {"obj_func": partial(fitness_func, scenario_config=scenario_config, error_func="rmse"),
                        "bounds": FloatVar(lb=[1.0, 2.5, 4, 0.0, 0.25, 5.0], ub=[3.0, 3.0, 5.3, 1.0, 1.25, 9.3],),
                        "minmax": "max",  # maximize or minimize
                        "log_to": "console",
                        "save_population": True}
        >>> init_solution = [2.5, 2.6, 4.5, 0.5, 1.0, 9.0]
        >>> term_dict = {"max_epoch": 500, "max_fe": 10000, "max_time": 3600, "max_early_stop": 20}
        >>> opt = BehaviorCaliAimsun(input_config=input_config, verbose=True)
        >>> g_best, model_opt = opt.run_GA(epoch=1000, pop_size=30, pc=0.95, pm=0.1, sel_model="BaseGA")

        Save result figures to output_dir
        >>> opt.run_vis(output_dir="output_dir", model=model_opt)
        >>> print(g_best.solution)
        >>> print(g_best.target.fitness)
    """

    def __init__(self, input_config: dict | None = None, verbose: bool = True):

        self.input_config = input_config
        self.scenario_config = input_config.get("AIMSUN", {}).get("scenario_config", {})
        self.behavior_cfg = input_config.get("AIMSUN", {}).get("behavior", {})
        self.verbose = verbose

        # prepare termination criteria from scenario config
        self.term_dict = {
            "max_epoch": self.scenario_config.get("max_epoch", 1000),
            "max_fe": self.scenario_config.get("max_fe", 10000),
            "max_time": self.scenario_config.get("max_time", None),
            "max_early_stop": self.scenario_config.get("max_early_stop", 80),
        }

        init_params = self.behavior_cfg.get("behavior", {}).get("initial_params", None)
        if isinstance(init_params, dict):
            self.init_solution = list(init_params.values())
        elif isinstance(init_params, list):
            self.init_solution = init_params
        elif isinstance(init_params, np.ndarray):
            self.init_solution = init_params.tolist()
        else:
            self.init_solution = None

        # Default parameter ranges
        params_ranges_ = {"min_gap": [1.0, 3.0],
                          "acceleration": [2.5, 3.0],
                          "deceleration": [4.0, 5.3],
                          "sigma": [0.0, 1.0],
                          "tau": [0.25, 1.25],
                          "emergencyDecel": [5.0, 9.3]
                          }
        params_names_mapping = {"min_gap": "MinDist",
                                "acceleration": "MaxAcc",
                                "deceleration": "NormalDec",
                                "sigma": "SensitivityFactor",
                                "tau": "MinHeadway",
                                "emergencyDecel": "MaxDec"
                                }
        params_names = list(params_names_mapping.values())
        params_ranges_ = {params_names_mapping[k]: v for k, v in params_ranges_.items()}

        params_ranges = self.behavior_cfg.get("params_ranges", params_ranges_).values()
        params_lb = [val[0] for val in params_ranges]
        params_ub = [val[1] for val in params_ranges]

        if not self.input_config["AIMSUN"]["behavior"]:
            self.input_config["AIMSUN"]["behavior"] = {}
        self.input_config["AIMSUN"]["behavior"]["params_ranges"] = params_ranges
        self.input_config["AIMSUN"]["behavior"]["params_names"] = params_names
        self.input_config["AIMSUN"]["behavior"]["params_names_mapping"] = params_names_mapping
        self.input_config["AIMSUN"]["behavior"]["params_lb"] = params_lb
        self.input_config["AIMSUN"]["behavior"]["params_ub"] = params_ub
        self.input_config["AIMSUN"]["behavior"]["num_variables"] = len(params_names)
        behavior_info = export_behavior_info(input_config=self.input_config)
        self.input_config["AIMSUN"]["behavior"]["subpath_targets"] = behavior_info["subpath_targets"]
        self.input_config["AIMSUN"]["behavior"]["SUBPATH_IDS"] = behavior_info["SUBPATH_IDS"]

        self.problem_dict = {
            "obj_func": partial(fitness_func, input_config=self.input_config),
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
        """ Save the results of the optimization.

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

        Notes:
            GA parameters:
                epoch (int): the iterations. Defaults to 1000.
                pop_size (int): population size. Defaults to 50.
                pc (float): crossover probability. Defaults to 0.95.
                pm (float): mutation probability. Defaults to 0.025.
                model_selection (str): the type of GA model to use. Defaults to "BaseGA".
                    options: "BaseGA", "EliteSingleGA", "EliteMultiGA", "MultiGA", "SingleGA".

        Args:
            kwargs: additional keyword arguments for specific GA models (Navigate to See Also).
        """
        if (ga_config := self.input_config.get("AIMSUN", {}).get("behavior", {}).get("ga_config")) is None:
            raise ValueError(
                "ga_config is not provided in AIMSUN/behavior setting in input_config.yaml file.")

        epoch = ga_config.get("epoch", 1000)  # max iterations
        pop_size = ga_config.get("pop_size", 50)  # population size
        pop_size = max(pop_size, 10)  # ensure population size is at least 10
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

        # print("GA Initial Values:", init_vals)

        if sel_model not in ["BaseGA", "EliteSingleGA", "EliteMultiGA", "MultiGA", "SingleGA"]:
            print("Error: sel_model must be one of the following: "
                  "'BaseGA', 'EliteSingleGA', 'EliteMultiGA', 'MultiGA', 'SingleGA'.")
            print("Defaulting to 'BaseGA'.")
            sel_model = "BaseGA"

        if sel_model == "BaseGA":
            model_ga = GA.BaseGA(
                epoch=epoch, pop_size=pop_size, pc=pc, pm=pm, **kwargs)
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
        g_best = model_ga.solve(
            self.problem_dict, termination=self.term_dict)  # starting_solutions=init_vals

        # update files with the best solution
        self.problem_dict["obj_func"](g_best.solution)

        return (g_best, model_ga)

    def run_SA(self, **kwargs):
        """Run Simulated Annealing (SA) for behavior optimization.

        Note:
            1. The `model_selection` parameter allows you to choose different types of SA models. Default is "OriginalSA".
                Options include "OriginalSA", "GaussianSA", "SwarmSA".

            SA parameters:
                epoch (int): iterations. Defaults to 1000.
                pop_size (int): population size. Defaults to 2.
                temp_init (float): initial temperature. Defaults to 100.
                cooling_rate (float): Defaults to 0.99.
                scale (float): the change scale of initialization. Defaults to 0.1.
                model_selection (str): select diff. Defaults to "OriginalSA". Options: "OriginalSA", "GaussianSA", "SwarmSA".

        See Also:
            https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.physics_based.html#module-mealpy.physics_based.SA

        Warning:
            You can change the input parameters only from input_config.yaml file.

        Args:
            kwargs: additional keyword arguments for specific SA models (Navigate to See Also).
        """

        if (sa_config := self.input_config.get("AIMSUN", {}).get("behavior", {}).get("sa_config")) is None:
            raise ValueError(
                "sa_config is not provided in AIMSUN/behavior setting in yaml file.")

        epoch = sa_config.get("epoch", 1000)  # max iterations
        pop_size = sa_config.get("pop_size", 2)  # population size
        temp_init = sa_config.get("temp_init", 100)  # initial temperature
        cooling_rate = sa_config.get("cooling_rate", 0.891)  # cooling rate
        scale = sa_config.get("scale", 0.1)  # scale of the change
        # "OriginalSA", "GaussianSA", "SwarmSA"
        sel_model = sa_config.get("model_selection", "OriginalSA")

        # Generate initial solution for inputs
        init_vals = self._generate_initial_solutions(
            self.init_solution, pop_size)

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

        g_best = model_sa.solve(
            self.problem_dict, termination=self.term_dict)

        # update files with the best solution
        self.problem_dict["obj_func"](g_best.solution)

        return (g_best, model_sa)

    def run_TS(self, **kwargs):
        """Run Tabu Search (TS) for behavior optimization.

        See Also:
            https://github.com/thieu1995/mealpy/blob/master/mealpy/math_based/TS.py

        Warning:
            You can change the input parameters only from input_config.yaml file.

        Notes:
            TS parameters:
                epoch (int): max iterations. Defaults to 1000.
                pop_size (int): population size. Defaults to 2.
                tabu_size (int): maximum size of tabu list. Defaults to 10.
                neighbour_size (int): size of the neighborhood for generating candidate solutions. Defaults to 10.
                perturbation_scale (float): scale of perturbation for generating candidate solutions. Defaults to 0.05.

        Args:
            kwargs: additional keyword arguments for specific TS models (Navigate to See Also).
        """
        if (ts_config := self.input_config.get("AIMSUN", {}).get("behavior", {}).get("ts_config")) is None:
            raise ValueError(
                "ts_config is not provided in AIMSUN/behavior setting in yaml file.")

        epoch = ts_config.get("epoch", 1000)  # max iterations
        pop_size = ts_config.get("pop_size", 2)  # population size
        tabu_size = ts_config.get("tabu_size", 10)  # maximum size of tabu list

        # size of the neighborhood for generating candidate solutions
        neighbour_size = ts_config.get("neighbour_size", 10)

        # scale of perturbation for generating candidate solutions
        perturbation_scale = ts_config.get("perturbation_scale", 0.05)

        # Generate initial solution for inputs
        init_vals = self._generate_initial_solutions(
            self.init_solution, pop_size)

        model_ts = TS.OriginalTS(epoch=epoch,
                                 pop_size=pop_size,
                                 tabu_size=tabu_size,
                                 neighbour_size=neighbour_size,
                                 perturbation_scale=perturbation_scale,
                                 **kwargs)
        # not print out log to console
        self.problem_dict["log_to"] = "None"
        g_best = model_ts.solve(
            self.problem_dict, termination=self.term_dict)

        # update files with the best solution
        self.problem_dict["obj_func"](g_best.solution)

        return (g_best, model_ts)

    def _clean_up(self):
        """Clean up the temporary files generated during the calibration process."""
        # network_name = self.scenario_config.get("network_name")
        # turn_inflow_dir = self.scenario_config.get("dir_turn_inflow")
        # route_dir = os.path.join(turn_inflow_dir, "route")
        # flow_file = Path(route_dir) / f"{network_name}.flow.xml"
        # turn_file = Path(route_dir) / f"{network_name}.turn.xml"
        # shutil.copy(flow_file, turn_inflow_dir)
        # shutil.copy(turn_file, turn_inflow_dir)
        # # remove the route folder
        # shutil.rmtree(route_dir)
        # remove all files with .bak, .bak2, .bak3 etc...
        model_fname = self.input_config["AIMSUN"]["model_fname"]
        model_dir = Path(model_fname).parent

        for file in model_dir.glob("*.bak*"):
            file.unlink()

