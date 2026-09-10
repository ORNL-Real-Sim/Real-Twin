##############################################################
# Created Date: Tuesday, September 8th 2026
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################

import os
from pathlib import Path
import json
import sqlite3
import subprocess
import time
import numpy as np
import pandas as pd
from functools import partial
from mealpy import FloatVar, SA, GA, TS
import re


def run_aconsole(cmd):
    """Capture script output even if Aimsun crashes during console shutdown."""
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONUNBUFFERED": "1"})
    out, _ = process.communicate()
    return process.returncode, out


def export_turn_inflow_info(input_config: dict | None = None, **kwargs):
    """Export the turning and inflow information from the Aimsun model.

    Args:
        input_config (dict): The configuration dictionary containing input parameters.
    Returns:
        None: The function does not return any value, but it generates the necessary files for calibration.
    """
    aconsole_path = input_config["AIMSUN"]["exe_path"]
    model_fname = input_config["AIMSUN"]["model_fname"]
    export_script_path = input_config["AIMSUN"]["aimsun_file"]["step6.1"]

    model_dir = Path(model_fname).parent
    info_path = model_dir / "calibration_info.json"

    input_config_serializable = {"AIMSUN": {"site_packages": input_config["AIMSUN"]["site_packages"]}}

    t0 = time.time()
    rc, out = run_aconsole([aconsole_path,
                            "-script",
                            export_script_path,
                            model_fname,
                            json.dumps(input_config_serializable)])

    # print(out)
    for line in out.splitlines():
        # if it's not log line, print it
        if not re.match(r"^\[[^\]]+\]\s*", line):
            print(f"{line}")

    if not (os.path.exists(info_path) and os.path.getmtime(info_path) >= t0 - 1):
        raise RuntimeError(f"calibration info export failed (return code {rc}) - "
                        "calibration_info.json was not written")
    if rc != 0:
        print(f"  :(aconsole exit code {rc} ignored - the export completed)")

    with open(info_path) as fh:
        info = json.load(fh)

    replication_id = info["replication_id"]
    field_df = pd.DataFrame(info["field_approaches"]).rename(
        columns={"count": "realcount"})
    inflow_sections = info["calib_inflow_sections"]

    # Group the calibration turnings by approach: one variable per turning,
    # normalized within its approach so the percentages sum to exactly 1
    approach_groups = []
    for turn in info["calib_turns"]:
        key = (turn["junction"], turn["from"])
        if not approach_groups or approach_groups[-1]["key"] != key:
            approach_groups.append({"key": key, "junction": turn["junction"],
                                    "from": turn["from"], "tos": []})
        approach_groups[-1]["tos"].append(turn["to"])

    n_turn_vars = sum(len(g["tos"]) for g in approach_groups)
    n_inflow_vars = len(inflow_sections)
    num_variables = n_turn_vars + n_inflow_vars

    print(f"  :replication id: {replication_id}")
    print(f"  :field approaches: {field_df['section'].tolist()}")
    print(f"  :{len(approach_groups)} approach group(s) ->{n_turn_vars} turning variable(s)")
    print(f"  :{n_inflow_vars} inflow variable(s): {inflow_sections}")
    print(f"  :total variables: {num_variables}")

    return {"replication_id": replication_id, "field_df": field_df, "approach_groups": approach_groups,
            "inflow_sections": inflow_sections,
            "n_turn_vars": n_turn_vars, "n_inflow_vars": n_inflow_vars, "num_variables": num_variables,
            "calibration_info_path": str(info_path)}


def assignNewTurn(solution, input_config: dict | None = None, **kwargs):
    """Decode a GA solution into turning-percentage and inflow tables."""

    INFLOW_MAX_VPH = input_config["Calibration"]["turn_inflow"]["max_inflow"]
    approach_groups = input_config["AIMSUN"]["turn_inflow"]["approach_groups"]
    n_turn_vars = input_config["AIMSUN"]["turn_inflow"]["n_turn_vars"]
    inflow_sections = input_config["AIMSUN"]["turn_inflow"]["inflow_sections"]

    turn_rows = []
    i = 0
    for group in approach_groups:
        k = len(group["tos"])
        vals = np.asarray(solution[i:i + k], dtype=float)
        i += k
        total = vals.sum()
        if total <= 0:
            vals = np.ones(k)
            total = float(k)
        pcts = vals / total * 100.0
        pcts[-1] = 100.0 - pcts[:-1].sum()    # force the sum to exactly 100
        for to_id, pct in zip(group["tos"], pcts):
            turn_rows.append((group["from"], to_id, pct))
    TurnDf = pd.DataFrame(turn_rows, columns=["entrance", "exit", "turn"])

    inflow_vals = np.asarray(
        solution[n_turn_vars:], dtype=float) * INFLOW_MAX_VPH
    InflowDf = pd.DataFrame({"entrance": inflow_sections,
                             "flow": np.round(inflow_vals).astype(int)})
    return TurnDf, InflowDf


def genDemand(solution, input_config: dict | None = None, **kwargs):

    aconsole_path = input_config["AIMSUN"]["exe_path"]
    model_fname = input_config["AIMSUN"]["model_fname"]
    assign_script_path = input_config["AIMSUN"]["aimsun_file"]["step6.2"]

    model_fname = input_config["AIMSUN"]["model_fname"]
    model_dir = Path(model_fname).parent
    turns_csv = model_dir / "calib_turns.csv"
    inflows_csv = model_dir / "calib_inflows.csv"

    TurnDf, InflowDf = assignNewTurn(solution, input_config=input_config, **kwargs)
    TurnDf.to_csv(turns_csv, index=False, header=False)
    InflowDf.to_csv(inflows_csv, index=False, header=False)

    # make input_config JSON serializable: keep only ["AIMSUN"]["site_packages"]
    input_config_serializable = {"AIMSUN": {"site_packages": input_config["AIMSUN"]["site_packages"]}}

    rc, out = run_aconsole([aconsole_path,
                            "-script",
                            assign_script_path,
                            model_fname,
                            json.dumps(input_config_serializable)])
    # Print out messages that are not log lines
    for line in out.splitlines():
        if not re.match(r"^\[[^\]]+\]\s*", line):
            print(f"{line}")

    # if ":Assigned" not in out:
    #     print(out)
    #     raise RuntimeError(f"demand assignment failed (return code {rc})")


def runAimsun(input_config: dict | None = None, **kwargs):
    """
    Clear this replication's old results, then execute it.

    Success is judged by fresh MISECT rows for this replication appearing in
    the sqlite output, not by the aconsole return code.
    """
    aconsole_path = input_config["AIMSUN"]["exe_path"]
    model_fname = input_config["AIMSUN"]["model_fname"]
    filename = Path(model_fname).stem
    model_dir = Path(model_fname).parent
    sqlite_path = os.path.join(model_dir, "Resources", "Outputs", f"{filename}.sqlite")
    info_path = model_dir / "calibration_info.json"

    with open(info_path) as fh:
        info = json.load(fh)

    replication_id = info["replication_id"]

    if os.path.exists(sqlite_path):
        try:
            con = sqlite3.connect(sqlite_path)
            con.execute("DELETE FROM MISECT WHERE did = ?", (replication_id,))
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

    deadline = time.time() + 30
    n_rows = 0
    while time.time() < deadline:
        if os.path.exists(sqlite_path):
            try:
                con = sqlite3.connect(sqlite_path)
                n_rows = con.execute("SELECT COUNT(*) FROM MISECT WHERE did = ?",
                                        (replication_id,)).fetchone()[0]
                con.close()
            except sqlite3.Error:
                n_rows = 0
        if n_rows > 0:
            break
        time.sleep(0.5)

    if n_rows == 0:
        print(out)
        raise RuntimeError(f"simulation produced no results (return code {rc})")


def resultAnalysis(input_config: dict | None = None, **kwargs):
    """Mean approach-level GEH over the whole simulation period."""

    model_fname = input_config["AIMSUN"]["model_fname"]
    model_dir = Path(model_fname).parent
    info_path = model_dir / "calibration_info.json"
    sqlite_path = os.path.join(model_dir, "Resources", "Outputs", f"{Path(model_fname).stem}.sqlite")
    calibration_target = input_config["Calibration"]["scenario_config"]["calibration_target"]

    with open(info_path) as fh:
        info = json.load(fh)

    replication_id = info["replication_id"]
    field_df = pd.DataFrame(info["field_approaches"]).rename(columns={"count": "realcount"})

    con = sqlite3.connect(sqlite_path)
    section0 = pd.read_sql_query(
        f"SELECT oid, count FROM MISECT WHERE did = {replication_id} AND sid = 0 AND ent = 0",
        con)
    con.close()
    section = section0.drop_duplicates(subset="oid", keep="last")
    compare = field_df.merge(section,
                             left_on="section",
                             right_on="oid",
                             how="left")
    compare = compare.dropna(subset=["count"])
    compare["GEH"] = np.sqrt(2 * ((compare["count"] - compare["realcount"]) ** 2)
                             / (compare["count"] + compare["realcount"]))
    meanGEH = compare["GEH"].mean()
    GEHPercent = (compare["GEH"] < calibration_target["GEH"]).mean()
    return meanGEH, GEHPercent


def fitness_func_turn_inflow_aimsun(solution: list | np.ndarray, input_config: dict | None = None, **kwargs) -> float:
    """
    Fitness function for Aimsun calibration.

    Args:
        solution (list | np.ndarray): The solution vector containing turn inflow and behavior parameters.
        input_config (dict): The configuration dictionary containing input parameters.
        **kwargs: Additional keyword arguments.

    Returns:
        float: The fitness value calculated based on the solution and scenario configuration.
    """

    genDemand(solution, input_config=input_config, **kwargs)

    runAimsun(input_config=input_config, **kwargs)

    meanGEH, GEHPercent = resultAnalysis(input_config=input_config, **kwargs)
    # # Report and save only when a better solution is found
    # if value < best_so_far[0]:
    #     best_so_far[0] = value
    #     print("eval %d: new best mean GEH = %.3f" % (eval_count[0], value))
    #     np.savetxt(best_txt_path, np.asarray(x, dtype=float), fmt="%f",
    #                 header="mean GEH = %.6f (evaluation %d)" % (value, eval_count[0]))
    print(f"  GEH: {meanGEH}, Percentage of GEH<5: {GEHPercent*100:.2f}%")
    return [meanGEH]


class TurnInflowCaliAimsun:
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

    def __init__(self, input_config: dict | None = None, verbose: bool = True, **kwargs):
        """Initialize the TurnInflowCalib class with scenario and turn inflow configurations."""

        self.input_config = input_config
        self.scenario_config = input_config.get("AIMSUN", {}).get("scenario_config", {})
        self.turn_inflow_cfg = input_config.get("AIMSUN", {}).get("turn_inflow", {})
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

        # turn_inflow_info = export_turn_inflow_info(input_config=input_config, **kwargs)
        # self.input_config["AIMSUN"]["turn_inflow"].update(turn_inflow_info)

        n_variable = self.input_config["AIMSUN"]["turn_inflow"].get("num_variables")
        n_inflow_variable = self.input_config["AIMSUN"]["turn_inflow"].get("n_inflow_vars")
        n_turn_variable = self.input_config["AIMSUN"]["turn_inflow"].get("n_turn_vars")
        max_inflow = self.turn_inflow_cfg.get("max_inflow", 200)  # max inflow for the inflow variables

        # fitness function for the optimization, default is fitness_func_turn_inflow
        self.fitness_func = fitness_func_turn_inflow_aimsun if kwargs.get("fitness_func") is None else kwargs.get("fitness_func")

        self.problem_dict = {
            "obj_func": partial(self.fitness_func, input_config=self.input_config),
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
        """Save the results of the optimization.

        See Also:
            https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.utils.html#module-mealpy.utils.history

        Args:
            output_dir (str): the directory to save the results.
            model: the optimized model object.
        """

        # check if output_dir exists
        os.makedirs(output_dir, exist_ok=True)
        if callable(getattr(model, "run_vis", None)):
            return model.run_vis(output_dir=output_dir)

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

            GA input parameters:
                epoch (int): the iterations. Defaults to 1000.
                pop_size (int): population size. Defaults to 50.
                pc (float): crossover probability. Defaults to 0.95.
                pm (float): mutation probability. Defaults to 0.025.
                ga_model (str): the type of GA model to use. Defaults to "BaseGA".
                    options: "BaseGA", "EliteSingleGA", "EliteMultiGA", "MultiGA", "SingleGA".

        See Also:
            https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.evolutionary_based.html#module-mealpy.evolutionary_based.GA

        Warning:
            You can change the input parameters only from input_config.yaml file.

        Args:
            **kwargs: additional keyword arguments for specific GA models.
        """
        if (ga_config := self.input_config["AIMSUN"]["turn_inflow"].get("ga_config")) is None:
            raise ValueError("  :Error: ga_config is not provided in the configuration file.")

        epoch = ga_config.get("epoch", 1000)  # max iterations
        pop_size = ga_config.get("pop_size", 50)  # population size
        # minimum population size for GA in mealpy is 10
        pop_size = max(pop_size, 10)

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
        # init_vals = self._generate_initial_solutions(self.init_solution, pop_size)

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
        self.term_dict["max_epoch"] = max(self.term_dict["max_epoch"], epoch)
        g_best = model_ga.solve(self.problem_dict, termination=self.term_dict)

        # update files with the best solution
        fitness_func_turn_inflow_aimsun(g_best.solution, input_config=self.input_config)

        return (g_best, model_ga)

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
        if (sa_config := self.input_config["AIMSUN"]["turn_inflow"].get("sa_config")) is None:
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
        fitness_func_turn_inflow_aimsun(g_best.solution, input_config=self.input_config)

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
        if (ts_config := self.input_config["AIMSUN"]["turn_inflow"].get("ts_config")) is None:
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
        fitness_func_turn_inflow_aimsun(g_best.solution, input_config=self.input_config)

        return (g_best, model_ts)

    def run_BO(self) -> tuple:
        """Optimize normalized turn weights/inflows and reapply the best candidate."""
        from realtwin.func_lib._f_calibration.algo_sumo._bayesian_opt import BayesianOptimization

        # assignNewTurn scales inflows by max_inflow and normalizes turn weights.
        n_variable = self.turn_inflow_cfg["num_variables"]
        model = BayesianOptimization(
            scenario_config=self.scenario_config,
            algo_config=self.turn_inflow_cfg,
            verbose=self.verbose,
            bounds=([0] * n_variable, [1] * n_variable),
            simulator="aimsun",
        )
        g_best = model.solve(
            self.fitness_func, objective_kwargs={"input_config": self.input_config})
        self.fitness_func(g_best.solution.copy(), input_config=self.input_config)
        return g_best, model

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

