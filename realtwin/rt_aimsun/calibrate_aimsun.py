##############################################################
# Created Date: Monday, September 7th 2026
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################


from pathlib import Path

from realtwin.rt_aimsun.cali_turn_inflow import TurnInflowCaliAimsun, export_turn_inflow_info
from realtwin.rt_aimsun.cali_behavior import BehaviorCaliAimsun


def cali_aimsun(*, sel_algo: dict | None = None, input_config: dict | None = None, verbose: bool = True, **kwargs) -> bool:
    """
    Run AIMSUN calibration based on the selected algorithm and input configuration.

        Args:
            sel_algo (dict): the dictionary of selected algorithm for turn_inflow and behavior. Defaults to None.
            input_config (dict): the dictionary contain configurations from input yaml file. Defaults to None.
            verbose (bool): print out processing message. Defaults to True.

        Raises:
            ValueError: if algo_config is not a dict with two levels with keys of 'ga', 'sa', 'ts', and 'bo'
            ValueError: if sel_algo is not a dict with keys of 'turn_inflow' and 'behavior'

        Returns:
            bool: True if calibration is successful, False otherwise.
    """
    # Test-driven Development: check selected algorithm from input
    if sel_algo is None:  # use default algorithm if not provided
        sel_algo = {"turn_inflow": "ga", "behavior": "ga"}

    if not isinstance(sel_algo, dict):
        print("  :Error:parameter sel_algo must be a dict with"
                " keys of 'turn_inflow' and 'behavior', using"
                " genetic algorithm as default values.")
        sel_algo = {"turn_inflow": "ga", "behavior": "ga"}

    # run calibration based on the selected algorithm: optimize turn and inflow
    if input_config["AIMSUN"]["turn_inflow"].get("turn_inflow", {}).get("is_calibration", False):
        print("\n  :Optimize Turn and Inflow...")
        turn_inflow_info = export_turn_inflow_info(input_config=input_config, **kwargs)
        input_config["AIMSUN"]["turn_inflow"].update(turn_inflow_info)
        turn_inflow = TurnInflowCaliAimsun(input_config=input_config, verbose=verbose)

        match sel_algo["turn_inflow"]:
            case "ga":
                g_best, model = turn_inflow.run_GA()
                path_model_result = "turn_inflow_ga_result"
            case "sa":
                g_best, model = turn_inflow.run_SA()
                path_model_result = "turn_inflow_sa_result"
            case "ts":
                g_best, model = turn_inflow.run_TS()
                path_model_result = "turn_inflow_ts_result"
            case "bo":
                g_best, model = turn_inflow.run_BO()
                path_model_result = str(
                    Path(input_config["AIMSUN"]["model_fname"]).parent / "turn_inflow_bo_result")
            case _:
                print(f"  :Error: unsupported algorithm {sel_algo['turn_inflow']}, using genetic algorithm as default.")
                g_best, model = turn_inflow.run_GA()
                path_model_result = "turn_inflow_ga_result"

        turn_inflow.run_vis(path_model_result, model)
        # clean up the temporary files generated during turn and inflow calibration
        turn_inflow._clean_up()
    else:
        print("\n  :Turn and Inflow calibration is skipped in the input configuration file.")

    if input_config["AIMSUN"]["behavior"].get("behavior", {}).get("is_calibration", False):
        print("\n  :Optimize Behavior parameters based on the optimized turn and inflow...")
        if not input_config["AIMSUN"]["turn_inflow"].get("calibration_info_path"):
            turn_inflow_info = export_turn_inflow_info(input_config=input_config, **kwargs)
            input_config["AIMSUN"]["turn_inflow"].update(turn_inflow_info)
        behavior = BehaviorCaliAimsun(input_config=input_config, verbose=verbose)

        match sel_algo["behavior"]:
            case "ga":
                g_best, model = behavior.run_GA()
                path_model_result = "behavior_ga_result"
            case "sa":
                g_best, model = behavior.run_SA()
                path_model_result = "behavior_sa_result"
            case "ts":
                g_best, model = behavior.run_TS()
                path_model_result = "behavior_ts_result"
            case "bo":
                g_best, model = behavior.run_BO()
                path_model_result = str(
                    Path(input_config["AIMSUN"]["model_fname"]).parent / "behavior_bo_result")
            case _:
                print(f"  :Error: unsupported algorithm {sel_algo['behavior']}, using genetic algorithm as default.")
                g_best, model = behavior.run_GA()
                path_model_result = "behavior_ga_result"

        behavior.run_vis(path_model_result, model)
        behavior._clean_up()
    else:
        print("\n  :Behavior calibration is skipped in the input configuration file.")
    return True