##############################################################
# Created Date: Monday, September 7th 2026
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################


from pathlib import Path

from realtwin.func_lib._f_calibration._calibration_workflow import run_calibration_algorithm
from realtwin.rt_aimsun.cali_turn_inflow import TurnInflowCaliAimsun, export_turn_inflow_info
from realtwin.rt_aimsun.cali_behavior import BehaviorCaliAimsun


def cali_aimsun(
    *,
    sel_algo: dict | None = None,
    input_config: dict | None = None,
    verbose: bool = True,
    **kwargs,
) -> bool:
    """Run enabled Aimsun calibration stages in turn/inflow, then behavior order.

    Args:
        sel_algo: Lowercase algorithm names for turn_inflow and behavior.
            Defaults to GA. Unsupported names fall back to GA. The public
            RealTwinAimsun.calibrate method validates names case-insensitively.
        input_config: Loaded configuration with AIMSUN stage settings.
        verbose: Print optimizer progress.
        **kwargs: Forwarded to the turn/inflow metadata export.

    Returns:
        True after all enabled stages finish. Simulation errors propagate.
    """
    if sel_algo is None:  # use default algorithm if not provided
        sel_algo = {"turn_inflow": "ga", "behavior": "ga"}

    if not isinstance(sel_algo, dict):
        print(
            "  :Error:parameter sel_algo must be a dict with"
            " keys of 'turn_inflow' and 'behavior', using"
            " genetic algorithm as default values."
        )
        sel_algo = {"turn_inflow": "ga", "behavior": "ga"}

    # run calibration based on the selected algorithm: optimize turn and inflow
    if (
        input_config["AIMSUN"]["turn_inflow"]
        .get("turn_inflow", {})
        .get("is_calibration", False)
    ):
        print("\n  :Optimize Turn and Inflow...")
        turn_inflow_info = export_turn_inflow_info(input_config=input_config, **kwargs)
        input_config["AIMSUN"]["turn_inflow"].update(turn_inflow_info)
        turn_inflow = TurnInflowCaliAimsun(input_config=input_config, verbose=verbose)

        model, algorithm = run_calibration_algorithm(
            turn_inflow, sel_algo["turn_inflow"]
        )
        path_model_result = f"turn_inflow_{algorithm}_result"
        if algorithm == "bo":
            path_model_result = str(
                Path(input_config["AIMSUN"]["model_fname"]).parent / path_model_result
            )

        turn_inflow.run_vis(path_model_result, model)
        # clean up the temporary files generated during turn and inflow calibration
        turn_inflow._clean_up()
    else:
        print(
            "\n  :Turn and Inflow calibration is skipped in the input configuration file."
        )

    if (
        input_config["AIMSUN"]["behavior"]
        .get("behavior", {})
        .get("is_calibration", False)
    ):
        print(
            "\n  :Optimize Behavior parameters based on the optimized turn and inflow..."
        )
        if not input_config["AIMSUN"]["turn_inflow"].get("calibration_info_path"):
            turn_inflow_info = export_turn_inflow_info(
                input_config=input_config, **kwargs
            )
            input_config["AIMSUN"]["turn_inflow"].update(turn_inflow_info)
        behavior = BehaviorCaliAimsun(input_config=input_config, verbose=verbose)

        model, algorithm = run_calibration_algorithm(behavior, sel_algo["behavior"])
        path_model_result = f"behavior_{algorithm}_result"
        if algorithm == "bo":
            path_model_result = str(
                Path(input_config["AIMSUN"]["model_fname"]).parent / path_model_result
            )

        behavior.run_vis(path_model_result, model)
        behavior._clean_up()
    else:
        print("\n  :Behavior calibration is skipped in the input configuration file.")
    return True