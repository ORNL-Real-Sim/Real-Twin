'''
##############################################################
# Created Date: Thursday, March 13th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


'''
##############################################################
# Created Date: Wednesday, February 26th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
from pathlib import Path
import pyufunc as pf
import copy

from realtwin.func_lib._f_calibration.algo_sumo_.calib_turn_inflow import TurnInflowCalib
from realtwin.func_lib._f_calibration.algo_sumo_.calib_behavior import BehaviorCalib


def prepare_scenario_config(input_config: dict) -> dict:
    """Prepare scenario_config from input_config"""

    scenario_config_dict = input_config.get("Calibration").get("scenario_config")

    # TODO : use dummy input dir for calibration in beta version, change in the future
    # # add input_dir to scenario_config from generated SUMO dir(scenario generation)
    # generated_sumo_dir = pf.path2linux(Path(input_config["output_dir"]) / "sumo")
    generated_sumo_dir = pf.path2linux(Path(__file__).parents[3] / "datasets/input_dir_dummy/")
    print(f"  :use dummy input dir: {generated_sumo_dir} for calibration in beta version")

    if Path(generated_sumo_dir).is_dir():
        scenario_config_dict["input_dir"] = generated_sumo_dir
    else:
        print(f"  :{generated_sumo_dir} is not a directory, pls run generate_concrete_scenario() first")
        return None

    # add network name to scenario_config and sim_name
    scenario_config_dict["network_name"] = input_config.get("Network").get("NetworkName")
    scenario_config_dict["sim_name"] = f"{input_config.get("Network").get("NetworkName")}.sumocfg"

    # TODO Do not copy generated files to generated_sumo_dir in beta version
#     # check whether required files exist in the input dir
#     required_files = {key: value for key, value in scenario_config_dict.items() if key.startswith("path_")}
#
#     if not pf.check_files_in_dir(required_files.values(), input_config.get("input_dir")):
#         return None
#
#     # copy required files to generated_sumo_dir
#     for key, file in required_files.items():
#         shutil.copy(Path(input_config["input_dir"]) / file, generated_sumo_dir)

    return scenario_config_dict


# for the beta version
def cali_sumo(*, sel_algo: dict = None, input_config: dict = None, verbose: bool = True) -> bool:
    """Run SUMO calibration based on the selected algorithm and input configuration.

    Args:
        sel_algo (dict): the dictionary of selected algorithm for turn_inflow and behavior. Defaults to None.
        input_config (dict): the dictionary contain configurations from input yaml file. Defaults to None.
        verbose (bool): print out processing message. Defaults to True.

    Raises:
        ValueError: if algo_config is not a dict with two levels with keys of 'ga', 'sa', and 'ts'
        ValueError: if sel_algo is not a dict with keys of 'turn_inflow' and 'behavior'

    Returns:
        bool: True if calibration is successful, False otherwise.
    """

    # Test-driven Development: check selected algorithm from input
    if sel_algo is None:  # use default algorithm if not provided
        sel_algo = {"turn_inflow": "ga",
                    "behavior": "ga"}

    if not isinstance(sel_algo, dict):
        print("  :Error:parameter sel_algo must be a dict with"
              " keys of 'turn_inflow' and 'behavior', using"
              " genetic algorithm as default values.")
        sel_algo = {"turn_inflow": "ga",
                    "behavior": "ga"}

    # Prepare scenario_config and algo_config from input_config
    scenario_config_turn_inflow = prepare_scenario_config(input_config)

    # Prepare Algorithm configure: e.g. {"ga": {}, "sa": {}, "ts": {}}
    algo_config = {selected_algo: input_config["Calibration"][f"{selected_algo}_config"]
                   for selected_algo in sel_algo.values()}

    # check algo_config with two levels
    if not all(isinstance(v, dict) for v in algo_config.values()):
        raise ValueError("  :algo_config must be a dict with two levels with keys of 'ga', 'sa', and 'ts'")
    # check whether configs are provided

    algo_turn_flow = {
        "ga": TurnInflowCalib,
        "sa": TurnInflowCalib,
        "ts": TurnInflowCalib}

    algo_behavior = {
        "ga": BehaviorCalib,
        "sa": BehaviorCalib,
        "ts": BehaviorCalib}

    # print(f"  : scenario_config: {scenario_config_turn_inflow}")
    # print(f"  : algo_config: {algo_config}")

    # run calibration based on the selected algorithm: optimize turn and inflow
    print("\n  :Optimize Turn and Inflow")
    turn_inflow = algo_turn_flow.get(sel_algo["turn_inflow"])(scenario_config_turn_inflow,
                                                              algo_config.get(sel_algo["turn_inflow"]),
                                                              verbose=verbose)
    turn_inflow.run_calibration()
    turn_inflow.run_vis()

    # run calibration based on the selected algorithm: optimize behavior
    # update path_turn and path_flow to generated xml files
    scenario_config_behavior = copy.deepcopy(scenario_config_turn_inflow)
    scenario_config_behavior["path_turn"] = f"{scenario_config_behavior.get("network_name")}.turn.xml"
    scenario_config_behavior["path_inflow"] = f"{scenario_config_behavior.get("network_name")}.flow.xml"
    print("\n  :Optimize Behavior parameters based on the optimized turn and inflow")
    behavior = algo_behavior.get(sel_algo["behavior"])(scenario_config_behavior,
                                                       algo_config.get(sel_algo["behavior"]),
                                                       verbose=verbose)
    behavior.run_calibration()
    behavior.run_vis()
    return True
